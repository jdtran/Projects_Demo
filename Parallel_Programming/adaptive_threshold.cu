/*
 =========================================================================
 Name        : adaptive_threshold.cu
 Author      : Vincy Chow, John Tran
 Version     : 10 June 2017
 Copyright   : None
 Description : CUDA compute reciprocals
 =========================================================================
 */
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <numeric>
#include <stdlib.h>

#define BLOCK_SIZE 16
#define HIST_SIZE 256

 // Windows
#ifdef _WIN32
#include <Windows.h>

double get_wall_time() {
   LARGE_INTEGER time, freq;
   if (!QueryPerformanceFrequency(&freq)) {
      //  Handle error
      return 0;
   }
   if (!QueryPerformanceCounter(&time)) {
      //  Handle error
      return 0;
   }
   return (double)time.QuadPart / freq.QuadPart;
}
double get_cpu_time() {
   FILETIME a, b, c, d;
   if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0) {
      //  Returns total user time.
      //  Can be tweaked to include kernel times as well.
      return
         (double)(d.dwLowDateTime |
         ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
   }
   else {
      //  Handle error
      return 0;
   }
}

// Posix/Linux
#else
#include <time.h>
#include <sys/time.h>
double get_wall_time() {
   struct timeval time;
   if (gettimeofday(&time, NULL)) {
      //  Handle error
      return 0;
   }
   return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time() {
   return (double)clock() / CLOCKS_PER_SEC;
}
#endif

cv::Mat imageRGBA;
cv::Mat imageGrey;
cv::Mat image;
uchar4 *d_rgbaImage__;

unsigned char *d_greyImage__;
size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }
const long numPixels = numRows() * numCols();

template<typename _Tp> static inline _Tp saturate_cast(int v) 
{ return _Tp(v); }

int *hist_cpu;
int *hist_gpu;

unsigned char *bw_cpu;
unsigned char *bw_gpu;
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const std::string &filename) {
   //make sure the context initializes ok
   cudaFree(0);
   //cv::Mat image;
   image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
   if (image.empty()) {
      std::cerr << "Couldn't open file: " << filename << std::endl;
      exit(1);
   }

   cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);
   //allocate memory for the output
   imageGrey.create(image.rows, image.cols, CV_8UC1);
   
   //This shouldn't ever happen given the way the images are created
   //but better to check
   if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
      std::cerr << "Images aren't continuous!! Exiting." << std::endl;
      exit(1);
   }
   *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
   *greyImage  = imageGrey.ptr<unsigned char>(0);
   const size_t numPixels = numRows() * numCols();

   cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels);
   cudaMalloc(d_greyImage, sizeof(char) * numPixels);
   cudaMemset(greyImage, 0, numPixels * sizeof(unsigned char));
   
   //copy input array to the GPU inputInage to d_rgbaImage
   cudaMemcpy(*d_rgbaImage, *inputImage, 
              sizeof(uchar4) * numPixels,cudaMemcpyHostToDevice);
   d_rgbaImage__ = *d_rgbaImage;
   d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file) {
   const int numPixels = numRows() * numCols();
   cudaMemcpy(imageGrey.ptr<unsigned char>(0), 
              d_greyImage__, 
              sizeof(unsigned char) * numPixels, 
              cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();
   
   //output the image
   cv::imwrite(output_file.c_str(), imageGrey);
}

void postProcess_BW() {
   const int numPixels = numRows() * numCols();
   cudaMemcpy(imageGrey.ptr<unsigned char>(0), 
              bw_gpu, sizeof(unsigned char) * numPixels, 
              cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();

   cv::imwrite("threshold.jpg", imageGrey);
   cv::imshow ("Output Image", imageGrey);
   ////cleanup
   cudaFree(d_rgbaImage__);
   cudaFree(d_greyImage__);
   cudaFree(hist_gpu);
   cudaFree(bw_gpu);
   free(bw_cpu);
}

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                  unsigned char* const greyImage,
                  int numRows, int numCols) {
   //Fill in the kernel to convert from color to greyscale
   //the mapping from components of a uchar4 to RGBA is:
   // .x -> R ; .y -> G ; .z -> B ; .w -> A
   //
   //The output (greyImage) at each pixel should be the result of
   //applying the formula: output = .299f * R + .587f * G + .114f * B;
   //Note: We will be ignoring the alpha channel for this conversion
   //First create a mapping from the 2D block and grid locations
   //to an absolute 2D location in the image, then use that to
   //calculate a 1D offset.

   int y = blockIdx.y * blockDim.y + threadIdx.y; // row Number
   int x = blockIdx.x * blockDim.x + threadIdx.x; // col Number
   if (x >= numCols || y >= numRows) {
      return;
   }
   uchar4 rgba = rgbaImage[x + y * numCols];
   greyImage[x + y * numCols] = 
            (.299f * rgba.x + .587f * rgba.y + .114f * rgba.z);
}

__global__
void threshold(unsigned char* const greyImage, unsigned char* bw,
           int numRows, int numCols, int threshVal) {

  int row= blockIdx.y * blockDim.y + threadIdx.y;  // row Number
  int col = blockIdx.x * blockDim.x + threadIdx.x; // col Number
  
  if( row < numRows && col < numCols ){
      int i = row*numCols+col;
      unsigned char pixel = greyImage[i];
      
      if (pixel <= threshVal) {
         bw[i] = 0;
      }
      else {
         bw[i] = 255;
      }
  }
}

__global__ void histogram(unsigned char* const greyImage, int *hist, 
                          int numRows, int numCols) {
   //TODO write the create histogram on GPU function
   // Pixel Location
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   int t = threadIdx.x + threadIdx.y * blockDim.x;
   unsigned char val = 0;
   __shared__ unsigned int localHist[1024];
   
   localHist[t] = 0;
   __syncthreads();
   //Have every thread add to their respective value
   if( row < numRows && col < numCols ){
      int i = row*numCols+col;
      //Thread's invidual pixel value
      val = greyImage[i];
      __syncthreads();
      atomicAdd(&localHist[val], 1);
   }
   //Combine fragments into the global memory
   __syncthreads();
   if (t < 256) {
      atomicAdd(&hist[t], localHist[t]);
   }
   __syncthreads();
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage,
                            uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage,
                            size_t numRows,
                            size_t numCols) {
   //TODO allocate mem for hist in cpu and for gpu
   unsigned int hist_size = 256;
   const size_t hs = size_t(hist_size) * sizeof(int);
   hist_cpu = (int *)malloc(hist_size * sizeof(int));
   memset(hist_cpu, 0, hs);
   cudaMalloc(&hist_gpu, hist_size * sizeof(int));
   cudaMemcpy(hist_gpu, hist_cpu, hist_size*sizeof(int), 
              cudaMemcpyHostToDevice);
   
   int threadSize=32;
   int gridSizeY=(numRows + (threadSize - 1))/threadSize ;
   int gridSizeX=(numCols + (threadSize - 1))/threadSize;  
   const dim3 blockSize(threadSize, threadSize, 1);   
   const dim3 gridSize(gridSizeX, gridSizeY, 1);
   //call kernel to trsnafrom to gray
   rgba_to_greyscale<<<gridSize, blockSize>>>
                    (d_rgbaImage,d_greyImage,numRows,numCols);
   cudaDeviceSynchronize();
   histogram<<<gridSize, blockSize, hist_size * sizeof(int)>>>
            (d_greyImage,hist_gpu, numRows, numCols);
   cudaDeviceSynchronize();
   cudaMemcpy(hist_cpu, hist_gpu, sizeof(int) * 256, 
              cudaMemcpyDeviceToHost);
}

void threshold_setup(unsigned char* const d_greyImage, int idx, 
                     size_t numRows, size_t numCols) {
   int numThreads = 32;
   int gridSizeY = (numRows + (numThreads - 1)) / numThreads;
   int gridSizeX = (numCols + (numThreads - 1)) / numThreads;
   int pixelCount = numCols * numRows;
   bw_cpu = (unsigned char *) malloc(pixelCount * sizeof(char));

   memset(bw_cpu, 0, sizeof(unsigned char) * pixelCount);
   cudaMalloc(&bw_gpu, sizeof(unsigned char) * pixelCount);
   cudaMemset(bw_gpu, 0, sizeof(unsigned char) * pixelCount);
   
   const dim3 blockSize(numThreads, numThreads, 1);
   const dim3 gridSize(gridSizeX, gridSizeY, 1);
   //call kernel to trsnafrom to gray
   threshold<<<gridSize, blockSize>>>(d_greyImage, bw_gpu, 
                                      numRows, numCols, idx);
   printf("Converted to black and white\n");
   cudaDeviceSynchronize();
   
   cudaError_t error = cudaGetLastError();
   if(error != cudaSuccess){
      // print the CUDA error message and exit
      printf("Assert CUDA error on line %d: %s\n",
             __LINE__,cudaGetErrorString(error));
      exit(-1);
   }
}

void Diff(int *vector, int *derivative, int numElements) {
   int i = 0;
   
   for (i = 0; i < numElements - 1; i++) {
      derivative[i] = vector[i+1] - vector[i]; 
   }
}

int *ZeroCrossings(int *firstOrder, int *secondOrder, int numElements) {
   int i = 0;
   int max[2] = {0};
   static int idx[2] = {0};
   
   // Zero crossings are signified by a sign change in the first order
   // This can be found by turning positive numbers to 1 and everything
   // else to 0 then taking the second derivative to see non-zero elements
   for (i = 0; i < numElements; i++) {
      if (firstOrder[i] > 1) {
         secondOrder[i] = 1;
      }
      else {
         secondOrder[i] = 0;
      }
   }
   
   int *crossings = (int *) malloc(sizeof(int) * HIST_SIZE);
   Diff(secondOrder, crossings, numElements);

   // Find the top two maximas
   for (i = 0; i < numElements; i++) {
      if (crossings[i] != 0) {
         if (firstOrder[i] > max[0]) {
            max[0] = firstOrder[i];
            idx[0] = i;
         }
         else if (firstOrder[i] > max[1]) {
            max[1] = firstOrder[i];
            idx[1] = i;
         }
      }
   }
      
   return idx;
}

int main(int argc, char **argv) {
   cudaDeviceReset();
   uchar4        *h_rgbaImage, *d_rgbaImage;
   unsigned char *h_greyImage, *d_greyImage;
   std::string input_file;
   std::string output_file;

   if (argc == 3) {
      input_file  = std::string(argv[1]);
      output_file = std::string(argv[2]);
   }
   else {
      std::cerr << "Usage: ./hw input_file output_file" << std::endl;
      exit(1);
   }

   double cpu0 = get_cpu_time();

   //load the image and give us our input and output pointers
   preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, 
              &d_greyImage, input_file);
   //call the students' code
   your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, 
                          d_greyImage, numRows(), numCols());
   cudaDeviceSynchronize();
   cudaGetLastError();
   printf("\n");
   postProcess(output_file);
   
   int a = 0;
   int temp = 0;
   // Strange bug in cuda where some pixel values got double counted
   for (a = 0; a < HIST_SIZE - 1; a++) {
      if (hist_cpu[a] == 0) {
         temp = hist_cpu[a - 1] / 2;
         hist_cpu[a] = temp;
         hist_cpu[a-1] = temp;
      }
   }
   
   int *firstOrder, *secondOrder; 
   firstOrder = (int *)malloc(sizeof(int) * HIST_SIZE);
   secondOrder = (int *)malloc(sizeof(int) * HIST_SIZE);
   memset(firstOrder, 0, sizeof(int) * HIST_SIZE);
   memset(secondOrder, 0, sizeof(int) * HIST_SIZE);
   Diff(hist_cpu, firstOrder, HIST_SIZE);
   
   int *max, avg = 0;
   max = ZeroCrossings(firstOrder, secondOrder, HIST_SIZE);
   avg = (max[0] + max[1]) / 2;
   printf("\nPeaks %d %d\n", max[0], max[1]);
   printf("Thresh Value %d\n", avg);
   threshold_setup(d_greyImage, avg, numRows(), numCols());
   postProcess_BW();
   double cpu1 = get_cpu_time();
   printf("CPU Time  = %lf\n", cpu1 - cpu0);
   cudaThreadExit();
   free(hist_cpu);

  return 0;
}
