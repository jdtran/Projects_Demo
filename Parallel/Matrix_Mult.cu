/*
 =========================================================================
 Name        : MMShared.cu
 Author      : John Tran
 Version     : 27 April 2017
 Copyright   : None
 Description : CUDA compute reciprocals
 =========================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

#define TILE_WIDTH 16

//
// Matrix Multiplication CPU for error checking
//

void mymatrixmult(float *fa, float *fb, float *fc,
                  int aHight, int aWidth, int bWidth) {
	int row, col, k;
	int Pvalue=0;
	for (row=0; row<aHight; row++){
		for(col=0; col<bWidth; col++) {
         Pvalue = 0;
			for(k=0; k<aWidth; k++){
				Pvalue+=fa[row*aWidth+k]*fb[k*bWidth+col];
         }
         fc[row*bWidth+col]=Pvalue;
      }
	}
}

void matrixmult(float *fa, float *fb, float *fc,int Hight, int Width){
	int row, col, k;
	float Pvalue=0;
	for (row=0; row<Hight; row++){
		for(col=0; col<Width; col++) {
        	Pvalue=0;
			for(k=0; k<Width; k++){
				Pvalue+=fa[row*Width+k]*fb[k*Width+col];
            }
			fc[row*Width+col]=Pvalue;
         }
	}
}

//Compute C=A*B in GPU non shared memory
// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //identify row and column to work on
  	int row=blockIdx.y*blockDim.y+threadIdx.y;
  	int col=blockIdx.x*blockDim.x+threadIdx.x;
  	int i= row*numCColumns+col;
  	float Pvalue=0; int k;
  	if(row<numARows && col<numBColumns){
  		for(k=0; k<numBColumns; k++){
			Pvalue+=A[row*numAColumns+k]*B[k*numBColumns+col];
  		}
  		C[i]=Pvalue;
  	}
}


// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ TODO Insert code to implement matrix multiplication here
    //@@ TODO You have to use shared memory for this MP
    __shared__ float ms[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ns[TILE_WIDTH][TILE_WIDTH];
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int row = block_y * TILE_WIDTH + thread_y;
    int col = block_x * TILE_WIDTH + thread_x;
    int k = 0;
    float p_val = 0;

    for (int i = 0; i < (numAColumns - 1) / TILE_WIDTH + 1; ++i) {
        if (row < numARows && i * TILE_WIDTH + thread_x < numAColumns) {
            ms[thread_y][thread_x] = 
            A[row * numAColumns + i * TILE_WIDTH + thread_x];
        } else {
            ms[thread_y][thread_x] = 0;
        }
        if (col < numBColumns && i * TILE_WIDTH+thread_y < numBRows)  {
           ns[thread_y][thread_x] = 
           B[(i * TILE_WIDTH + thread_y) * numBColumns + col];
        } else {
           ns[thread_y][thread_x] = 0;
        }
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            p_val += ms[thread_y][k] * ns[k][thread_x];
        }
        __syncthreads();
    }
    if (row < numCRows & col < numCColumns) {
        C[row * numCColumns + col] = p_val;
    }
}

void initialize(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = .5*(i+1);
}

void printmm(float *m, int numCRows, int numCColumns) {
   int row = 0;
   int col = 0;
	for (row=0; row<numCRows; row++){
		for(col=0; col<numCColumns; col++) {
			printf("%lf ",m[row*numCColumns+col]);
		}
		printf("\n");
	}
}

void printsum(float *m, int total) {
   int i = 0;
   float sum = 0;
   
   for (i = 0; i < total; i++) {
      sum += m[i];
   }
   
   printf("sum: %5.2f\n", sum);
}

int main(int argc, char ** argv) {
   int numARows=1024; // number of rows in the matrix A
   int numAColumns=512; // number of columns in the matrix A
   int numBRows=512; // number of rows in the matrix B
   int numBColumns=1024; // number of columns in the matrix B
   int numCRows=numARows; // number of rows in the matrix C 
   int numCColumns=numBColumns; // number of columns in the matrix C 
   int ctotal = numCColumns * numCRows;
   cudaEvent_t n_start, n_stop, shared_start, shared_stop;
   //check if you can do the MM
   if(numAColumns != numBRows){
      printf("This matrix cannot be multiplied");
      return -1;
   }
   //alloc memory
   float *hostA = new float[numARows*numAColumns];
   initialize (hostA, numARows*numAColumns);
   float *hostB = new float[numBRows*numBColumns];
   initialize (hostB, numBRows*numBColumns);
   float * hostC=new float[numCRows*numCColumns];; // The output C matrix
   float time = 0;
   //do MM on CPU for timing
   mymatrixmult(hostA, hostB, hostC, numARows, numAColumns, numBRows);
   matrixmult(hostA, hostB, hostC, numARows, numAColumns);
   printf("CPU Reference ");
   printsum(hostC, ctotal);
   printf("\n");
   float * deviceA;
   float * deviceB;
   float * deviceC;

   float s_time = 0;
   float gpu_time = 0;

   //@@ Allocate GPU memory here
   unsigned int size_A = numARows * numAColumns;
   unsigned int mem_size_A = sizeof(float) * size_A;
   unsigned int size_B = numBRows * numBColumns;
   unsigned int mem_size_B = sizeof(float) * size_B;
   unsigned int size_C = numCRows * numCColumns;
   unsigned int mem_size_C = sizeof(float) * size_C;

   cudaMalloc((void**) &deviceA, mem_size_A);
   cudaMalloc((void**) &deviceB, mem_size_B);
   cudaMalloc((void**) &deviceC, mem_size_C);


   //@@ Copy memory to the GPU here
   cudaMemcpy(deviceA, hostA, mem_size_A, cudaMemcpyHostToDevice) ;
   cudaMemcpy(deviceB, hostB, mem_size_B, cudaMemcpyHostToDevice) ;
   
   cudaEventCreate(&n_start);
   cudaEventCreate(&n_stop);

   //@@ Initialize the grid and block dimensions here
   dim3 threads(TILE_WIDTH, TILE_WIDTH,1); //TODO change to correct values
   dim3 grid((numCColumns-1)/TILE_WIDTH+1, (numCRows-1)/TILE_WIDTH+1, 1);
   //MM with shared memory
   cudaEventRecord(n_start);
   matrixMultiply<<< grid, threads>>>(deviceA, deviceB, deviceC,
                                      numARows, numAColumns, 
                                      numBRows, numBColumns, 
                                      numCRows, numCColumns);
   cudaEventRecord(n_stop);
   cudaEventSynchronize(n_stop);
   cudaEventElapsedTime(&time, n_start, n_stop);
   cudaDeviceSynchronize();
   //@@ Copy the GPU memory back to the CPU here
   cudaMemcpy(hostC, deviceC, mem_size_C, cudaMemcpyDeviceToHost) ;
   print MM result
   printf("Non-Shared gpu ");
   printsum(hostC, numCColumns * numCRows);
   printf("\n");
   
   cudaEventCreate(&shared_start);
   cudaEventCreate(&shared_stop);
   cudaEventRecord(shared_start);
   matrixMultiplyShared<<< grid, threads>>>(deviceA, deviceB, deviceC,
                                            numARows, numAColumns, 
                                            numBRows, numBColumns, 
                                            numCRows, numCColumns);
   cudaEventRecord(shared_stop);
   //cudaEventSynchronize(shared_stop);
   cudaEventElapsedTime(&s_time, shared_start, shared_stop);
   cudaThreadSynchronize();
    	
	//@@ Copy the GPU memory back to the CPU here
	cudaMemcpy(hostC, deviceC, mem_size_C, cudaMemcpyDeviceToHost) ;

   //verify result  all elements added
   //accum of error and if less than xxx
   //print MM result
   printf("Shared gpu ");
   printsum(hostC, numCColumns * numCRows);
   
   printf("\n");
   printf("Total GPU time (non shared mem) %lf\n", time);
   printf("\nTotal Shared memory GPU Time: %lf\n\n", s_time);
   //@@ Free the GPU memory here
   cudaFree(deviceA);
   cudaFree(deviceB);
   cudaFree(deviceC);

   free(hostA);
   free(hostB);
   free(hostC);

   return 0;
}