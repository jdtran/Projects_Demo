#!/usr/bin/env python

import cv2, math
import numpy as np
from matplotlib import pyplot as plt

MEDIAN_FILTER_DIM = 5  # dimension of the median filter
RECENT_FRAMES = 10     # number of frames to look back by

def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def main():
   video = cv2.VideoCapture("a1.mp4")

   i = 0

   handFrames = []
   while video.isOpened():
      ret, frame = video.read()

      if not ret:
         break

      # convert to greyscale
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      print('{} {}'.format(i, ret))
      i += 1

      # apply median filter
      med = cv2.medianBlur(gray, MEDIAN_FILTER_DIM)

      # apply otsu's thresholding to get binary image of hand pixels
      ret, otsu = cv2.threshold(med, 0, 255, 
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      otsu = np.array(otsu == 255, dtype=int)
      handFrames.append(otsu)

      # reduce search space of heat trace to be where the hand has 
      # recently traveled
      start = i-RECENT_FRAMES if i-RECENT_FRAMES >= 0 else 0
      recent = np.logical_or.reduce(handFrames[start:])
      recent = np.array(recent, dtype=int)

      # format search space so it works with OpenCV functions
      searchSpace = np.subtract(recent, otsu) 
      searchSpace[searchSpace == 1] = 255
      searchSpace = np.array(searchSpace, dtype=np.uint8)

      if i == 30:
         searchImg = cv2.bitwise_and(med,med, mask=searchSpace)
         plt.imshow(searchImg, cmap='gray')
         plt.show()
         #cv2.startWindowThread()
         #cv2.namedWindow("preview")
         #cv2.imshow(searchImg)

   video.release()

if __name__ == "__main__":
   main()