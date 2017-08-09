#!/usr/bin/env python

import cv2, math
import numpy as np
from matplotlib import pyplot as plt

def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def main():
   video = cv2.VideoCapture("b2.mp4")
   ret = True
   i = 0

   while video.isOpened():
      ret, frame = video.read()

      if not ret:
         break

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      print('{} {}'.format(i, ret))
      i += 1

      if (i == 30):
         # output file name
         outName = "seqB2" + str(i) + ".png"
         
         # apply median filter
         med = cv2.medianBlur(gray, 5)

         # apply otsu's
         ret, otsu = cv2.threshold(med, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
         
         # find contours (note: this modifies otsu so we create a copy)
         image, contours, _ = cv2.findContours(np.copy(otsu), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

         # find contour with largest area, a.k.a. the hand
         handContour = max(contours, key=lambda c:cv2.contourArea(c))

         # find centroid
         M = cv2.moments(handContour)
         cx = int(M['m10']/M['m00'])
         cy = int(M['m01']/M['m00'])
         centroid = [cx, cy]

         # compute convex hull 
         hull = cv2.convexHull(handContour)

         # find fingertip
         ft = max(hull, key=lambda h:distance(h[0], centroid))[0]
         
         # display image and plot points above it
         plt.imshow(gray, cmap='gray'),plt.title("Finding fingertip")
         plt.plot([centroid[0]], [centroid[1]], 'bo')
         plt.plot([ft[0]], [ft[1]], 'ro')
         plt.xticks([]), plt.yticks([])
         
         # save the figure to file
         plt.savefig(outName)

   video.release()

if __name__ == "__main__":
   main()