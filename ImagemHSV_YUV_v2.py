import cv2
import numpy as np

img = cv2.imread('image.jpg',1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # BGR -> Gray
cv2.imshow('image1',gray)
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)        # BGR -> HSV
cv2.imshow('image2',hsv)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)

yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)        # BGR -> YUV
cv2.imshow('image3',yuv)
cv2.namedWindow('image3', cv2.WINDOW_NORMAL)

cv2.waitKey(0)
cv2.destroyAllWindows()
