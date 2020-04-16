import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def resize(img,size):
    w_img = int(img.shape[1] * size / 75)
    h_img = int(img.shape[0] * size / 75)
    size_img = (w_img,h_img)
    img_resized = cv.resize(img, size_img, interpolation = cv.INTER_AREA)
    return img_resized
    
cap = cv.VideoCapture('cambada.mp4')

while(True):
	ret, frame = cap.read()
	frame=resize(frame,20)
	frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	#thresholded
	laplacian = cv.Laplacian(frame,cv.CV_64F)
	sobelx = cv.Sobel(frame,cv.CV_64F,1,0,ksize=5)
	sobely = cv.Sobel(frame,cv.CV_64F,0,1,ksize=5)
	scharrx64 = cv.Scharr(frame,cv.CV_8U,1,0)
	scharry64 = cv.Scharr(frame,cv.CV_8U,0,1)
	
	# show thresholded video
	cv.imshow('LaPlacian', laplacian)            
	cv.imshow('SobelX', sobelx)
	cv.imshow('SobelY', sobely)
	cv.imshow('ScharrX', scharrx64)
	cv.imshow('ScharrY', scharry64)
	if(cv.waitKey(1) & 0xFF == ord('q')):
		break
