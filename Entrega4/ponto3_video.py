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
	frame = resize(frame,20)

	framegray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	cv.imshow('Original Frame',frame)						# show original frame
	edges = cv.Canny(framegray,100,180)						# Second and third arguments are our minVal and maxVal respectively ( Sobel kernel used for find image gradient)
	contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)			# approx_simple ->  It removes all redundant points and compresses the contour, thereby saving memory.
	cnt = contours[5]
	cv.drawContours(frame, [cnt], 0 , (0,255,0), 2,cv.LINE_8)		# draw the countours cv.drawContours(	image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]	)
	cv.imshow('Frame with contours', frame)								

	if(cv.waitKey(1) & 0xFF == ord('q')):
		break
