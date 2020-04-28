import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import argparse
def nothing(x):
    print(x)
    
def resize(img,size):
    w_img = int(img.shape[1] * size / 75)
    h_img = int(img.shape[0] * size / 75)
    size_img = (w_img,h_img)
    img_resized = cv.resize(img, size_img, interpolation = cv.INTER_AREA)
    return img_resized

cap = cv.VideoCapture('cambada1.mov')
cv.namedWindow('Original Video vs Altered Video')

legend = '(0)all_obj  (1)all_balls (2)blue_robots (3)orange_balls (4)lines'
cv.createTrackbar(legend, 'Original Video vs Altered Video', 0, 4, nothing)

while(True):
	ret, frame = cap.read()
	frame = resize(frame,30)

	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	
	pos = cv.getTrackbarPos(legend, 'Original Video vs Altered Video')
	
    #all_balls
	lower_hsv_ball = np.array([12, 115, 90])
	higher_hsv_ball = np.array([40, 255, 255])
	mask_ball = cv.inRange(hsv, lower_hsv_ball, higher_hsv_ball)
	#blue_robots
	lower_hsv_blue = np.array([90, 80, 50])
	higher_hsv_blue = np.array([101, 255, 255])
	mask_blue = cv.inRange(hsv, lower_hsv_blue, higher_hsv_blue)
	#orange_balls
	lower_hsv_orange = np.array([0, 90, 0])
	higher_hsv_orange = np.array([20, 255, 196])
	mask_orange = cv.inRange(hsv, lower_hsv_orange, higher_hsv_orange)
	#lines
	lower_hsv_lines = np.array([0, 0, 164])
	higher_hsv_lines = np.array([180, 50, 255])
	mask_lines = cv.inRange(hsv, lower_hsv_lines, higher_hsv_lines)
	
	if pos == 0:
		mask  = mask_ball+mask_blue+mask_lines+mask_orange
		frame_filtered = cv.bitwise_or(frame, frame, mask=mask)
		frame_filtered = cv.cvtColor(frame_filtered, cv.COLOR_BGR2GRAY)
		frame_filtered = cv.Laplacian(frame_filtered, cv.CV_64F)
		frame_filtered = np.uint8(np.absolute(frame_filtered))  

	elif pos == 1:   
		frame_filtered = cv.bitwise_and(frame, frame, mask=mask_ball)
		frame_filtered = cv.cvtColor(frame_filtered, cv.COLOR_BGR2GRAY)
		frame_filtered = cv.Laplacian(frame_filtered, cv.CV_64F)
		frame_filtered = np.uint8(np.absolute(frame_filtered))       

	elif pos == 2:   
		frame_filtered = cv.bitwise_and(frame, frame, mask=mask_blue)
		frame_filtered = cv.cvtColor(frame_filtered, cv.COLOR_BGR2GRAY)
		frame_filtered = cv.Laplacian(frame_filtered, cv.CV_64F)
		frame_filtered = np.uint8(np.absolute(frame_filtered))   

	elif pos == 3:
		frame_filtered = cv.bitwise_and(frame, frame, mask=mask_orange)
		frame_filtered = cv.cvtColor(frame_filtered, cv.COLOR_BGR2GRAY)
		frame_filtered = cv.Laplacian(frame_filtered, cv.CV_64F)
		frame_filtered = np.uint8(np.absolute(frame_filtered))

	elif pos == 4:
		frame_filtered = cv.bitwise_and(frame, frame, mask=mask_lines)
		frame_filtered = cv.cvtColor(frame_filtered, cv.COLOR_BGR2GRAY)
		frame_filtered = cv.Laplacian(frame_filtered, cv.CV_64F)
		frame_filtered = np.uint8(np.absolute(frame_filtered))   
   

	# show thresholded image
	frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	numpy_horizontal = np.hstack((frame, frame_filtered))
	cv.imshow('Original Video vs Altered Video', numpy_horizontal)

	if(cv.waitKey(1) & 0xFF == ord('q')):
		break

