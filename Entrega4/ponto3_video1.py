import cv2 
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    print(x)

def resize(image,scl):
    w_image = int(image.shape[1] * scl / 100)
    h_image = int(image.shape[0] * scl / 100)
    dim_image = (w_image,h_image)
    image_resized = cv2.resize(image, dim_image, interpolation = cv2.INTER_AREA)
    return image_resized

cap = cv2.VideoCapture('cambada1.mov')
cv2.namedWindow('Original Video vs Altered Video')

legend = '(0)obj_all  (1)all_balls (2)blue_robots (3)orange_balls (4)lines'
cv2.createTrackbar(legend, 'Original Video vs Altered Video', 0, 4, nothing)

while(True):
	ret, frame = cap.read()
	frame = resize(frame,40)

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	pos = cv2.getTrackbarPos(legend, 'Original Video vs Altered Video')

	#balls
	lower_hsv_ball = np.array([25, 115, 85])
	higher_hsv_ball = np.array([40, 255, 255])
	mask_ball = cv2.inRange(hsv, lower_hsv_ball, higher_hsv_ball)
	#blue_robots
	lower_hsv_blue = np.array([90, 80, 45])
	higher_hsv_blue = np.array([100, 255, 255])
	mask_blue = cv2.inRange(hsv, lower_hsv_blue, higher_hsv_blue)
	#orange_balls
	lower_hsv_orange = np.array([0, 90, 0])
	higher_hsv_orange = np.array([20, 255, 195])
	mask_orange = cv2.inRange(hsv, lower_hsv_orange, higher_hsv_orange)
	#lines
	lower_hsv_lines = np.array([0, 0, 160])
	higher_hsv_lines = np.array([180, 50, 255])
	mask_lines = cv2.inRange(hsv, lower_hsv_lines, higher_hsv_lines)
	
	if pos ==0:
		mask  = mask_ball+mask_blue+mask_lines+mask_orange
		frame_filtered = cv2.bitwise_or(frame, frame, mask=mask)
		frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
		frame_filtered = cv2.Laplacian(frame_filtered, cv2.CV_64F)
		frame_filtered = np.uint8(np.absolute(frame_filtered))  

	elif pos==1:   
		frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_ball)
		frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
		frame_filtered = cv2.Laplacian(frame_filtered, cv2.CV_64F)
		frame_filtered = np.uint8(np.absolute(frame_filtered))       

	elif pos==2:   
		frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_blue)
		frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
		frame_filtered = cv2.Laplacian(frame_filtered, cv2.CV_64F)
		frame_filtered = np.uint8(np.absolute(frame_filtered))   

	elif pos==3:
		frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_orange)
		frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
		frame_filtered = cv2.Laplacian(frame_filtered, cv2.CV_64F)
		frame_filtered = np.uint8(np.absolute(frame_filtered))

	elif pos==4:
		frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_lines)
		frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
		frame_filtered = cv2.Laplacian(frame_filtered, cv2.CV_64F)
		frame_filtered = np.uint8(np.absolute(frame_filtered))   

	# show thresholded image
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	numpy_horizontal = np.hstack((frame, frame_filtered))
	cv2.imshow('Original Video vs Altered Video', numpy_horizontal)

	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break
