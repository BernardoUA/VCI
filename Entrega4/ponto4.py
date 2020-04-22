import cv2
import numpy as np

img = cv2.imread('2.png',0)
gray_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)	# Convert to gray-scale
lower_red=np.array([169,100,100])
upper_red=np.array([189,255,255])
mask=cv2.inRange(gray_img,lower_red,upper_red)

blured = cv2.medianBlur(img,3)					# Blur the image to reduce noise


circles = cv2.HoughCircles(blured,cv2.HOUGH_GRADIENT,1,img.shape[0]/64,param1=300,param2=40,minRadius=0,maxRadius=0)  # image, dp, minDistance, minRadious and maxRadious

while True:

	image_cercle = np.uint16(np.around(circles)) 

	for pt in image_cercle[0,0:2]:

		#centre_x,centre_y,rayon=pt[0],pt[1],pt[2]
		cv2.circle(gray_img,(pt[0],pt[1]),pt[2],(0,255,255),4)
		cv2.circle(gray_img,(pt[0],pt[1]),1,(255,0,0),3)
		# show image
		cv2.namedWindow('image mask ', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('image mask ', mask)
		cv2.namedWindow('color image', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('color image', gray_img)
		cv2.waitKey(1)
cv2.imshow('detected circles',gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
