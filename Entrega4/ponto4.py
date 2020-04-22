import cv2
import numpy as np

img = cv2.imread('2.png',0)
gray_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)	# Convert to gray-scale

blured = cv2.medianBlur(img,3)					# Blur the image to reduce noise


circles = cv2.HoughCircles(blured,cv2.HOUGH_GRADIENT,1	, img.shape[0]/64,
                            param1=300,param2=40,minRadius=0,maxRadius=0)  # image, dp, minDistance, minRadious and maxRadious


if circles is not None:
	circles = np.uint16(np.around(circles))
	for i in circles[0, :]:
		# draw the outer circle
		cv2.circle(gray_img,(i[0],i[1]),i[2],(0,255,0),2)
		# draw the center of the circle
		
		cv2.circle(gray_img,(i[0],i[1]),1,(0,0,255),3)

cv2.imshow('detected circles',gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

