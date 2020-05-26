import cv2
from matplotlib import pyplot as plt
import numpy as np


def resize(image,scl):
    w_image = int(image.shape[1] * scl / 100)
    h_image = int(image.shape[0] * scl / 100)
    dim_image = (w_image,h_image)
    image_resized = cv2.resize(image, dim_image, interpolation = cv2.INTER_AREA)
    return image_resized


cap = cv2.VideoCapture('cambada1.mov')

while(True):
	ret, frame = cap.read()
	frame = resize(frame,30)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	for i in range(3):
		# Read the template 
		template = cv2.imread('image'+str(i)+'.png',0)
		#cv2.imshow('Test',template)
		#Store width and height of template in w and h 
		w, h = template.shape[::-1]
		# Perform match operations. 
		res = cv2.matchTemplate(gray,template,cv2.TM_SQDIFF_NORMED )

		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

		top_left = min_loc
		bottom_right = (top_left[0] + w, top_left[1] + h)

		cv2.rectangle(frame,top_left, bottom_right, 255, 1)
		cv2.putText(frame, 'Detected Object ID: '+str(i), (top_left[0],top_left[1]-10), 
				cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255))


	cv2.imshow('Multi-tracking objects',frame)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()		
