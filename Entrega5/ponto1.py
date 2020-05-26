import numpy as np
import cv2 as cv
import math
import argparse

def resize(image,scl):
    w_image = int(image.shape[1] * scl / 100)
    h_image = int(image.shape[0] * scl / 100)
    dim_image = (w_image,h_image)
    image_resized = cv.resize(image, dim_image, interpolation = cv.INTER_AREA)
    return image_resized

def __draw_label(image, text, position, bg_col):
    font = cv.FONT_HERSHEY_DUPLEX 
    scale = 0.6
    color = (0, 0, 0)		
    thickness = cv.FILLED
    margin = 2
    size = cv.getTextSize(text, font, scale, thickness)
    x = position[0] + size[0][0] + margin
    y = position[1] - size[0][1] - margin
    cv.rectangle(image, position, (x, y), bg_col, thickness)
    cv.putText(image, text, position, font, scale, color, 1, cv.LINE_AA)

def area_limited(frame):
        # mask for area of interest
        rect = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
        rect = cv.rectangle(rect,(0,52), (426, frame.shape[1]),(255, 255, 255), -1)   #---the dimension of the ROI
        gray = cv.cvtColor(rect,cv.COLOR_BGR2GRAY)               
        ret,b_mask = cv.threshold(gray,127,255,0)
        area = cv.bitwise_and(frame,frame,mask = b_mask)
        return area

def tracking_with_ID():
    cap = cv.VideoCapture('cambada1.mov')
    cv.namedWindow('Multi-object tracking')
    ret, frame = cap.read()
    frame=resize(frame,25)
    height, width = frame.shape[:2]
	            
    while(True):
        ret, frame = cap.read()
        if ret==True:
            frame=resize(frame,25)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            #balls
            lower_hsv_ball = np.array([25, 115, 85])
            higher_hsv_ball = np.array([40, 255, 255])
            mask_ball = cv.inRange(hsv, lower_hsv_ball, higher_hsv_ball)
            #blue_robots
            lower_hsv_blue = np.array([90, 80, 45])
            higher_hsv_blue = np.array([100, 255, 255])
            mask_blue = cv.inRange(hsv, lower_hsv_blue, higher_hsv_blue)
			#orange_balls
            lower_hsv_orange = np.array([0, 90, 0])
            higher_hsv_orange = np.array([20, 255, 195])
            mask_orange = cv.inRange(hsv, lower_hsv_orange, higher_hsv_orange)
            #lines
            lower_hsv_lines = np.array([0, 0, 160])
            higher_hsv_lines = np.array([180, 50, 255])
            mask_lines = cv.inRange(hsv, lower_hsv_lines, higher_hsv_lines)

			
            limited = area_limited(frame)

            mask  = mask_ball+mask_orange
            frame_filtered = cv.bitwise_or(limited, limited, mask=mask)
            frame_filtered = cv.cvtColor(frame_filtered, cv.COLOR_BGR2GRAY)
            frame_filtered = cv.Canny(frame_filtered,150,200,L2gradient=True)
            frame_filtered = cv.GaussianBlur(frame_filtered,(7,7),cv.BORDER_DEFAULT)
            cv.imshow('Balls filtered with Canny & GB',frame_filtered)
            
            frame_filtered_robot = cv.bitwise_and(limited, limited, mask=mask_blue)
            frame_filtered_robot = cv.cvtColor(frame_filtered_robot, cv.COLOR_BGR2GRAY)
            frame_filtered_robot = cv.Canny(frame_filtered_robot,150,200,L2gradient=True)
            frame_filtered_robot = cv.GaussianBlur(frame_filtered_robot,(7,7),cv.BORDER_DEFAULT)
            cv.imshow('Robots filtered with Canny & GB',frame_filtered_robot)
        
            
            contours, hierarchy = cv.findContours(frame_filtered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours_robot, hierarchy_robot = cv.findContours(frame_filtered_robot, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            lenc=len(contours)

            for i in range(lenc):
                x,y,w,h = cv.boundingRect(contours[i])

                if lenc==1:
                    __draw_label(frame, 'Ball N1', (x,y-5), (0,255,0))
                if lenc==2:
                    if i==0:
                        __draw_label(frame, 'Ball N2', (x,y-5), (0,255,0))
                    if i==1:
                        __draw_label(frame, 'Ball N1', (x,y-5), (0,255,0))
                if lenc==3:
                    if i==0:
                        __draw_label(frame, 'Ball N3', (x,y-5), (0,255,0))
                    if i==1:
                        __draw_label(frame, 'Ball N2', (x,y-5), (0,255,0))
                    if i==2:
                        __draw_label(frame, 'Ball N1', (x,y-5), (0,255,0))
                if lenc==4:
                    if i==0:
                        __draw_label(frame, 'Ball N4', (x,y-5), (0,255,0))
                    if i==1:
                        __draw_label(frame, 'Ball N3', (x,y-5), (0,255,0))
                    if i==2:
                        __draw_label(frame, 'Ball N2', (x,y-5), (0,255,0))
                    if i==3:
                        __draw_label(frame, 'Ball N1', (x,y-5), (0,255,0))
                if lenc==5:
                    if i==0:
                        __draw_label(frame, 'Ball N5', (x,y-5), (0,255,0))
                    if i==1:
                        __draw_label(frame, 'Ball N4', (x,y-5), (0,255,0))
                    if i==2:
                        __draw_label(frame, 'Ball N3', (x,y-5), (0,255,0))
                    if i==3:
                        __draw_label(frame, 'Ball N2', (x,y-5), (0,255,0))
                    if i==4:
                        __draw_label(frame, 'Ball N1', (x,y-5), (0,255,0))
                if lenc==6:
                    if i==0:
                        __draw_label(frame, 'Ball N6', (x,y-5), (0,255,0))
                    if i==1:
                        __draw_label(frame, 'Ball N5', (x,y-5), (0,255,0))
                    if i==2:
                        __draw_label(frame, 'Ball N4', (x,y-5), (0,255,0))
                    if i==3:
                        __draw_label(frame, 'Ball N3', (x,y-5), (0,255,0))
                    if i==4:
                        __draw_label(frame, 'Ball N2', (x,y-5), (0,255,0))
                    if i==5:
                        __draw_label(frame, 'Ball N1', (x,y-5), (0,255,0))

            
                frame = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            lenc=len(contours_robot)
            for i in range(lenc):
                x,y,w,h = cv.boundingRect(contours_robot[i])

                if lenc==1:
                    __draw_label(frame, 'Robot N1', (x-10,y-5), (0,255,0))
                if lenc>=2:
                    if i==0:
                        __draw_label(frame, 'Robot N2', (x-10,y-5), (0,255,0))
                    if i==1:
                        __draw_label(frame, 'Robot N1', (x-10,y-5), (0,255,0))
                frame = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv.imshow('Multi-object tracking', frame)
            cv.waitKey(10)

            if(cv.waitKey(1) & 0xFF == ord('q')):
                break
        else:
            cap.release()
            cv.destroyAllWindows()
            break

tracking_with_ID()
