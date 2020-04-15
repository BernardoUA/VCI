import cv2 
import numpy as np
from matplotlib import pyplot as plt
import argparse

def nothing(x):
    print(x)

def resize(image,scl):
    w_image = int(image.shape[1] * scl / 100)
    h_image = int(image.shape[0] * scl / 100)
    dim_image = (w_image,h_image)
    image_resized = cv2.resize(image, dim_image, interpolation = cv2.INTER_AREA)
    return image_resized

def tracking_realtime_img_grad():
    cap = cv2.VideoCapture('cambada_video.mp4')
    cv2.namedWindow('Original vs Filtered')

    switch = '(1)tracking_all  (2)tracking_ball (3)tracking_blue_team (4)tracking_orange_team (5)tracking_lines'
    cv2.createTrackbar(switch, 'Original vs Filtered', 1, 5, nothing)

    while(True):
        ret, frame = cap.read()
        frame=resize(frame,75)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        s = cv2.getTrackbarPos(switch, 'Original vs Filtered')

        #tracking_ball
        lower_hsv_ball = np.array([22, 114, 88])
        higher_hsv_ball = np.array([41, 254, 255])
        mask_ball = cv2.inRange(hsv, lower_hsv_ball, higher_hsv_ball)
        #tracking_blue_team()
        lower_hsv_blue = np.array([91, 78, 46])
        higher_hsv_blue = np.array([101, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_hsv_blue, higher_hsv_blue)
        #tracking_orange_team()
        lower_hsv_orange = np.array([0, 89, 0])
        higher_hsv_orange = np.array([20, 255, 196])
        mask_orange = cv2.inRange(hsv, lower_hsv_orange, higher_hsv_orange)
        #tracking_lines()
        lower_hsv_lines = np.array([0, 0, 162])
        higher_hsv_lines = np.array([179, 49, 255])
        mask_lines = cv2.inRange(hsv, lower_hsv_lines, higher_hsv_lines)
        
        if s==1:
            mask  = mask_ball+mask_blue+mask_lines+mask_orange
            frame_filtered = cv2.bitwise_or(frame, frame, mask=mask)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            frame_filtered = cv2.Laplacian(frame_filtered, cv2.CV_64F)
            frame_filtered = np.uint8(np.absolute(frame_filtered))  

        elif s==2:   
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_ball)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            frame_filtered = cv2.Laplacian(frame_filtered, cv2.CV_64F)
            frame_filtered = np.uint8(np.absolute(frame_filtered))       

        elif s==3:   
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_blue)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            frame_filtered = cv2.Laplacian(frame_filtered, cv2.CV_64F)
            frame_filtered = np.uint8(np.absolute(frame_filtered))   

        elif s==4:
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_orange)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            frame_filtered = cv2.Laplacian(frame_filtered, cv2.CV_64F)
            frame_filtered = np.uint8(np.absolute(frame_filtered))

        elif s==5:
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_lines)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            frame_filtered = cv2.Laplacian(frame_filtered, cv2.CV_64F)
            frame_filtered = np.uint8(np.absolute(frame_filtered))   

        # show thresholded image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        numpy_horizontal = np.hstack((frame, frame_filtered))
        cv2.imshow('Original vs Filtered', numpy_horizontal)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

def tracking_realtime_canny():
    cap = cv2.VideoCapture('cambada_video.mp4')
    cv2.namedWindow('Original vs Filtered')

    switch = '(1)tracking_all  (2)tracking_ball (3)tracking_blue_team (4)tracking_orange_team (5)tracking_lines'
    cv2.createTrackbar(switch, 'Original vs Filtered', 1, 5, nothing)

    while(True):
        ret, frame = cap.read()
        frame=resize(frame,75)

        cv2.imshow('Orig', frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        s = cv2.getTrackbarPos(switch, 'Original vs Filtered')

     #tracking_ball
        lower_hsv_ball = np.array([22, 114, 88])
        higher_hsv_ball = np.array([41, 254, 255])
        mask_ball = cv2.inRange(hsv, lower_hsv_ball, higher_hsv_ball)
        #tracking_blue_team()
        lower_hsv_blue = np.array([91, 78, 46])
        higher_hsv_blue = np.array([101, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_hsv_blue, higher_hsv_blue)
        #tracking_orange_team()
        lower_hsv_orange = np.array([0, 89, 0])
        higher_hsv_orange = np.array([20, 255, 196])
        mask_orange = cv2.inRange(hsv, lower_hsv_orange, higher_hsv_orange)
        #tracking_lines()
        lower_hsv_lines = np.array([0, 0, 162])
        higher_hsv_lines = np.array([179, 49, 255])
        mask_lines = cv2.inRange(hsv, lower_hsv_lines, higher_hsv_lines)
        
        if s==1:
            mask  = mask_ball+mask_blue+mask_lines+mask_orange
            frame_filtered = cv2.bitwise_or(frame, frame, mask=mask)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(frame_filtered,100,200,L2gradient=True)
            contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0,255,0), 3)      

        elif s==2:   
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_ball)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(frame_filtered,150,200,L2gradient=True)
            contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0,255,0), 3)     

        elif s==3:   
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_blue)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(frame_filtered,100,200,L2gradient=True)
            contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0,255,0), 3) 

        elif s==4:
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_orange)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(frame_filtered,100,200,L2gradient=True)
            contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0,255,0), 3) 

        elif s==5:
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_lines)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(frame_filtered,100,200,L2gradient=True)
            contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0,255,0), 3)    

        cv2.imshow('With Contours', frame)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

def image_gradients_first():

    cap = cv2.VideoCapture('cambada_video.mp4')

    while(True):
            ret, frame = cap.read()
            frame=resize(frame,55)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            laplacian = cv2.Laplacian(frame, cv2.CV_8U)
            sobelx = cv2.Sobel(frame,cv2.CV_8U,1,0,ksize=5)
            sobely = cv2.Sobel(frame,cv2.CV_8U,0,1,ksize=5)
            scharrx = cv2.Scharr(frame,cv2.CV_8U,1,0)
            scharry = cv2.Scharr(frame,cv2.CV_8U,0,1)

            # show thresholded image
            cv2.imshow('LaPlacian', laplacian)            
            cv2.imshow('SobelX', sobelx)
            cv2.imshow('SobelY', sobely)
            cv2.imshow('ScharrX', scharrx)
            cv2.imshow('ScharrY', scharry)

            if(cv2.waitKey(1) & 0xFF == ord('q')):
                break

def image_gradients_second():

    cap = cv2.VideoCapture('cambada_video.mp4')

    while(True):
            ret, frame = cap.read()
            frame=resize(frame,55)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            laplacian64 = cv2.Laplacian(frame, cv2.CV_64F)
            sobelx64 = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
            sobely64 = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)
            scharrx64 = cv2.Scharr(frame,cv2.CV_8U,1,0)
            scharry64 = cv2.Scharr(frame,cv2.CV_8U,0,1)

            laplacian_v2 = np.uint8(np.absolute(laplacian64))
            sobelx_v2 = np.uint8(np.absolute(sobelx64))
            sobely_v2 = np.uint8(np.absolute(sobely64))
            scharrx_v2 = np.uint8(np.absolute(scharrx64))
            scharry_v2 = np.uint8(np.absolute(scharry64))

            cv2.imshow('LaPlacian', laplacian_v2)
            cv2.imshow('SobelX', sobelx_v2)
            cv2.imshow('SobelY', sobely_v2)
            cv2.imshow('ScharrX', scharrx_v2)
            cv2.imshow('ScharrY', scharry_v2)

            if(cv2.waitKey(1) & 0xFF == ord('q')):
                break

def canny_edge():
    cap = cv2.VideoCapture('cambada_video.mp4')

    while(True):
            ret, frame = cap.read()
            frame=resize(frame,70)

            framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('original',frame)
            edges = cv2.Canny(framegray,100,180)
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0,255,0), 2)
            cv2.imshow('with contours', frame)

            if(cv2.waitKey(1) & 0xFF == ord('q')):
                break

def canny_edge_pic():
    
    img = cv2.imread('cambada_image.png')
    img=resize(img,40)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('original',img)
    edges1 = cv2.Canny(img_gray,100,170,L2gradient=True)
    contours, hierarchy = cv2.findContours(edges1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,255,0), 2)
    cv2.imshow('with contours',img)

    cv2.waitKey()

#tracking_realtime_img_grad()
tracking_realtime_canny()
#image_gradients_first()
#image_gradients_second()
#canny_edge()
#canny_edge_pic()

#cv2.destroyAllWindows()