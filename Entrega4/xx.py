from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    capture = cv2.VideoCapture(0)
    
    if capture.isOpened() is False:
        raise("IO Error")
        
    # setting
    capture.set(cv2.CAP_PROP_FPS, 12)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)

    # background image
    ret, bg = capture.read()
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)

    idx = 0

    while True:
        ret, image = capture.read()

        if ret == False:
            continue
        
        # filter
        img = cv2.GaussianBlur(image, (5, 5), 0)
        cimg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        dimg = cv2.medianBlur(img, 5)
        ret, dimg = cv2.threshold(dimg, 10, 255, cv2.THRESH_TOZERO)
        dimg = cv2.equalizeHist(cimg)
        
        # detect circles
        circles = cv2.HoughCircles(dimg, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=32, param2=40, minRadius=20, maxRadius=50)
        
        image_cercle = np.uint16(np.around(circles)) 
        for pt in image_cercle[0,0:2]:

		#centre_x,centre_y,rayon=pt[0],pt[1],pt[2]
		cv2.circle(cimg,(pt[0],pt[1]),pt[2],(0,255,255),4)
		cv2.circle(cimg,(pt[0],pt[1]),1,(255,0,0),3)
		# show image
        cv2.imshow('color image', cimg)
        keyCode = cv2.waitKey(33)
        if keyCode >= 0:
            print("keyCode: {}".format(keyCode))
            if keyCode == 27:  # Esc
                cv2.imwrite("image.png", image)
                break
            if keyCode == 32:  # space
                idx = (idx + 1) % 2
        
        # show image
        image = [image, dimg][idx]
        cv2.imshow("Capture", image)

    cv2.destroyAllWindows() 

main()
