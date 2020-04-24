from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt

def resize(img,size):
    w_img = int(img.shape[1] * size / 75)
    h_img = int(img.shape[0] * size / 75)
    size_img = (w_img,h_img)
    img_resized = cv2.resize(img, size_img, interpolation = cv2.INTER_AREA)
    return img_resized

def main():
    capture = cv2.VideoCapture('cambada1.mov')

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
        image = resize(image,30)
        if ret == False:
            continue

        # filter
        img = cv2.GaussianBlur(image, (15, 15), 15)
        cimg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        #dimg = cv2.medianBlur(img, 5)
        #ret, dimg = cv2.threshold(dimg, 10, 255, cv2.THRESH_TOZERO)
        dimg = cv2.equalizeHist(cimg)
        # detect circles
        circles = cv2.HoughCircles(dimg,cv2.HOUGH_GRADIENT, dp=1, minDist=dimg.shape[0]/64,
            param1=300, param2=30, minRadius=10, maxRadius=30)
        if circles is None:
            #cv2.imshow("preview", frame)
            continue
        print circles
        #image_cercle = np.uint16(np.around(circles))
        for pt in circles[0,:]:
            cv2.circle(cimg,(pt[0],pt[1]),pt[2],(0,255,255),2)
            cv2.circle(cimg,(pt[0],pt[1]),1,(255,0,0),1)
		# show image
        cv2.imshow('color image', cimg)
        keyCode = cv2.waitKey(33)
        if keyCode >= 0:
            print("keyCode: {}".format(keyCode))
            if keyCode == 27:  # Esc
                cv2.imwrite("image.png", image)
                break
            #if keyCode == 32:  # space
            #    idx = (idx + 1) % 2

        # show image
        image = [image, dimg][idx]
        cv2.imshow("Capture", image)

    cv2.destroyAllWindows()

main()
