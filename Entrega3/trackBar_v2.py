# Com este programa queria tentar mexer na trackBar e mudar as cores da imagem

import numpy as np
import cv2 as cv
def nothing(x):
    pass
    

# Create a black image, a window
# img = np.zeros((300,512,3), np.uint8)
# cv.namedWindow('image')


img = cv.imread('Cores2.png') 
b,g,r = cv.split(img) # Separa em 3 arrays
cv.namedWindow('image')

# create trackbars for color change
cv.createTrackbar('R','image',0,255,nothing)
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)


# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'image',0,1,nothing)


while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == 27: #27 é a tecla 'ESC'
        break

    # get current positions of four trackbars
    rTracked = cv.getTrackbarPos('R','image')
    gTracked = cv.getTrackbarPos('G','image')
    bTracked = cv.getTrackbarPos('B','image')
    s = cv.getTrackbarPos(switch,'image')
    if s == 0:
        img
    else:
		r[:] = rTracked
        #img[:] = [bTracked,gTracked,rTracked]
        img = cv.merge([b,g,r])
cv.destroyAllWindows()
