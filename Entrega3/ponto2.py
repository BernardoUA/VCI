import numpy as np
import cv2

def nothing(x):
    pass

cap = cv2.VideoCapture('cambada.mp4')

while(True):

    # Make a window for the video feed  
    cv2.namedWindow('frame',0)

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Make the trackbar used for HSV masking    
    cv2.createTrackbar('HSV','frame',0,255,nothing)

    # Name the variable used for mask bounds
    j = cv2.getTrackbarPos('HSV','frame')

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of color in HSV
    lower = np.array([j-10,-100,-100])
    upper = np.array([j+10,255,255])

    # Threshold the HSV image to get only selected color
    mask = cv2.inRange(hsv, lower, upper)

    # Bitwise-AND mask the original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    # Display the resulting frame
    cv2.imshow('frame',res)

    # Press q to quit
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
