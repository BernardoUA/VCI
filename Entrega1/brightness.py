import numpy as np
import cv2 as cv 

cap = cv.VideoCapture(0)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Change the brightness modify the number
    bri = cv.add(frame,np.array([170.0]))
   
    # Display the resulting bri
    cv.imshow('frame', bri)
    if cv.waitKey(25) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
