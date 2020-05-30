import numpy as np
import cv2 as cv

def resize(img,size):
    w_img = int(img.shape[1] * size / 75)
    h_img = int(img.shape[0] * size / 75)
    size_img = (w_img,h_img)
    img_resized = cv.resize(img, size_img, interpolation = cv.INTER_AREA)
    return img_resized
    
cap = cv.VideoCapture(0)

# Modify the video settings
#cap.set(3,1280)
#cap.set(4,1020)
#cap.set(10, 1)
#cap.set(11, 1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # resize
    frame = resize(frame,70)
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
