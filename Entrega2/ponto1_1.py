import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg',1)
Converted_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) #BGR2HSV
													   #BGR2GRAY
											           #BGR2YUV
											           
plt.subplot(221);plt.imshow(Converted_YUV) # expects distorted color

Converted_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #BGR2HSV
													   #BGR2GRAY
											           #BGR2YUV
plt.subplot(222);plt.imshow(Converted_Gray) # expects distorted color										           



Converted_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #BGR2HSV
													   #BGR2GRAY
											           #BGR2YUV
plt.subplot(223);plt.imshow(Converted_HSV) # expects distorted color
plt.show()											       
											           
cv2.waitKey(0)
cv2.destroyAllWindows()

