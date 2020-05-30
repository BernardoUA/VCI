import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
b,g,r = cv2.split(img) # Separa em 3 arrays
img2 = cv2.merge([r,g,b]) # Junta os 3 arrays, mas troca-se a primeira e terceira coluna
plt.subplot(121);plt.imshow(img) # expects distorted color
plt.subplot(122);plt.imshow(img2) # expect true color
plt.show()

cv2.imshow('bgr image',img) # expects true color
cv2.imshow('rgb image',img2) # expects distorted color
cv2.waitKey(0)
cv2.destroyAllWindows()
