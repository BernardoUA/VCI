import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Cores2.png')
img2 = cv2.imread('Ruido.png')

blur = cv2.blur(img,(5,5))
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

blur1 = cv2.GaussianBlur(img,(5,5),0)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur1),plt.title('Gaussian Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

blur2 = cv2.medianBlur(img2,11)
plt.subplot(121),plt.imshow(img2),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur2),plt.title('Median')
plt.xticks([]), plt.yticks([])
plt.show()

blur3 = cv2.bilateralFilter(img,9,75,75)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur3),plt.title('Bilateral Filter')
plt.xticks([]), plt.yticks([])
plt.show()


