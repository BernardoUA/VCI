import numpy as np
import cv2 
from matplotlib import pyplot as plt
img = cv2.imread('Cores2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hist,bins = np.histogram(gray.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(gray.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')


# Equalização
equ = cv2.equalizeHist(gray)
res = np.hstack((gray,equ))
#cv.imwrite('res.png',res)
cv2.imshow('res', res)

plt.show()
