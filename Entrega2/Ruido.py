import numpy as np
import cv2
import glob 

mean = 0
var = 10
sigma = var ** 0.5
gaussian = np.random.normal(mean, sigma, (224, 224)) 



for image in glob.glob('Cores.jpg'):
    img = cv2.imread(image)
    noisy_image = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:
        noisy_image = img + gaussian
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian
        noisy_image[:, :, 1] = img[:, :, 1] + gaussian
        noisy_image[:, :, 2] = img[:, :, 2] + gaussian

        cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
noisy_image = noisy_image.astype(np.uint8)

cv2.imshow('img', img)
cv2.imshow('gaussian', gaussian)
cv2.imshow('noisy', noisy_image)
cv2.waitKey(0)
