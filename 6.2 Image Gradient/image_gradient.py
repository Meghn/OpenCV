import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

path = os.path.join('.','data','messi5.jpg')
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

lap = cv2.Laplacian(img, cv2.CV_64F, ksize=1)
lap = np.uint8(np.absolute(lap))

sobelx = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0)
sobelx = np.uint8(np.absolute(sobelx))

sobely = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1)
sobely = np.uint8(np.absolute(sobely))

sobelCombined = cv2.bitwise_or(sobelx, sobely)

sobel = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=1)
sobel = np.uint8(np.absolute(sobel))

titles = ['image', 'Laplacian', 'SobelX', 'SobelY', 'Combined', 'Sobel']
images = [img, lap, sobelx, sobely, sobelCombined, sobel]

for i in range(len(images)):
    plt.subplot(2,3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()