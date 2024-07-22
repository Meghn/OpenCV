import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

path = os.path.join('.','data','smarties.png')
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

_, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

kernal = np.ones((2,2), np.uint8)

dilation = cv2.dilate(mask, kernal, iterations=2)

erosion = cv2.erode(mask, kernal, iterations=2)

opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)

morph_gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernal)

top_hat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernal)

titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing', 'mg', 'th']
images = [img, mask, dilation, erosion, opening, closing, morph_gradient, top_hat]

for i in range(len(images)):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()