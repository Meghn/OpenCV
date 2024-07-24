import cv2
import os
import numpy as np

path = os.path.join('.','data','lena.jpg')
img = cv2.imread(path)

layer = img.copy()
gaussian_pyramid = [layer]

for i in range(6):
    layer = cv2.pyrDown(layer)
    gaussian_pyramid.append(layer)

layer = gaussian_pyramid[-1]
cv2.imshow("upper level Gaussian pyramid", layer)

laplacian_pyramid = [layer]

for i in range(5, 0, -1):
    gaussian_extended = cv2.pyrUp(gaussian_pyramid[i])
    laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_extended)
    cv2.imshow(str(i), laplacian)

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()