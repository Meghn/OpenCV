import cv2
import os
import numpy as np

path = os.path.join('.','data','lena.jpg')
img = cv2.imread(path)

layer = img.copy()
gausiian_pyramid = [layer]

for i in range(6):
    layer = cv2.pyrDown(layer)
    gausiian_pyramid.append(layer)
    cv2.imshow(str(i), layer)

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()