import os

import cv2


img = cv2.imread(os.path.join('.','data','bird.jpg'))

print(img.shape)

cropped_img = img[120:240, 120:260]

cv2.imshow('img', img)
cv2.imshow('cropped_img', cropped_img)
cv2.waitKey(0)