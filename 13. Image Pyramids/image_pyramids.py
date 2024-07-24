import cv2
import os
import numpy as np

path = os.path.join('.','data','lena.jpg')
img = cv2.imread(path)

lr1 = cv2.pyrDown(img)
lr2 = cv2.pyrDown(lr1)

hr2 = cv2.pyrUp(lr2)

cv2.imshow("Original Image", img)
cv2.imshow("Lower Resolution 1", lr1)
cv2.imshow("Lower Resolution 2", lr2)
cv2.imshow("High Resolution", hr2)
cv2.waitKey(0)
cv2.destroyAllWindows()