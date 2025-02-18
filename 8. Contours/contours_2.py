import os
import cv2

img = cv2.imread(os.path.join('.','data','opencv-logo.png'))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img_gray, 100, 255, 0)


# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print("Number of Contours: ", str(len(contours)))
print(contours[0])

cv2.drawContours(img, contours, -1, (255,255,0), 3)


cv2.imshow('img', img)
cv2.imshow('img_gray', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
