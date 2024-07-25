import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

# img = np.zeros((200,200), np.uint8)
# cv.rectangle(img, (0,100),(200,200),(255), -1)
# cv.rectangle(img, (50,50), (150,150), (127), -1)

path = os.path.join('.','data','lena.jpg')
img = cv.imread(path)

# b, g, r = cv.split(img)

histb = cv.calcHist([img],[0], mask=None, histSize=[256], ranges=[0,256])
histg = cv.calcHist([img],[1], mask=None, histSize=[256], ranges=[0,256])
histr = cv.calcHist([img],[2], mask=None, histSize=[256], ranges=[0,256])
plt.plot(histb)
plt.plot(histg)
plt.plot(histr)
plt.show()


cv.imshow("img", img)
# cv.imshow("b", b)
# cv.imshow("g", g)
# cv.imshow("r", r)


# plt.hist(b.ravel(), 256, [0, 256])
# plt.hist(g.ravel(), 256, [0, 256])
# plt.hist(r.ravel(), 256, [0, 256])
# plt.show()

cv.waitKey(0)
cv.destroyAllWindows()