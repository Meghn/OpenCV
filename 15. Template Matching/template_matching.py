import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

path = os.path.join('.','data','messi5.jpg')
img = cv2.imread(path)

# face = img[87:130, 222:260]
# cv2.imwrite(os.path.join('.', 'data', 'messi_face.jpg'), face)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


face_path = os.path.join('.','data','messi_face.jpg')
template = cv2.imread(face_path, 0)

res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
print(np.max(res))

threshold = 0.9
loc = np.where(res >= threshold)
print(loc)

w, h = template.shape[::-1] # column and row value in reverse order
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)


cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
