from matplotlib import pyplot as plt
import cv2
import os

path = os.path.join('.','data','lena.jpg')
img = cv2.imread(path, -1)
cv2.imshow('image',img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.xticks([]), plt.yticks([])
plt.show()

_, th1  = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
_, th2 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

titles = ["Original Image", "BINARY", "BINARY_INV"]
images = [img, th1, th2]

for i in range(3):
    plt.subplot(1,3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()