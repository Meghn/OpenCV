import cv2
import os

image_path = os.path.join('.','data','messi5.jpg')
image_path_2 = os.path.join('.','data','opencv-logo.png')
img = cv2.imread(image_path)
img_2 = cv2.imread(image_path_2)
cv2.imshow('original image', img)
cv2.imshow('logo image', img_2)
ball = img[280:340,330:390]

img[273:333,100:160] = ball
img = cv2.resize(img, (512,512))
img_2 = cv2.resize(img_2, (512,512))

# dst = cv2.add(img, img_2)

dst = cv2.addWeighted(img,0.3,img_2,0.7,0)

cv2.imshow('new image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()