import cv2
import os

image_path = os.path.join('.','data','lena.jpg')
img = cv2.imread(image_path)

print(f"Image shape: {img.shape}")
print(f"Image size: {img.size}")
print(f"Image data type: {img.dtype}")

b, g, r = cv2.split(img)
new_img = cv2.merge((r, g, b))

cv2.imshow('Original Image', img)
cv2.imshow('New Image', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
