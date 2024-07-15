import os
import cv2

# read image
image_path = os.path.join('.','data','bird.jpg')
print(image_path)
img = cv2.imread(image_path)
# write image
cv2.imwrite(os.path.join('.', 'data', 'bird_out.jpg'), img)

# visualize image
cv2.imshow('Bird', img)
cv2.waitKey(0)