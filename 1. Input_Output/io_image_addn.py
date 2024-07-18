import os
import cv2

# read image
image_path = os.path.join('.','data','lena.jpg')
print(image_path)
img = cv2.imread(image_path, 0)

# visualize image
cv2.imshow('Lena', img)
k = cv2.waitKey(0) & 0xFF

if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for's' key to save and exit
    # write image
    cv2.imwrite(os.path.join('.', 'data', 'lena_out.jpg'), img)
    cv2.destroyAllWindows()