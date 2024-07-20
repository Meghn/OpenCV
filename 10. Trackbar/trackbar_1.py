import numpy as np
import cv2 as cv

def nothing(x):
    print(x)


# Create a black image, a window
img = np.zeros((300,512,3), np.uint8)
cv.namedWindow('image')

cv.createTrackbar('B', 'image', 0, 255, nothing)
cv.createTrackbar('G', 'image', 0, 255, nothing)
cv.createTrackbar('R', 'image', 0, 255, nothing)

# Add a switch
switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'image', 0, 1, nothing)

while True:
    cv.imshow('image', img)

    # Get current positions of trackbars
    b = cv.getTrackbarPos('B', 'image')
    g = cv.getTrackbarPos('G', 'image')
    r = cv.getTrackbarPos('R', 'image')

    s = cv.getTrackbarPos(switch, 'image')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, g, r]

    # Get key input
    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break

    # # If 's' is pressed, save the image
    # elif k == ord('s'):
    #     cv.imwrite('
