import numpy as np
import cv2 as cv
import os 

def nothing(x):
    print(x)

path = os.path.join('.','data','lena.jpg')
img = cv.imread(path)
cv.namedWindow('image')

cv.createTrackbar('CP', 'image', 0, 255, nothing)
# Add a switch
switch = 'color/gray'
cv.createTrackbar(switch, 'image', 0, 1, nothing)

while True:
    img = cv.imread(path)
    
    pos = cv.getTrackbarPos('CP','image')
    
    font = cv.FONT_HERSHEY_COMPLEX
    
    cv.putText(img,str(pos),(50,150), font, 4, (0,0,255))
    
    s = cv.getTrackbarPos(switch,'image')
    
    if s == 0:
        pass
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img = cv.imshow('image', img)

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break