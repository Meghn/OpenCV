import numpy as np
import cv2
import os

# events = [i for i in dir(cv2) if 'EVENT' in i]
# print(events)

# mouse callback function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'X: {x}, Y: {y}')
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'({x}, {y})'
        cv2.putText(img, text, (x, y), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('image', img)

    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y,x,0]
        green = img[y,x,1]
        red = img[y,x,2]
        print(f'Blue: {blue}, Green: {green}, Red: {red}')
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'(B:{blue}, G:{green}, R:{red})'
        cv2.putText(img, text, (x, y), font, .5, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('image', img)

# img = np.zeros((512,512,3), np.uint8)
image_path = os.path.join('.','data','lena.jpg')
img = cv2.imread(image_path)
cv2.imshow('image', img)

cv2.setMouseCallback('image',click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()