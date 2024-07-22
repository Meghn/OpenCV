import cv2
import numpy as np
import os

def nothing(x):
    pass

cv2.namedWindow('Tracking')

cv2.createTrackbar('LH','Tracking', 0, 255, nothing)
cv2.createTrackbar('LS','Tracking', 0, 255, nothing)
cv2.createTrackbar('LV','Tracking', 0, 255, nothing)
cv2.createTrackbar('UH','Tracking', 255, 255, nothing)
cv2.createTrackbar('US','Tracking', 255, 255, nothing)
cv2.createTrackbar('UV','Tracking', 255, 255, nothing)

cap = cv2.VideoCapture(0)
# path = os.path.join('.','data','smarties.png') 

while True:
    # frame = cv2.imread(path)
    ret, frame = cap.read()

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        l_h = cv2.getTrackbarPos('LH', 'Tracking')
        l_s = cv2.getTrackbarPos('LS', 'Tracking')
        l_v = cv2.getTrackbarPos('LV', 'Tracking')
        
        u_h = cv2.getTrackbarPos('UH', 'Tracking')
        u_s = cv2.getTrackbarPos('US', 'Tracking')
        u_v = cv2.getTrackbarPos('UV', 'Tracking')

        l_b = np.array([l_h,l_s,l_v])
        u_b = np.array([u_h,u_s,u_v])

        mask = cv2.inRange(hsv, l_b, u_b)

        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)
    
    # Break the loop on 'ESC' key press
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        print("ESC key pressed, exiting...")
        break
cap.release()
cv2.destroyAllWindows()