import numpy as np
import cv2

def get_limits(color):
    c = np.uint8([[color]]) # here we insert the bgr value which we want to convert to hsv
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    lowerLimit = hsvC[0][0][0] - 10,100,100
    lowerLimit = np.array(lowerLimit, dtype=np.uint8)

    upperLimit = hsvC[0][0][0] + 10,255,255
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit