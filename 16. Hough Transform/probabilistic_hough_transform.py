import cv2
import os
import numpy as np

path = os.path.join('.','data','road.jpeg')
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Step 1:
edges = cv2.Canny(gray, 200, 256, apertureSize=3) # img, first thresh, second thresh, aperture size.
cv2.imshow("edges", edges)
# Step 2:
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
# source image, rho, theta, threshold, min line length, max line gap
print(lines.shape)
print(lines[0])

for line in lines:
    x1,y1,x2,y2 = line[0]
    # Draw the line on the image
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Detected Lines', img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()