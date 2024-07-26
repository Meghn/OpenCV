import cv2
import os
import numpy as np

path = os.path.join('.','data','sudoku.png')
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Step 1:
edges = cv2.Canny(gray, 50, 150, apertureSize=3) # img, first thresh, second thresh, aperture size.
cv2.imshow("edges", edges)
# Step 2:
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
# source image, rho, theta, threshold
print(lines.shape)
print(lines[0])

for line in lines:
    rho, theta = line[0]

    a = np.cos(theta)
    b= np.sin(theta)

    x0 = a*rho
    y0 = b*rho

    # x1 stores the rounded off values of (rho * cos(theta) - 1000*sin(theta))
    x1 = int(x0 + 1000*(-b))
    # y1 stores the rounded off values of (rho * sin(theta) + 1000*cos(theta))
    y1 = int(y0 + 1000*(a))
    # x2 stores the rounded off values of (rho * cos(theta) + 1000*sin(theta))
    x2 = int(x0 - 1000*(-b))
    # y2 stores the rounded off values of (rho * sin(theta) - 1000*cos(theta))
    y2 = int(y0 - 1000*(a))
    # Draw the line on the image
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('Detected Lines', img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()