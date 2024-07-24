import cv2
import os
import numpy as np

# Load the images

apple_path = os.path.join('.','data','apple.jpg')
apple = cv2.imread(apple_path)

orange_path = os.path.join('.','data','orange.jpg')
orange = cv2.imread(orange_path)

print(apple.shape)
print(orange.shape)

apple_orange = np.hstack((apple[:,:256],orange[:, 256:]))
merged = cv2.addWeighted(apple, 0.5, orange, 0.5, 0)

# Generate Gaussian Pyramid

apple_copy = apple.copy()
gp_apple = [apple_copy]

orange_copy = orange.copy()
gp_orange = [orange_copy]

num_levels = 6

for i in range(num_levels):
    apple_copy = cv2.pyrDown(apple_copy)
    gp_apple.append(apple_copy)

    orange_copy = cv2.pyrDown(orange_copy)
    gp_orange.append(orange_copy)

# Generate Laplacian Pyramid

apple_copy = gp_apple[num_levels-1]
lp_apple = [apple_copy]

orange_copy = gp_orange[num_levels-1]
lp_orange = [orange_copy]

for i in range(num_levels-1, 0, -1):
    gaussian_extended_apple = cv2.pyrUp(gp_apple[i])
    laplacian_apple = cv2.subtract(gp_apple[i - 1], gaussian_extended_apple)
    lp_apple.append(laplacian_apple)

    gaussian_extended_orange= cv2.pyrUp(gp_orange[i])
    laplacian_orange = cv2.subtract(gp_orange[i - 1], gaussian_extended_orange)
    lp_orange.append(laplacian_orange)


# Add left and right halves of images in each level

apple_orange_pyramid = []
n = 0
for apple_lap, orange_lap in zip(lp_apple, lp_orange):
    n += 1
    cols, rows, ch = apple_lap.shape
    laplacian_combined = np.hstack((apple_lap[:, 0:int(cols/2)], orange_lap[:, int(cols/2):]))
    apple_orange_pyramid.append(laplacian_combined)

# Reconstruct the image

apple_orange_reconstruct = apple_orange_pyramid[0].copy()
for i in range(1, num_levels):
    apple_orange_reconstruct = cv2.pyrUp(apple_orange_reconstruct)
    apple_orange_reconstruct = cv2.add(apple_orange_pyramid[i], apple_orange_reconstruct)


cv2.imshow("apple", apple)
cv2.imshow("orange",orange)
cv2.imshow("apple_orange", apple_orange)
cv2.imshow("merged", merged)
cv2.imshow("apple_orange_reconstruct", apple_orange_reconstruct)
cv2.waitKey(0)
cv2.destroyAllWindows()