import cv2
import os

# read video
video_path = os.path.join('.','data','elefant_1280p.mp4')
video = cv2.VideoCapture(video_path)
# visualize video
ret = True
while ret:
    ret, frame = video.read()
    if ret:
        cv2.imshow('Elephant', frame)
        cv2.waitKey(20)

video.release()
cv2.destroyAllWindows()