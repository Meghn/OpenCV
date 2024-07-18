import cv2
import os
# read webcam
webcam = cv2.VideoCapture(0)
w = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(w,h)
# write
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = os.path.join('.','data','output.avi')
out = cv2.VideoWriter(output_path, fourcc, 20.0,(w,h))
# visulize webcam
while (webcam.isOpened()):
    ret, frame = webcam.read()
    if ret:
        # grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Webcam', frame)
        out.write(gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()
