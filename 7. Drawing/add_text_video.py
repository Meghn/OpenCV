import cv2
import os

import datetime
# read webcam
webcam = cv2.VideoCapture(0)
w = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(w,h)

# modify
# webcam.set(3, 1000) # width
# webcam.set(4, 720) # height
# w_1 = webcam.get(3)
# h_1 = webcam.get(4)
# print(w_1,h_1)

# write
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = os.path.join('.','data','video_text.mp4')
out = cv2.VideoWriter(output_path, fourcc, 20.0,(w,h))
# visulize webcam
while (webcam.isOpened()):
    ret, frame = webcam.read()
    if ret:
        # grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Width: {str(webcam.get(3))}, , Height: {str(webcam.get(4))}"
        datet = str(datetime.datetime.now())
        cv2.putText(frame, text, (10,100), font, 1, (0, 255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, datet, (10,50), font, 1, (0, 255,255), 2, cv2.LINE_AA)
        cv2.imshow('Webcam', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()
