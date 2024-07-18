import cv2

# read webcam
webcam = cv2.VideoCapture(0)
# visulize webcam
while True:
    ret, frame = webcam.read()
    if ret:
        # grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Webcam', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()