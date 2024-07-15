import cv2

# read webcam
webcam = cv2.VideoCapture(1)
# visulize webcam
while True:
    ret, frame = webcam.read()
    if ret:
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()