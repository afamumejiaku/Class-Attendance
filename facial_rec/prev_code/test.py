
import cv2
import time

cap = cv2.VideoCapture('http://192.168.1.10:8080/?action=stream')
time.sleep(10)

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
