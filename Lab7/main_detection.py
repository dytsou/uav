import cv2
import numpy as np
from face_detection import face_detection

cap = cv2.VideoCapture(1)

while(True):
    c_ret, frame = cap.read()
    if not c_ret:
        break

    img = face_detection(frame)

    cv2.imshow('frame', img)

    if cv2.waitKey(33) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()