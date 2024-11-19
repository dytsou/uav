import cv2
import numpy as np
from face_detection import face_detection
from pedestrian_detection import pedestrian_detection

cap = cv2.VideoCapture(0)

while(True):
    c_ret, frame = cap.read()
    if not c_ret:
        break
    fs = cv2.FileStorage("1.xml", cv2.FILE_STORAGE_READ)

    intrinsic = fs.getNode("instrinsic").mat()
    distorsion = fs.getNode("distortion").mat()

    f_x = intrinsic[0, 0] #x焦距
    f_y = intrinsic[1, 1] #y焦距


    img = face_detection(frame, 20, f_y)
    img = pedestrian_detection(frame, 160, f_y)

    cv2.imshow('frame', img)

    if cv2.waitKey(33) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()