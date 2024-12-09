import cv2
import numpy as np


def face_detection(img, height, focal_length):
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if(len(rects)==0):
        return -1,-1,-1
    mx, my, mw, mh = -1, -1, -1, -1
    mdistance = -1
    for x, y, w, h in rects:
        distance = (height * focal_length) / h
        print(f"Face Estimated distance: {distance:.2f} cm")
<<<<<<< HEAD
        if w*h > mw*mh: 
=======
        if(w > mw) and (h > mh): 
>>>>>>> 01add34eae07636565133d79c22f6b626bc11545
            mx = x
            my = y
            mw = w
            mh = h
            mdistance = distance
        rects = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cx = mx + (mw // 2)
    cy = my + (mh // 2)
    print(type(cx))
    print(type(cy))
    return int(cx), int(cy), int(mdistance)