import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_detection(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if(len(rects)==0):return img

    for x, y, w, h in rects:
        rects = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return rects

if __name__ == "__main__":
    print("name=main")