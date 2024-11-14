import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

def face_detection(img, height, focal_length):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if(len(rects)==0):return img

    for x, y, w, h in rects:
        distance = (height * focal_length) / h
        print(f"Face Estimated distance: {distance:.2f} cm")
        
        rects = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return rects

if __name__ == "__main__":
    print("name=main")