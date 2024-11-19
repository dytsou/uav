import cv2
import numpy as np

def pedestrian_detection(img, height, focal_length):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    winStride = (4, 4)
    scale = 1.1
    rects, weights = hog.detectMultiScale(img, winStride=winStride, scale=scale, useMeanshiftGrouping=False)
    
    for (x, y, w, h) in rects:
        if w < 120 or h < 240:
            continue
        
        distance = (height * focal_length) / h
        print(f"human Estimated distance: {distance:.2f} cm")
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)
        cv2.putText(img, f"{distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return img
