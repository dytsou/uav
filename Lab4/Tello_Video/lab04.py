import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID

def marker_detection(frame):
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIDs, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

    frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIDs)

    fs = cv2.FileStorage("1.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("instrinsic").mat()
    distorsion = fs.getNode("distortion").mat()
    
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 2, intrinsic, distorsion)
    print(rvecs, tvecs)
    
    if rvecs is not None and tvecs is not None:
        x, y, z = tvecs[0][0]
        frame = cv2.aruco.drawAxis(frame, intrinsic, distorsion, rvecs, tvecs, 10)
        frame = cv2.putText(frame, f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
     
    return frame

def main():
    # Tello
    drone = Tello()
    drone.connect()
    #time.sleep(10)
    drone.streamon()
    frame_read = drone.get_frame_read()
    

    while True:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = marker_detection(frame)
        
        cv2.imshow("drone", frame)
        
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break
    
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

