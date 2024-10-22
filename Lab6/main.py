import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard

MAX_SPEED = 60

def marker_detection(frame, drone):
    return frame, drone

def dodge_marker(drone, id, z):
        if z>20:
            drone.send_rc_control(0, 1, 0, 0)
        elif id == 1:
            Tello.move(drone, "right", 20)
        elif id == 2:
            Tello.move(drone, "left", 20)

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
        
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

        parameters = cv2.aruco.DetectorParameters_create()

        markerCorners, markerIDs, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        print(markerIDs)

        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIDs)

        fs = cv2.FileStorage("1.xml", cv2.FILE_STORAGE_READ)
        intrinsic = fs.getNode("instrinsic").mat()
        distorsion = fs.getNode("distortion").mat()
        
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distorsion)
        # print(rvecs, tvecs)
        if drone.is_flying:
            x, y, z = tvecs[0][0]
            frame = cv2.aruco.drawAxis(frame, intrinsic, distorsion, rvecs, tvecs, 10)
            frame = cv2.putText(frame, f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            for i in range(rvecs.shape[0]):
                id = markerIDs[i][0]
                if id is not None:
                    dodge_marker(drone, id, z)
                
        cv2.imshow("drone", frame)
        
        key = cv2.waitKey(100)
        if key != -1:
            keyboard(drone, key)
        # print(key)
    
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

