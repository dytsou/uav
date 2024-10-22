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
    print(id)
    if id < 3 and z > 40:
        Tello.move(drone, "forward", 20)
        print("C")
    elif id == 1:
        Tello.move(drone, "right", 100)
        print("A")
        cv2.waitKey(1500)
    elif id == 2:
        Tello.move(drone, "left", 100)
        print("B")
        cv2.waitKey(1500)


def main():
    # Tello
    drone = Tello()
    drone.connect()
    battery = drone.get_battery()
    print(f"Battery: {battery}%")
    time.sleep(5)
    drone.streamon()
    frame_read = drone.get_frame_read()
    
    while True:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        parameters = cv2.aruco.DetectorParameters_create()
        markerCorners, markerIDs, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        # print(markerIDs)
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIDs)
        fs = cv2.FileStorage("1.xml", cv2.FILE_STORAGE_READ)
        intrinsic = fs.getNode("instrinsic").mat()
        distorsion = fs.getNode("distortion").mat()
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distorsion)
        # print(rvecs, tvecs)
        
        if rvecs is not None and tvecs is not None:
            for i in range(rvecs.shape[0]):
                x, y, z = tvecs[0][0]
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

