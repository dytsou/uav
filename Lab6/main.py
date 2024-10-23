import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard

ID = 1
MAX_SPEED = 60

def marker_detection(frame, drone):
    return frame, drone

def dodge_marker(drone, id, tvecs):
    x, y, z = tvecs[0]
    global ID
    print([ID, id, tvecs[0]])
    Z_BOUND = 60
    if id == ID and (z > Z_BOUND or x > 10 or x < -10 or y < -10 or y > 10):
        if y > 10:
            Tello.move(drone, "down", 20)
        elif y < -10:
            Tello.move(drone, "up", 20)
        elif x < -10:
            Tello.move(drone, "left", 20)
        elif x > 10:
            Tello.move(drone, "right", 20)
        elif z > Z_BOUND:
            Tello.move(drone, "forward", max(20, (int)(z - 30)))
        print("C")
    elif id == 1:
        Tello.move(drone, "right", 80)
        print("A")
        time.sleep(1)
        ID = 2
    elif id == 2:
        Tello.move(drone, "left", 60)
        print("B")
        time.sleep(1)


def main():
    # Tello
    drone = Tello()
    drone.connect()
    battery = drone.get_battery()
    print(f"Battery: {battery}%")
    time.sleep(1)
    drone.streamon()
    frame_read = drone.get_frame_read()
    
    lasttvecs = None
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
            if drone.is_flying:
                for i in range(rvecs.shape[0]):
                    id = markerIDs[i][0]
                    # print([ID, id, tvecs[i][0]])
                    global ID
                    if id == ID:
                        x,y,z = tvecs[i][0]
                        if lasttvecs is not None:
                            lx, ly, lz = lasttvecs
                        if id is not None:
                            if lasttvecs is None or abs(x - lx) < 50 or abs(y - ly) < 50 or abs(z - lz) < 50:
                                dodge_marker(drone, id, tvecs[i])
                                lasttvecs = tvecs[i][0]
                
        cv2.imshow("drone", frame)
        
        key = cv2.waitKey(33)
        if key != -1:
            keyboard(drone, key)
        # print(key)
    
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

