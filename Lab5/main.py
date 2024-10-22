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

def main():
    # Tello
    drone = Tello()
    drone.connect()
    #time.sleep(10)
    drone.streamon()
    frame_read = drone.get_frame_read()
    
    z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    x_pid = PID(kP=0.6, kI=0.0002, kD=0.1)
    yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    
    yaw_pid.initialize()
    z_pid.initialize()
    y_pid.initialize()
    x_pid.initialize()
    
    z_update = 0
    y_update = 0
    x_update = 0
    yaw_update = 0
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
        if rvecs is not None and tvecs is not None:
            for i in range(rvecs.shape[0]):
                if markerIDs[i][0] == 2:
                    x, y, z = tvecs[0][0]
                    frame = cv2.aruco.drawAxis(frame, intrinsic, distorsion, rvecs, tvecs, 10)
                    frame = cv2.putText(frame, f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    z_update = tvecs[i, 0, 2] - 60
                    y_update = tvecs[i, 0, 1]
                    x_update = tvecs[i, 0, 0]
                    rotM = np.zeros(3,3)
                    cv2.Rodrigues(rvecs[i], rotM)
                    yaw_update = np.arctan2(rotM[1, 0], rotM[0, 0])
                    # print("org_z: ", str(z_update), "org_y: ", str(y_update), "org_x: ", str(x_update))
                    z_update = z_pid.update(z_update, sleep=0)
                    y_update = y_pid.update(y_update, sleep=0)
                    x_update = x_pid.update(x_update, sleep=0)
                    yaw_update = yaw_pid.update(yaw_update, sleep=0)
                    # print("z_update: ", str(z_update), "y_update: ", str(y_update), "x_update: ", str(x_update))
                    if z_update > MAX_SPEED:
                        z_update = MAX_SPEED
                    elif z_update < -MAX_SPEED:
                        z_update = -MAX_SPEED
                    if y_update > MAX_SPEED:
                        y_update = MAX_SPEED
                    elif y_update < -MAX_SPEED:
                        y_update = -MAX_SPEED
                    if x_update > MAX_SPEED:
                        x_update = MAX_SPEED
                    elif x_update < -MAX_SPEED:
                        x_update = -MAX_SPEED
                    if yaw_update > MAX_SPEED:
                        yaw_update = MAX_SPEED
                    elif yaw_update < -MAX_SPEED:
                        yaw_update = -MAX_SPEED
            
        cv2.imshow("drone", frame)
        
        key = cv2.waitKey(100)
        if key != -1:
            keyboard(drone, key)
        else:
            drone.send_rc_control(int(x_update) // 1, int(z_update) // 5, int(y_update) // 1, 0)
        # print(key)
    
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

