import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard

ID = 1
MAX_SPEED = 60
Z_BOUND = 60
LAST_HEIGHT = -1

def marker_detection(frame, drone):
    return frame, drone

def dodge_marker(drone, id, tvecs):
    x, y, z = tvecs[0]
    global ID
    global Z_BOUND
    global LAST_HEIGHT
    print([ID, id, tvecs[0]])
    if id == 0:
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
            break_flag = False
            frame = frame_read.frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # for OpenCV >= 4.7
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
            parameters = cv2.aruco.DetectorParameters()

            # dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
            # parameters = cv2.aruco.DetectorParameters_create()
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
                    if markerIDs[i][0] == ID:
                        x, y, z = tvecs[0][0]
                        z_update = tvecs[i, 0, 2] - 40
                        y_update = -tvecs[i, 0, 1]
                        x_update = tvecs[i, 0, 0]
                        rotM = np.zeros((3,3))
                        cv2.Rodrigues(rvecs[i], rotM)
                        z_prime = np.matmul(rotM, np.array([0,0,1]))
                        yaw_update = math.atan2(z - z_prime[2], x - z_prime[0])
                        # print("org_z: ", str(z_update), "org_y: ", str(y_update), "org_x: ", str(x_update))
                        z_update = z_pid.update(z_update, sleep=0)
                        y_update = y_pid.update(y_update, sleep=0)
                        x_update = x_pid.update(x_update, sleep=0)
                        yaw_update = yaw_pid.update(yaw_update, sleep=0)
                        print("z_update: ", str(z_update), "y_update: ", str(y_update), "x_update: ", str(x_update), "yaw_update: ", str(yaw_update))
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
                    elif len(markerIDs)==1 and markerIDs[i][0] == 4:
                        break_flag = True
                        break
            else:
                z_update = 0
                y_update = 0
                x_update = 0
                yaw_update = 0
                
            cv2.imshow("drone", frame)
            
            key = cv2.waitKey(100)
            if break_flag:
                ID = 4
                break

            if key != -1:
                keyboard(drone, key)
            else:
                drone.send_rc_control(int(x_update) * 1, int(z_update) // 1, 0, int(yaw_update) * (-40))
            # print(key)
    elif id == ID and ID !=0 and (z > Z_BOUND or x > 10 or x < -10 or y < -10 or y > 10):
        if y > 10:
            Tello.move(drone, "down", 20)
        elif y < -10:
            Tello.move(drone, "up", 20)
        elif x < -10:
            Tello.move(drone, "left", 20)
        elif x > 10:
            Tello.move(drone, "right", 20)
        elif z > Z_BOUND:
            Tello.move(drone, "forward", max(20, (int)(z - Z_BOUND)))
        print("C")
    elif id == 1:
        Tello.move(drone, "right", 80)
        print("A")
        time.sleep(1)
        ID = 2
    elif id == 2:
        Tello.move(drone, "left", 70)
        Tello.move(drone, "forward", 100)
        print("B")
        time.sleep(1)
        ID = 3
    elif id == 3:
        Tello.move(drone, "down", 40)
        Tello.move(drone, "forward", 150)
        Tello.move(drone, "up", 80)
        time.sleep(1)
        ID = 0

    elif id == 4:
        print("id 4")
        Tello.rotate_clockwise(drone, 90)
        # Z_BOUND = 45
        ID = 5

    elif id == 5:
        # Tello.move(drone, "left", 150)
        drone.send_rc_control(-20, -6, 0, 0)
        ID = 6

    elif id == 6:
        Z_BOUND = 110
        drone.send_rc_control(0, -20, 0, 0)
        print("back")
        print(f"height %d, last-heigth %d", drone.get_distance_tof(), LAST_HEIGHT)
        if LAST_HEIGHT == -1:
            LAST_HEIGHT = drone.get_distance_tof()

        if LAST_HEIGHT - drone.get_distance_tof() > 40:
            print("land")
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(1)
            drone.land()
        elif y > 10:
            Tello.move(drone, "down", 20)
        elif y < -10:
            Tello.move(drone, "up", 20)
        elif x < -8:
            Tello.move(drone, "left", 20)
        elif x > 15:
            Tello.move(drone, "right", 20)
        elif z < Z_BOUND: 
            Tello.move(drone, "back", 20)
        return


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

        # for OpenCV >= 4.7
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters = cv2.aruco.DetectorParameters()

        # dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        # parameters = cv2.aruco.DetectorParameters_create()
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
                    global ID
                    # print([ID, id, tvecs[i][0]])
                    print("ID", ID)
                    if id == 0:
                        ID=0
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
