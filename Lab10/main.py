import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard

MAX_SPEED = 60
cell_width = 5

up_movement = [0, 0, 15, 0]
left_movement = [-7, 0, 0, 0]
right_movement = [7, 0, 0, 0]
down_movement = [0, 0, -15, 0]

def movement(frame, drone, angle):
    # [0] [1] [2]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2,2), np.uint8)
    frame = cv2.erode(frame, kernel, iterations=15)

    dx = frame.shape[0]//3
    dy = frame.shape[1]//3

    up_frame = frame[:dx, dy:2*dy]
    left_frame = frame[dx:2*dx, :dy]
    mid_frame = frame[dx:2*dx, dy:2*dy]
    right_frame = frame[dx:2*dx, 2*dy:]
    down_frame = frame[2*dx:, dy:2*dy]

    up_frame = cv2.countNonZero(up_frame)
    left_frame = cv2.countNonZero(left_frame)
    mid_frame = cv2.countNonZero(mid_frame)
    right_frame = cv2.countNonZero(right_frame)
    down_frame = cv2.countNonZero(down_frame)

    print("      ", up_frame,"\n", left_frame, mid_frame, right_frame, "\n", "     ", down_frame, "\n")
    
    blocksize = (frame.shape[1]//3 * frame.shape[0]//3)
    # tmp = blocksize * 0.7
    tmp = 65000
    up = up_frame < tmp
    left = left_frame < tmp 
    mid = mid_frame < tmp
    right = right_frame < tmp 
    down = down_frame < tmp
    print("      ", up,"\n", left, mid, right, "\n", "     ", down, end="\n")
    x = 0
    y = 0

    if (angle <= 45 or angle > 315) or (angle > 135 and angle <= 225):
        if not mid:
            if right:
                #move right
                drone.send_rc_control(right_movement[0], right_movement[1], right_movement[2], right_movement[3])
            elif left:
                #move left
                drone.send_rc_control(left_movement[0], left_movement[1], left_movement[2], left_movement[3])
            else:
                print("Where is black line\n")
        
        else:
            if  right:
                #turn right
                angle = 90
                Tello.move(drone, "right", 30)
            elif left:
                #turn left
                angle = 270
                Tello.move(drone, "left", 30)
            elif (angle <= 45 or angle > 315) and up:
                drone.send_rc_control(up_movement[0], up_movement[1], up_movement[2], up_movement[3])
            elif angle > 135 and angle <= 225 and down:
                drone.send_rc_control(down_movement[0], down_movement[1], down_movement[2], down_movement[3])
            else:
                print("At the end of line\n")

    else:
        if not mid:
            if up:
                #move up
                drone.send_rc_control(up_movement[0], up_movement[1], up_movement[2], up_movement[3])
            elif down:
                #move down
                drone.send_rc_control(down_movement[0], down_movement[1], down_movement[2], down_movement[3])
            else:
                print("Where is black line\n")
        
        else:
            if up:
                #turn up
                angle = 0
                Tello.move(drone, "up", 30)
            elif down:
                #turn down
                angle = 180
                Tello.move(drone, "down", 30)
            elif angle>45 and angle <= 135 and right:
                #move right
                drone.send_rc_control(right_movement[0], right_movement[1], right_movement[2], right_movement[3])
            elif angle > 225 and angle <= 315 and left: 
                #move left
                drone.send_rc_control(left_movement[0], left_movement[1], left_movement[2], left_movement[3])
            else:
                print("At the end of line\n")
        
    return frame, drone, angle

def rotate(frame, angle):
    image = frame  # 替換成你的圖片路徑
    (h, w) = image.shape[:2]  # 取得圖像的高度與寬度

    # 設定旋轉中心、角度和縮放比例
    center = (w // 2, h // 2)  # 以圖像中心為旋轉點
    scale = 1.0  # 不縮放

    # 計算旋轉矩陣
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # 進行仿射變換
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    return rotated_image


def main():
    # Tello
    drone = Tello()
    drone.connect()
    battery = drone.get_battery()
    print(f"Battery: {battery}%")
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

    break_flag = False
    angle = 90
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    while True:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if break_flag == False:
        # if False:
            # dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
            # parameters = cv2.aruco.DetectorParameters()
            markerCorners, markerIDs, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
            print(markerIDs)
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIDs)
            fs = cv2.FileStorage("drone.xml", cv2.FILE_STORAGE_READ)
            intrinsic = fs.getNode("instrinsic").mat()
            distorsion = fs.getNode("distortion").mat()
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distorsion)
            # print(rvecs, tvecs)
            if rvecs is not None and tvecs is not None:
                for i in range(rvecs.shape[0]):
                    if markerIDs[i][0] == 2:
                        x, y, z = tvecs[0][0]
                        z_update = tvecs[i, 0, 2] - 50
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

                        print("xyz: ", x, y, z)
                        if(abs(x)<=10 and abs(y)<=10 and abs(z-65)<=5): 
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            Tello.move(drone, "right", 30)
                            break_flag = True
                            print("Break!")
            else:
                z_update = 0
                y_update = 0
                x_update = 0
                yaw_update = 0 
            cv2.imshow("drone", frame)
            key = cv2.waitKey(100)
            if key != -1:
                keyboard(drone, key)
            else:
                drone.send_rc_control(int(x_update) // 1, int(z_update) // 2, int(y_update) // 1, 0)
        else:
            key = cv2.waitKey(100)
            markerCorners, markerIDs, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
            if(markerIDs is not None and angle==180 and markerIDs[0][0]==2):
                drone.land()
            if key != -1:
                keyboard(drone, key)
            else:
                print("angle(before):", angle)
                # frame = rotate(frame, angle) 
                frame, drone, angle = movement(frame, drone, angle)   
                print("angle:", angle)
            cv2.imshow("drone", frame)

                    
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

