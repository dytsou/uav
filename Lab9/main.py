import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard

MAX_SPEED = 60
cell_width = 5
# forward_movement = np.array([5, 0, 0, 0])
left_movement = np.array([0, 0, -5, 0])
right_movement = np.array([0, 0, 5, 0])

def setMovement(angle):
    global forward_movement, left_movement, right_movement
    if angle <= 45 or angle > 315:
        # forward_movement = np.array([0, 0, -5, 0])
        left_movement = np.array([-5, 0, 0, 0])
        right_movement = np.array([5, 0, 0, 0])

    elif angle > 45 and angle <= 135:
        # forward_movement = np.array([5, 0, 0, 0])
        left_movement = np.array([0, 0, -5, 0])
        right_movement = np.array([0, 0, 5, 0])

    elif angle > 135 and angle <= 225:
        # forward_movement = np.array([0, 0, 5, 0])
        left_movement = np.array([5, 0, 0, 0])
        right_movement = np.array([-5, 0, 0, 0])

    else:
        # forward_movement = np.array([-5, 0, 0, 0])
        left_movement = np.array([0, 0, 5, 0])
        right_movement = np.array([0, 0, -5, 0])

def movement(frame, drone, angle):
    setMovement(angle)
    # [0] [1] [2]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)
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

    # print("{:^{5}}", "{up_frame:^{5}}","\n",
    #       "{left_frame:^{5}}", "{mid_frame:^{5}}", "{right_frame:^{5}}\n",
    #       "{:^{5}}", "{down_frame:^{5}}", "\n")
    print(up_frame,"\n", left_frame, mid_frame, right_frame, "\n", down_frame, "\n")
    
    blocksize = (frame.shape[1]//3 * frame.shape[0]//3)
    tmp = blocksize * 0.7
    up = up_frame > tmp
    left = left_frame > tmp 
    mid = mid_frame > tmp
    right = right_frame > tmp 
    down = down_frame > tmp
    print(up,"\n", left, mid, right, "\n", down, end="\n")
    x = 0
    y = 0

    if not mid:
        if left:
            #move right
            drone.send_rc_control(*right_movement)
        elif right:
            #move left
            drone.send_rc_control(*left_movement)
        else:
            print("Where is black line\n")
        return frame, drone, angle
    
    else:
        if up:
            #no need to do 
            angle = angle
        elif right:
            #turn right
            print("trun right\n")
            angle = (angle+90)%360
        elif left:
            #turn left
            angle = (angle-90)%360
        else:
            print("At the end of line\n")
        
    x = x + np.cos(angle * np.pi / 180)
    y = y + np.sin(angle * np.pi / 180)
    drone.send_rc_control(int(x * 5), 0, int(y * 5), 0)
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
    time.sleep(5)
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
    while True:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if break_flag == False:
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
            parameters = cv2.aruco.DetectorParameters()
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

                        print("xyz: ", x, y, z)
                        if(abs(x-10)<=10 and abs(y)<=10 and z<=60): 
                            Tello.move(drone, "right", 20)
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
                drone.send_rc_control(int(x_update) // 5, int(z_update) // 2, int(y_update) // 2, int(yaw_update) * (-1))
        else:
            key = cv2.waitKey(100)
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

