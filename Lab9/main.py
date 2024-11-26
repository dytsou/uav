import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard

MAX_SPEED = 60

def movement(frame, drone, angle):
    # [0] [1] [2]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)
    left = frame[:, :frame.shape[1]//3]
    center = frame[:, frame.shape[1]//3: 2 * frame.shape[1]//3]
    right = frame[:, 2*frame.shape[1]//3:]
    left = cv2.countNonZero(left)
    center = cv2.countNonZero(center)
    right = cv2.countNonZero(right)
    print(left, center, right)
    blocksize = (frame.shape[1]//3 * frame.shape[0])
    tmp = blocksize // 10
    block_left = left > tmp
    block_center = center > tmp 
    block_right = right > tmp
    x = 0
    y = 0
    if not block_left and block_center and not block_right:
        # go forward
        angle = angle
    elif not block_left and not block_center and block_right:
        # go right
        angle = (angle + 10) % 360
    elif block_left and not block_center and not block_right:
        # go left
        angle = (angle - 10) % 360
    elif not block_left and block_center and block_right:
        # slight right
        angle = (angle + 5) % 360
    elif block_left and block_center and not block_right:
        # slight left
        angle = (angle - 5) % 360
    else:
        Tello.send_rc_control(0, 0, 0, 0)
        return drone
    
    x = x + np.cos(angle * np.pi / 180)
    y = x + np.sin(angle * np.pi / 180)
    Tello.send_rc_control(int(x * 30), 0, int(y * 30), 0)
    return drone

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
    while True:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        parameters = cv2.aruco.DetectorParameters()
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
                if markerIDs[i][0] == 0:
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
            drone.send_rc_control(int(x_update) * 1, int(z_update) // 1, int(y_update) * 1, int(yaw_update) * (-10))
        # print(key)
    
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

