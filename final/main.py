import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard
from face_detection import face_detection
from detectDoll import detectDoll

MAX_SPEED = 60
cell_width = 5

LEVEL = 0

up_movement = [0, 0, 20, 0]
left_movement = [-10, 0, 0, 0]
right_movement = [10, 0, 0, 0]
down_movement = [0, 0, -20, 0]


def movement9(frame, drone, angle, cnt):
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
    tmp = 63000
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
            if left:
                #move left
                drone.send_rc_control(left_movement[0], left_movement[1], left_movement[2], left_movement[3])
            elif right:
                #move right
                # print('right')
                drone.send_rc_control(right_movement[0], right_movement[1], right_movement[2], right_movement[3])
            else:
                print("Where is black line\n")        
        else:
            if (angle <= 45 or angle > 315) and up:
                drone.send_rc_control(up_movement[0], up_movement[1], up_movement[2], up_movement[3])
            elif angle > 135 and angle <= 225 and down:
                drone.send_rc_control(down_movement[0], down_movement[1], down_movement[2], down_movement[3])
            elif right:
                #turn right
                # angle = 90
                cnt += 1
            elif left:
                #turn left
                angle = 270
                cnt += 1
            else:
                print("At the end of line\n")

    else:
        if not mid:
            if left and right:
                drone.send_rc_control(left_movement[0], left_movement[1], left_movement[2], left_movement[3])
            elif up:
                #move up
                drone.send_rc_control(up_movement[0], up_movement[1], up_movement[2], up_movement[3])
            elif down:
                #move down
                drone.send_rc_control(down_movement[0], down_movement[1], down_movement[2], down_movement[3])
            else:
                print("Where is black line\n")
        
        else:
            if angle > 225 and angle <= 315 and left: 
                #move left
                drone.send_rc_control(left_movement[0], left_movement[1], left_movement[2], left_movement[3])
            elif angle>45 and angle <= 135 and right:
                #move right
                # print('right')
                # drone.send_rc_control(right_movement[0], right_movement[1], right_movement[2], right_movement[3])
                drone.send_rc_control(left_movement[0], left_movement[1], left_movement[2], left_movement[3])
            elif up:
                #turn up
                angle = 0
                cnt += 1
            elif down:
                #turn down
                angle = 180
                cnt += 1
            else:
                print("At the end of line\n")
                        
    return frame, drone, angle, cnt

def movement10(frame, drone, angle, cnt):
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
    tmp = 63000
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
            if left:
                #move left
                drone.send_rc_control(left_movement[0], left_movement[1], left_movement[2], left_movement[3])
            elif right:
                #move right
                # print('right')
                drone.send_rc_control(right_movement[0], right_movement[1], right_movement[2], right_movement[3])
            else:
                print("Where is black line\n")        
        else:
            if left:
                #turn left
                angle = 270
                cnt += 1
            elif right:
                #turn right
                # angle = 90
                cnt += 1
            elif (angle <= 45 or angle > 315) and up:
                drone.send_rc_control(up_movement[0], up_movement[1], up_movement[2], up_movement[3])
            elif angle > 135 and angle <= 225 and down:
                drone.send_rc_control(down_movement[0], down_movement[1], down_movement[2], down_movement[3])
            else:
                print("At the end of line\n")

    else:
        if not mid:
            if left and right:
                drone.send_rc_control(left_movement[0], left_movement[1], left_movement[2], left_movement[3])
            elif up:
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
                cnt += 1
            elif down:
                #turn down
                angle = 180
                cnt += 1
            elif angle > 225 and angle <= 315 and left: 
                #move left
                drone.send_rc_control(left_movement[0], left_movement[1], left_movement[2], left_movement[3])
            elif angle>45 and angle <= 135 and right:
                #move right
                # print('right')
                # drone.send_rc_control(right_movement[0], right_movement[1], right_movement[2], right_movement[3])
                drone.send_rc_control(left_movement[0], left_movement[1], left_movement[2], left_movement[3])
            else:
                print("At the end of line\n")
                        
    return frame, drone, angle, cnt


def level1(drone, frame_read):
    # 起飛，看人臉往上往前往下，看人臉往下往前往上
    state = 0
    while True:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fs = cv2.FileStorage("drone.xml", cv2.FILE_STORAGE_READ)
        intrinsic = fs.getNode("instrinsic").mat()
        distorsion = fs.getNode("distortion").mat()
        
        f_x = intrinsic[0, 0] #x焦距
        f_y = intrinsic[1, 1] #y焦距
        
        # todo:數字須調整
        cx, cy, dis = face_detection(frame, 30, f_y)
        print(cx, cy, dis, state)
        mx = frame.shape[0]/2
        my = frame.shape[1]/2
        print(mx, my)
        
        key = cv2.waitKey(100)
        if key != -1:
            keyboard(drone, key)
        elif (state == 0 or state == 2):
            if dis<0:
                tmp = 0
            elif (cx > mx+100): 
                drone.send_rc_control(10, 0, 0, 0)
            elif (cx < mx-100): 
                drone.send_rc_control(-10, 0, 0, 0)
            # elif (cy > my+100): 
            #     drone.send_rc_control(0, 0, 20, 0)
            # elif (cy < my-100): 
                # drone.send_rc_control(0, 0, -20, 0)
            elif(dis > 120): 
                drone.send_rc_control(0, 15, 0, 0)
            else: state += 1
        elif (state == 1):
            Tello.move(drone, "up", 80)
            time.sleep(1)
            Tello.move(drone, "forward", 120)
            time.sleep(1)
            Tello.move(drone, "down", 160)
            time.sleep(1)
            state += 1
        elif (state == 3):
            Tello.move(drone, "down", 60)
            # time.sleep(1)
            Tello.move(drone, "forward", 150)
            time.sleep(1)
            Tello.move(drone, "up", 100)
            time.sleep(1)
            break
        
        cv2.imshow("drone", frame)
    print("fin")

def level2(drone, frame_read):
    # 識別娃娃，回傳決定追線二不同階段
    # todo:放入lab8的東東
    # return 1
    while(1):
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        doll = detectDoll(frame) #1 Carno, 2 Melody, 0 no detect
        if(doll != 0):
            return doll
        key = cv2.waitKey(100)
        if key!=-1:
            keyboard(drone, key)
    return 0

def trace_with_prior(drone, frame_read, prior):
    start_time = time.time()
    angle = 0
    cnt = 0
    running_prior = prior
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    while(1):
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        markerCorners, markerIDs, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        key = cv2.waitKey(100)
        if key != -1:
            keyboard(drone, key)
        else:
            if running_prior == 2: frame, drone, angle, cnt = movement10(frame, drone, angle, cnt) 
            else: frame, drone, angle, cnt = movement9(frame,drone, angle, cnt) 
            time.sleep(0.3)

        cv2.imshow("drone", frame)
        print(angle)
        if(running_prior==prior and time.time()-start_time>60): 
            if prior==2:
                running_prior = 1 
            else:
                running_prior = 2
            print("prior change!!!!!!!!!")

        fs = cv2.FileStorage("drone.xml", cv2.FILE_STORAGE_READ)
        intrinsic = fs.getNode("instrinsic").mat()
        distorsion = fs.getNode("distortion").mat()
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distorsion)
        if(rvecs is not None and tvecs is not None and markerIDs[0][0] == 2): break

def trace_aruco(drone, frame_read):
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
    while break_flag == False:
        frame = frame_read.frame
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
                    z_update = tvecs[i, 0, 2] - 20
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
                    if(abs(x)<=10 and abs(y)<=10 and abs(z-40)<=5): 
                        drone.send_rc_control(0, 0, 0, 0)
                        time.sleep(1)
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
            drone.send_rc_control(int(x_update) // 1, int(z_update) // 2, int(y_update) // 1, int(yaw_update) * (-1))
    Tello.rotate_clockwise(drone, 180)

def trace(drone, frame_read, prior):
    # 往上，追線優先往上/左，追線往下鑽過table，追線優先往左/上，往左走看到marker，轉180 結束
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
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    while break_flag == False:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                if markerIDs[i][0] == 1:
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
                    if(abs(x)<=10 and abs(y)<=10 and abs(z-65)<=5): 
                        drone.send_rc_control(0, 0, 0, 0)
                        Tello.move(drone, "up", 25)
                        time.sleep(1)
                        break_flag = True
                        print("Break!")
        else:
            z_update = 0
            y_update = 0
            x_update = 0
            yaw_update = 0 
            
        key = cv2.waitKey(100)
        if key != -1:
            keyboard(drone, key)
        else:
            drone.send_rc_control(int(x_update) // 1, int(z_update)*2 // 5, int(y_update) // 1, 0)
        cv2.imshow("drone", frame)
    
    trace_with_prior(drone, frame_read, prior)
    print("fin")
    Tello.move(drone, "back", 20)
    time.sleep(1)
    trace_aruco(drone, frame_read)
    
def level3(drone, frame_read, leftOrRight):
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
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    while break_flag == False:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        markerCorners, markerIDs, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        print(markerIDs)
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIDs)
        fs = cv2.FileStorage("drone.xml", cv2.FILE_STORAGE_READ)
        intrinsic = fs.getNode("instrinsic").mat()
        distorsion = fs.getNode("distortion").mat()
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distorsion)
        # print(rvecs, tvecs)
        if rvecs is not None and tvecs is not None:
            i, x, y, z = 0,0,0,0
            if rvecs.shape[0]>=2:
                x1, y1, z1 = tvecs[0][0]
                x2, y2, z2 = tvecs[1][0]
                if(x2 < x1): 
                    x1, y1, z1, x2, y2, z2 = x2, y2, z2, x1, y1, z1
                if(leftOrRight==2):
                    x, y, z = x1, y1, z1 
                    i = 0
                else:
                    x, y, z = x2, y2, z2
                    i = 1
            else:
                x, y, z = tvecs[0][0]
                i=0
            if markerIDs[i][0] == 3:
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
                if(abs(x)<=10 and abs(y)<=10 and abs(z-65)<=5): 
                    drone.send_rc_control(0, 0, 0, 0)
                    break_flag = True
                    print("Break!")
        else:
            z_update = 0
            y_update = 0
            x_update = 0
            yaw_update = 0 
            
        key = cv2.waitKey(100)
        if key != -1:
            keyboard(drone, key)
        else:
            drone.send_rc_control(int(x_update) // 1, int(z_update)*2 // 5, int(y_update) // 1, 0)
        cv2.imshow("drone", frame)
    drone.land()


def main():
    # Tello
    drone = Tello()
    drone.connect()
    battery = drone.get_battery()
    print(f"Battery: {battery}%")
    drone.streamon()
    frame_read = drone.get_frame_read()

    while 1:
        frame = frame_read.frame
        cv2.imshow("frame", frame)
        key = cv2.waitKey(100)
        if key  == ord("p"):
            break
        elif key!=-1:
            keyboard(drone, key)
    level1(drone, frame_read) # 起飛，看人臉往上往前往下，看人臉往下往前往上
    prior = level2(drone, frame_read) # 識別娃娃，回傳決定追線二不同階段
    print("prior",prior)
    time.sleep(1)
    trace(drone, frame_read, prior) # 往上，追線優先往上/左，追線往下鑽過table，追線優先往左/上，往左走看到marker，轉180 結束
    # trace_aruco(drone, frame_read)
    time.sleep(1)
    leftOrRight = level2(drone, frame_read) # 識別娃娃，回傳決定追線二不同階段
    level3(drone, frame_read, leftOrRight) # 寫死往左前方或右前方後降落
    # while 1:
    #     frame = frame_read.frame
    #     cv2.waitKey(100)
    #     if(key != -1):
    #         keyboard(drone, key)
    #         # if(key == ord("m")):
    #         #     LEVEL += 1
    #         # if(key == ord("n")):
    #         #     LEVEL -= 1
    #     cv2.imshow("frame", frame)
if __name__ == '__main__':
    main()

