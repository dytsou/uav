import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard
from face_detection import face_detection
from detectDoll import detectDoll

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
        frame, cx, cy, dis = face_detection(frame, 30, f_y)
        if (state == 0 or state == 2):
            if (cx > 0): Tello.move(drone, "left", 20)
            elif (cx < 0): Tello.move(drone, "right", 20)
            elif (cy > 0): Tello.move(drone, "up", 20)
            elif (cy < 0): Tello.move(drone, "down", 20)
            elif(dis > 30): Tello.move(drone, "forward", 20)
            else: state += 1
        elif (state == 1):
            Tello.move(drone, "up", 100)
            Tello.move(drone, "forward", 100)
            Tello.move(drone, "down", 100)
            state += 1
        elif (state == 3):
            Tello.move(drone, "down", 100)
            Tello.move(drone, "forward", 100)
            Tello.move(drone, "up", 100)
            break
        
        cv2.imshow("drone", frame)

def level2(drone, frame_read):
    # 識別娃娃，回傳決定追線二不同階段
    # todo:放入lab8的東東
    while(1):
        frame = frame_read.frame
        doll = detectDoll(frame) #1 Carno, 2 Melody, 0 no detect
        if(doll != 0):
            return doll; 
    return 0

def trace_prior_up(drone, frame_read, break_cond, init_angle):
    # 優先往上飛(轉彎)，追線，cnt >= break_cond時break
    cnt = 0

def trace_prior_left(drone, frame_read, break_cond, init_angle):
    # 優先往左飛(直走)，追線，cnt >= break_cond時break
    cnt = 0
    
def trace_until_aruco(drone, frame_read):
    # 往左追線，看到marker時轉180
    dummy = 0

def trace(drone, frame_read, prior):
    # 往上，追線優先往上/左，追線往下鑽過table，追線優先往左/上，往左走看到marker，轉180 結束
    if(prior == 1):
        trace_prior_up(drone, frame_read, 5, 0)
        trace_prior_left(drone, frame_read, 4, 270)
        trace_until_aruco(drone, frame_read)
    elif(prior == 2):
        trace_prior_left(drone, frame_read, 2, 0)
        trace_prior_up(drone, frame_read, 7, 180)
        trace_until_aruco(drone, frame_read)
    else:
        print("unknown prior")
    
    
def level3(drone, frame_read, leftOrRight):
    # 寫死往左前方或右前方後降落
    if(leftOrRight == 1): Tello.move(drone, "left", 50)
    if(leftOrRight == 2): Tello.move(drone, "right", 50)
    Tello.move(drone, "forward", 180)


def main():
    # Tello
    drone = Tello()
    drone.connect()
    battery = drone.get_battery()
    print(f"Battery: {battery}%")
    drone.streamon()
    frame_read = drone.get_frame_read()

    level1(drone, frame_read) # 起飛，看人臉往上往前往下，看人臉往下往前往上
    prior = level2(drone, frame_read) # 識別娃娃，回傳決定追線二不同階段
    trace(drone, frame_read, prior) # 往上，追線優先往上/左，追線往下鑽過table，追線優先往左/上，往左走看到marker，轉180 結束
    leftOrRight = level2(drone, frame_read) # 識別娃娃，回傳決定追線二不同階段
    level3(drone, frame_read, leftOrRight) # 寫死往左前方或右前方後降落

if __name__ == '__main__':
    main()

