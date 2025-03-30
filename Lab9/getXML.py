import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard

boardSize = (9, 6)
obj_p = np.zeros((boardSize[0]*boardSize[1], 3), np.float32)
obj_p[:, :2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2) #3D to 2D

imagePoints = []
objectPoints = []

found = 1

drone = Tello()
drone.connect()
battery = drone.get_battery()
print(f"Battery: {battery}%")
drone.streamon()
frame_read = drone.get_frame_read()
while(True):
    frame = frame_read.frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if not c_ret:
    #     break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    d_ret, corners = cv2.findChessboardCorners(img, boardSize, None)
    if d_ret:
        print("chesboard found:", found)
        found = found+1

        cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        objectPoints.append(obj_p)
        imagePoints.append(corners)
        cv2.drawChessboardCorners(frame, boardSize, corners, d_ret)
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(33) & 0xFF == ord("q"):
        break
    

if found >= 4:
    imageSize = (frame.shape[1], frame.shape[0])
    #an extra None for OpenCV >= 3.0
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, imageSize, None, None)

    f = cv2.FileStorage("1.xml", cv2.FILE_STORAGE_WRITE)
    f.write("instrinsic", cameraMatrix)
    f.write("distortion", distCoeffs)
    f.release

cv2.destroyAllWindows()