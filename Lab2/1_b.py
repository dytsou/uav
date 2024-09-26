import cv2
import numpy as np

def histogramlize(channel):
    histogram = np.zeros(256, dtype=int)
    for pixel in channel.flatten(): 
        histogram[pixel] += 1

    Axxsum = histogram.cumsum()

    # normalize
    his_normalize = ((Axxsum)/(Axxsum[255]))*255
    his_normalize = np.round(his_normalize).astype(np.uint8)

    modified_channel = his_normalize[channel]
    return modified_channel

img = cv2.imread('histogram.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(img_hsv)

new_v = histogramlize(v)

img_output = cv2.merge((h, s, new_v))

img_output = cv2.cvtColor(img_output, cv2.COLOR_HSV2BGR)

cv2.imwrite("1_b.jpg", img_output)
# cv2.imshow('Image', img_output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()