import cv2
import numpy as np

def histogramlize(channel):
    histogram = np.zeros(256, dtype=int)
    for pixel in channel.flatten(): 
        histogram[pixel] += 1

    Axxsum = histogram.cumsum()

    # normalize
    his_normalize = ((Axxsum)/(Axxsum[255]))*255
    his_normalize = np.round(his_normalize).astype(int)

    modified_channel = his_normalize[channel]
    return modified_channel

img = cv2.imread('histogram.jpg')

b, g, r = cv2.split(img)

his_b = histogramlize(b)
his_g = histogramlize(g)
his_r = histogramlize(r)

img_output = cv2.merge((his_b, his_g, his_r))
img_output = np.clip(img_output, 0, 255).astype(np.uint8)

cv2.imwrite("1_a.jpg", img_output)
# cv2.imshow('Image', img_output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()