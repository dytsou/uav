import cv2
import numpy as np

img = cv2.imread('input.jpg')
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([grey_img], [0], None, [256], [0, 256])
total = grey_img.shape[0] * grey_img.shape[1]
sum_total, sum_background, background_pixel, foreground_pixel, var_max, threshold = 0, 0, 0, 0, 0, 0
for i in range(256):
    sum_total += i * hist[i]
    
for i in range(256):
    background_pixel += hist[i]
    if background_pixel == 0:
        continue
    foreground_pixel = total - background_pixel
    if foreground_pixel == 0:
        break
    
    sum_background += i * hist[i]
    mean_background = sum_background / background_pixel
    mean_foreground = (sum_total - sum_background) / foreground_pixel
    var_between = background_pixel * foreground_pixel * (mean_background - mean_foreground) ** 2
    if var_between > var_max:
        var_max = var_between
        threshold = i
        
print(threshold) # 118

_, result = cv2.threshold(grey_img, threshold, 255, cv2.THRESH_BINARY)

cv2.imshow('image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('2.jpg', result)