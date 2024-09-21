import numpy as np
import cv2

img = cv2.imread('test.jpg')
b, g, r = cv2.split(img)
mask = (b + g) * 0.3 > r

contrast = int(input('Contrast: '))
brightness = int(input('Brightness: '))
new_img = np.array(img, dtype=np.int32)
new_img[mask] = (new_img[mask] - 127) * (contrast / 127 + 1) + 127 + brightness
new_img = np.clip(new_img, 0, 255)
result = np.array(new_img, dtype=np.uint8)

cv2.imshow('image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('1-2.png', result)