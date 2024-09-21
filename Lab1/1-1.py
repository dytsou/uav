import numpy as np
import cv2

img = cv2.imread('test.jpg')

b, g, r = cv2.split(img)
blue_mask = (b > 100) & (b*0.6 > g) & (b*0.6 > r)
img_blue = cv2.bitwise_and(img, img, mask=blue_mask.astype(np.uint8))

img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_grey_forCombine = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)

result = np.where(img_blue == 0, img_grey_forCombine, img_blue)


cv2.imshow('image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('1-1.png', result)