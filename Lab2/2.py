import cv2
import numpy as np

img = cv2.imread('input.jpg')
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, result = cv2.threshold(grey_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('2.jpg', result)