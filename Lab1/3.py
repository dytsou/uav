import cv2
import numpy as np

img = cv2.imread('ive.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 0)

Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

grad_x = cv2.filter2D(img, -1, Gx).astype(np.int32)
grad_y = cv2.filter2D(img, -1, Gy).astype(np.int32)

G = np.sqrt(grad_x**2 + grad_y**2)
G = G / G.max() * 255
result = np.uint8(np.clip(G, 0, 255))

cv2.imshow('image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('3.png', result)