import numpy as np
import cv2
import math

oimg = cv2.imread('ive.jpg')

zoom = float(input("ratio:"))
oh = oimg.shape[0]
ow = oimg.shape[1]
h = int(oh * zoom)
w = int(ow * zoom)

q2img = np.zeros([h, w, 3], np.uint8)

x = np.linspace(0, oh - 1, h)
y = np.linspace(0, ow - 1, w)

x1 = np.floor(x).astype(int)
x2 = np.clip(x1 + 1, 0, oh - 1)
y1 = np.floor(y).astype(int)
y2 = np.clip(y1 + 1, 0, ow - 1)

tx = x - x1
ty = y - y1
Ia = oimg[x1[:, None], y1[None,:], 1]
for k in range(3):
    q2img[:, :, k] = \
    (1 - ty)[None, :] * (1 - tx)[:,None] * oimg[x1[:,None], y1[None,:], k] + \
    (1 - ty)[None, :] * tx[:,None] * oimg[x2[:, None], y1[None, :], k] + \
    ty[None, :] * (1 - tx)[:,None] * oimg[x1[:, None], y2[None, :], k] + \
    ty[None, :] * tx[:,None] * oimg[x2[:, None], y2[None, :], k]

cv2.imshow("q2", q2img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('2.png', q2img)