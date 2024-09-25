import cv2
import numpy as np

img = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)
w = img.shape[0]
h = img.shape[1]
cur_label = np.uint32(0)
INF = ~np.uint32(0)
labelParent = {INF:INF, cur_label:cur_label}

label = np.zeros([w,h], np.uint32)
label = ~label
def join(a,b): #a<b
    labelParent[b] = labelParent[a] = labelParent[labelParent[a]]

def searchN(x,y):
    global label
    global labelParent
    global cur_label
    global img
    if img[x][y] > 127:
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                if i < w and i >= 0 and j < h and j >= 0 and (i != x or j != y):
                    if label[x][y] > label[i][j]:
                        if label[x][y] != INF:
                            join(label[i][j],label[x][y])
                        label[x][y] = label[i][j]
                    elif label[x][y] < label[i][j]:
                        if label[i][j] != INF:
                            join(label[x][y],label[i][j])
        if label[x][y] == INF:
            label[x][y] = cur_label
            cur_label = cur_label + 1
            labelParent[cur_label] = cur_label
            # print(cur_label)
        
for i in range(w):
    for j in range(h):
        searchN(i,j)
        
for i in range(0,cur_label):
    if labelParent[labelParent[i]] != labelParent[i]:
        labelParent[i] = labelParent[labelParent[i]]

for i in range(w):
    for j in range(h):
        label[i][j] = labelParent[label[i][j]]
# print(np.unique(label).shape)
cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()

clr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
rdclr = {}
for i in range(cur_label):
    B = np.random.randint(0,256)
    G = np.random.randint(0,256)
    R = np.random.randint(0,256)
    rdclr[np.uint32(i)] = np.array([B,G,R], np.uint8)

for i in range(w):
    for j in range(h):
        if label[i,j] == INF:
            clr[i,j] = np.array([0,0,0], np.uint8)
        else:
            clr[i,j] = np.array(rdclr[label[i,j]], np.uint8)

cv2.imshow('image', clr)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('3.jpg', clr)