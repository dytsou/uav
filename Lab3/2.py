import cv2
import numpy as np

def warp_perspective(image, M, output_size):
    width, height = output_size
    y, x = np.indices((height, width))
    homo_coords = np.stack([x, y, np.ones_like(x)], axis=-1) # [:, :, 0] = x, [:, :, 1] = y, [:, :, 2] = 1
    
    # Invert the matrix M to map the output coordinates back to the input coordinates
    M_inv = np.linalg.inv(M)
    
    src_coords = homo_coords @ M_inv.T
    src_coords /= src_coords[:, :, 2:3]
    src_x, src_y = src_coords[:, :, 0], src_coords[:, :, 1]
    mask = (src_x >= 0) & (src_x < image.shape[1] - 1) & (src_y >= 0) & (src_y < image.shape[0] - 1)
    
    output_image = np.zeros((height, width, 3), dtype=np.uint8)
    valid_y, valid_x = np.where(mask)
    output_image[valid_y, valid_x] = bilinear_interpolation(image, src_x[mask], src_y[mask])
    
    return output_image

def bilinear_interpolation(image, x, y):
    x1, y1 = np.floor(x).astype(int), np.floor(y).astype(int)
    x2, y2 = np.minimum(x1 + 1, image.shape[1] - 1), np.minimum(y1 + 1, image.shape[0] - 1)
    dx, dy = x - x1, y - y1
    
    top_left = image[y1, x1]
    top_right = image[y1, x2]
    bottom_left = image[y2, x1]
    bottom_right = image[y2, x2]
    
    top = (1 - dx)[:, None] * top_left + dx[:, None] * top_right
    bottom = (1 - dx)[:, None] * bottom_left + dx[:, None] * bottom_right
    return ((1 - dy)[:, None] * top + dy[:, None] * bottom).astype(np.uint8)

def transform(img):
    ref_points = np.array([[0, 0], [0, 1080], [1920, 1080], [1920, 0]], np.float32)
    new_points = np.array([[415, 868], [332, 1410], [1648, 1254], [1634, 220]], np.float32)
    m = cv2.getPerspectiveTransform(ref_points, new_points)
    print(m)
    output = warp_perspective(img, m, (2634, 1888))
    # output = cv2.warpPerspective(img, m, (2634, 1888))
    return output

def combine(img, frame):
    frame = cv2.resize(frame, (img.shape[1], img.shape[0]))
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(grey, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    mask_inv = mask_inv.astype(np.uint8)
    
    img = cv2.bitwise_and(img, img, mask=mask_inv)
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    combine = cv2.add(img, frame)
    return combine

def main():
    cap = cv2.VideoCapture(0)
    img = cv2.imread('screen.jpg')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
        frame = cv2.resize(frame, (1920, 1080))
        frame = transform(frame)
        combine_img = combine(img, frame)
        combine_img = cv2.resize(combine_img, (800, 600))
        cv2.imshow('combine', combine_img)
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()