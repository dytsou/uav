import cv2
import numpy as np

def warp_perspective(image, M, output_size):
    width, height = output_size
    output_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Invert the matrix M to map the output coordinates back to the input coordinates
    M_inv = np.linalg.inv(M)
    
    for y in range(height):
        for x in range(width):
            # Inverse transformation
            source_coords = M_inv @ np.array([x, y, 1])
            source_x = source_coords[0] / source_coords[2]
            source_y = source_coords[1] / source_coords[2]
            
            # Check coordinates are in bounds
            if 0 <= source_x < image.shape[1] and 0 <= source_y < image.shape[0]:
                # Get pixel values using bilinear interpolation
                output_image[y, x] = bilinear_interpolation(image, source_x, source_y)
    
    return output_image

def bilinear_interpolation(image, x, y):
    x1, y1 = int(x), int(y)
    x2, y2 = min(x1 + 1, image.shape[1] - 1), min(y1 + 1, image.shape[0] - 1)
    dx, dy = x - x1, y - y1
    
    top = (1 - dx) * image[y1, x1] + dx * image[y1, x2]
    bottom = (1 - dx) * image[y2, x1] + dx * image[y2, x2]
    
    return (1 - dy) * top + dy * bottom

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