import cv2
import numpy as np

def canny_manual(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('1. Original Grayscale Image', img)

    blurred = cv2.GaussianBlur(img, (5, 5), 1.4)
    cv2.imshow('2. Gaussian Blur', blurred)

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)

    Gx = cv2.filter2D(blurred, cv2.CV_32F, sobel_x)
    Gy = cv2.filter2D(blurred, cv2.CV_32F, sobel_y)

    magnitude = cv2.magnitude(Gx, Gy)
    angle = cv2.phase(Gx, Gy, angleInDegrees=True)
    cv2.imshow('3. Gradient Magnitude', magnitude / magnitude.max())
    cv2.imshow('4. Gradient Angle', angle / 360)

    Z = np.zeros(magnitude.shape, dtype=np.uint8)
    angle_q = np.round(angle / 45) * 45

    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            q = 255
            r = 255
            if (0 <= angle_q[i, j] < 45) or (315 <= angle_q[i, j] <= 360):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 45 <= angle_q[i, j] < 135:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 135 <= angle_q[i, j] < 225:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif 225 <= angle_q[i, j] < 315:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                Z[i, j] = magnitude[i, j]
            else:
                Z[i, j] = 0

    cv2.imshow('5. Non-Maximum Suppression', Z)

    high = np.max(Z) * 0.2
    low = high * 0.5
    res = np.zeros_like(Z, dtype=np.uint8)

    strong = 255
    weak = 50

    strong_i, strong_j = np.where(Z >= high)
    weak_i, weak_j = np.where((Z <= high) & (Z >= low))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    for i in range(1, res.shape[0] - 1):
        for j in range(1, res.shape[1] - 1):
            if res[i, j] == weak:
                if np.any(res[i-1:i+2, j-1:j+2] == strong):
                    res[i, j] = strong
                else:
                    res[i, j] = 0

    cv2.imshow('6. Hysteresis Thresholding', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

canny_manual('Untitled10.png')
