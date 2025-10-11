import cv2
import numpy as np
import os

def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def custom_gaussian_blur(image, kernel):
    h, w = image.shape
    k_size = kernel.shape[0]
    pad = k_size // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    result = np.zeros_like(image, dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+k_size, j:j+k_size]
            result[i, j] = np.sum(region * kernel)
    return result.astype(np.uint8)

cap = cv2.VideoCapture(r'Костя, сюда свой путь блять, не текст, путь до файла, полный путь, с шрифтом пкм по картинке', cv2.CAP_ANY)
ret, image = cap.read()
if not(ret):
    print("break")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for size in [3, 5, 7]:
    kernel = gaussian_kernel(size, sigma=1.0)
    print(f"Размер {size}x{size} (sigma=1.0):")
    print(np.round(kernel, 4))
    print(f"Сумма элементов: {np.sum(kernel):.6f}")
    print()

params = [
    (3, 0.5), (3, 2.0),  # Разные sigma для размера 3
    (5, 0.5), (5, 2.0)   # Разные sigma для размера 5
]

for size, sigma in params:
    kernel = gaussian_kernel(size, sigma)
    filtered = custom_gaussian_blur(gray, kernel)
    filename = f'blur_{size}_{sigma}.jpg'
    cv2.imwrite(filename, filtered)
    print(f"Создано: {filename}")

opencv_blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=2.0)
custom_blur = custom_gaussian_blur(gray, gaussian_kernel(5, 2.0))

cv2.imshow('Original', gray)
cv2.imshow('OpenCV Blur (5x5, sigma=2.0)', opencv_blur)
cv2.imshow('Custom Blur (5x5, sigma=2.0)', custom_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()

mse = np.mean((opencv_blur.astype(float) - custom_blur.astype(float)) ** 2)

results = [gray]
titles = ['Original']

for size, sigma in params:
    filtered = custom_gaussian_blur(gray, gaussian_kernel(size, sigma))
    results.append(filtered)
    titles.append(f'Custom {size}x{size} sigma={sigma}')

results.append(opencv_blur)
titles.append('OpenCV 5x5 sigma=2.0')

rows = 2
cols = 4
mosaic = np.zeros((gray.shape[0] * rows, gray.shape[1] * cols), dtype=np.uint8)

for i, img in enumerate(results):
    row = i // cols
    col = i % cols
    mosaic[row*gray.shape[0]:(row+1)*gray.shape[0], 
           col*gray.shape[1]:(col+1)*gray.shape[1]] = img

print("\n" + "="*50)
print("="*50)

for size in [3, 5, 7]:
    kernel = gaussian_kernel(size, sigma=1.0)
    print(f"\nМатрица Гаусса {size}x{size}:")
    for row in kernel:
        print("[" + " ".join([f"{x:.4f}" for x in row]) + "]")
    print(f"Сумма: {np.sum(kernel):.6f}")
