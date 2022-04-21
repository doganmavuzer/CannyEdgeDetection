import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_kernel_x(size, sigma):
    if size % 2 == 0:
        size = size + 1

    max_point = size // 2  # both directions (x,y) maximum cell start point
    min_point = -max_point  # both directions (x,y) minimum cell start point

    K_x = np.zeros((size, size))  # kernel matrix
    for x in range(min_point, max_point + 1):
        for y in range(min_point, max_point + 1):
            value = (-x / (2 * np.pi * (sigma ** 4)) * np.exp((-(x ** 2 + y ** 2)) / (2 * (sigma ** 2))))
            K_x[x - min_point, y - min_point] = value

    return K_x


def gaussian_kernel_y(size, sigma):
    if size % 2 == 0:
        size = size + 1

    max_point = size // 2  # both directions (x,y) maximum cell start point
    min_point = -max_point  # both directions (x,y) minimum cell start point

    K_y = np.zeros((size, size))  # kernel matrix
    for x in range(min_point, max_point + 1):
        for y in range(min_point, max_point + 1):
            value = (-y / (2 * np.pi * (sigma ** 4)) * np.exp((-(x ** 2 + y ** 2)) / (2 * (sigma ** 2))))
            K_y[x - min_point, y - min_point] = value

    return K_y


img = cv2.imread("images/tiger.jpg", cv2.IMREAD_GRAYSCALE)
img = np.float64(img)

I_x = cv2.filter2D(img, -1, gaussian_kernel_x(3, 0.5))
I_y = cv2.filter2D(img, -1, gaussian_kernel_y(3, 0.5))

plt.figure(1)
plt.subplot(131)
plt.imshow(img, cmap="gray")
plt.title("Gaussian Image")

plt.subplot(132)
plt.imshow(I_x, cmap="gray")
plt.title("X-Derivative")

plt.subplot(133)
plt.imshow(I_y, cmap="gray")
plt.title("Y-Derivative")
plt.show()
