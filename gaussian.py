import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_kernel(size, sigma):
    if size % 2 == 0:
        size = size + 1

    max_point = size // 2  # both directions (x,y) maximum cell start point
    min_point = -max_point  # both directions (x,y) minimum cell start point

    K = np.zeros((size, size))  # kernel matrix
    for x in range(min_point, max_point + 1):
        for y in range(min_point, max_point + 1):
            value = (1 / (2 * np.pi * (sigma ** 2)) * np.exp((-(x ** 2 + y ** 2)) / (2 * (sigma ** 2))))
            K[x - min_point, y - min_point] = value

    return K


img = cv2.imread("images/zebras.jpg", cv2.IMREAD_GRAY)

kernel = gaussian_kernel(11, 2)

img_gaussian = cv2.filter2D(img, -1, kernel)

plt.figure(1)
plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.title("Original Image")

plt.subplot(122)
plt.imshow(img_gaussian, cmap="gray")
plt.title("Gaussian Filter Image")
plt.show()
