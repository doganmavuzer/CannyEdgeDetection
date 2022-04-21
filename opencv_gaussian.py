import cv2
import matplotlib.pyplot as plt


img = cv2.imread("images/zebras.jpg", cv2.IMREAD_GRAYSCALE)

kernel = cv2.getGaussianKernel(11, 2)

kernel = kernel.dot(kernel.T)

img_gaussian = cv2.filter2D(img, -1, kernel)

plt.figure(1)
plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.title("Original Image")

plt.subplot(122)
plt.imshow(img_gaussian, cmap="gray")
plt.title("Gaussian Filter Image")
plt.show()
