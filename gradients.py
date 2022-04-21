import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters


def f_NMS(Gm, Gd):
    num_rows, num_cols = Gm.shape[0], Gm.shape[1]
    Gd_bins = 45 * (np.round(Gd / 45))

    G_NMS = np.zeros(Gm.shape)

    neighbor_a, neighbor_b = 0., 0.

    for r in range(1, num_rows - 1):
        for c in range(1, num_cols - 1):
            angle = Gd_bins[r, c]
            if angle == 180. or angle == -180. or angle == 0.:
                neighbor_a, neighbor_b = Gm[r + 1, c], Gm[r - 1, c]
            elif angle == 90. or angle == -90.:
                neighbor_a, neighbor_b = Gm[r, c - 1], Gm[r, c + 1]
            elif angle == 45. or angle == -135.:
                neighbor_a, neighbor_b = Gm[r + 1, c + 1], Gm[r - 1, c - 1]
            elif angle == -45. or angle == 135.:
                neighbor_a, neighbor_b = Gm[r - 1, c + 1], Gm[r + 1, c - 1]
            else:
                print("error")
                return

            if Gm[r, c] > neighbor_a and Gm[r, c] > neighbor_b:
                G_NMS[r, c] = Gm[r, c]

    return G_NMS


img = cv2.imread("images/zebras.jpg", cv2.IMREAD_GRAYSCALE)

kernel = cv2.getGaussianKernel(11, 2)

kernel = kernel.dot(kernel.T)

img_gaussian = cv2.filter2D(img, -1, kernel)

img_gaussian = np.float64(img_gaussian)

mask_x = np.zeros((2, 1))
mask_x[0] = -1
mask_x[1] = 1

I_x = cv2.filter2D(img_gaussian, -1, mask_x)
mask_y = mask_x.T
I_y = cv2.filter2D(img_gaussian, -1, mask_y)

Gm = (I_x ** 2 + I_y ** 2) ** 0.5
Gd = np.rad2deg(np.arctan2(I_y, I_x))

bins = np.array([-180., -135., -90., -45., 0., 45., 90., 135., 180.])

inds = np.digitize(Gd, bins) - 1
Gd_bin = bins[inds.flatten()].reshape(Gd.shape)

print(Gd.max(), Gd.min())
G_NMS = f_NMS(Gm, Gd)

L = G_NMS.mean()
H = L + G_NMS.std()
E = filters.apply_hysteresis_threshold(G_NMS, L, H)

plt.figure(1)
# plt.subplot(131)
# plt.imshow(img_gaussian, cmap="gray")
# plt.title("Gaussian Image")

# plt.subplot(132)
# plt.imshow(I_x, cmap="gray")
# plt.title("X-Derivative")

# plt.subplot(133)
# plt.imshow(I_y, cmap="gray")
# plt.title("Y-Derivative")

# plt.subplot(121)
# plt.imshow(Gm, cmap="gray")
# plt.title("Gradient Magnitude")

# plt.subplot(122)
# plt.imshow(Gd, cmap="gray")
# plt.title("Gradient Direction")

plt.subplot(121)
plt.imshow(G_NMS>15, cmap="gray")
plt.title("NMS")

plt.subplot(122)
plt.imshow(E, cmap="gray")
plt.title("Canny Edge Detection")

plt.show()
