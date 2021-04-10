import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

# Read image
img = cv2.imread("test_img.jpeg", cv2.IMREAD_GRAYSCALE)

# Resize the image for faster processing / less memory
resize_factor = 0.1
im_dim = np.shape(img)
img = cv2.resize(img, (int(im_dim[1]*resize_factor), int(im_dim[0]*resize_factor)))

im_dim = np.shape(img)[0:2]
N = im_dim[0]*im_dim[1]

# Vectorize the image
img_vec = np.reshape(img, (N, 1))

# Compressively sense the image
compression_factor = 0.1
M = int(np.floor(compression_factor * N))

A_gauss = (np.random.randn(M, N))
# A_gauss = np.transpose(scipy.linalg.orth(np.transpose(A_gauss)))
# print(np.shape(A_gauss))
y_gauss = np.matmul(A_gauss, img_vec)

y_img = np.reshape(np.matmul(np.transpose(A_gauss), y_gauss), (im_dim[0], im_dim[1]))

plt.figure()
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.subplot(122)
plt.imshow(y_img, cmap='gray')
plt.title("Proxy")
plt.show()
