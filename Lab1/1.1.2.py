import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the image
image = cv2.imread('apple.jpg')
Original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Transpose the image
transpose = Original_image.transpose()

# Move the axes of the image
moveaxis = np.moveaxis(Original_image, 2, 0)

# Reshape the image
h, w, ch = Original_image.shape
reshape = Original_image.reshape((ch, w, h))

# Display the images
plt.figure(figsize=(15, 6))

plt.subplot(1, 4, 1)
plt.imshow(Original_image[:, :, 0], cmap='gray')
print(f'Original Y(H, W, CH) --> {Original_image.shape}')
plt.title("Original Image")

plt.subplot(1, 4, 2)
plt.imshow(transpose[0, :, :], cmap='gray')
print(f'Transpose Y(CH, W, H) --> {transpose.shape}')
plt.title("Transposed Image")

plt.subplot(1, 4, 3)
plt.imshow(moveaxis[0, :, :], cmap='gray')
print(f'Moveaxis Y(CH, H, W) --> {moveaxis.shape}')
plt.title("Moveaxis Image")

plt.subplot(1, 4, 4)
plt.imshow(reshape[0, :, :], cmap='gray')
print(f'Reshape Y(CH, H, W) --> {reshape.shape}')
plt.title("Reshape Image")

plt.show()
