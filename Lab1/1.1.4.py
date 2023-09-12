import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('apple.jpg', cv2.IMREAD_GRAYSCALE)

height, width = image.shape

x = np.arange(0, width, 1)
y = np.arange(0, height, 1)
x, y = np.meshgrid(x, y)

z = image / 200.0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='gray')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Intensity')
plt.show()

image = cv2.imread('apple.jpeg')
bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
ycrcb_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

# List of image channels and corresponding titles
channels = [
    (bgr_image, 'RGB', ['R', 'G', 'B']),
    (hsv_image, 'HSV', ['H', 'S', 'V']),
    (hls_image, 'HLS', ['H', 'L', 'S']),
    (ycrcb_image, 'YCrCb', ['Y', 'Cr', 'Cb'])
]

plt.figure(figsize=(15, 10))

for i, (img, title, channel_names) in enumerate(channels):
    for j, channel in enumerate(cv2.split(img)):
        plt.subplot(4, 4, i * 4 + j + 1)
        plt.imshow(channel, cmap='gray')
        plt.title(f"{title} - {channel_names[j]}")

plt.show()