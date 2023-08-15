import cv2
import numpy as np

image = cv2.imread("./dog.jpg")
height, width, channels = image.shape


def adjust_pixel(image, a, b):
    new_image = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            for l in range(channels):
                x = a * image[i, j, l] + b
                if x > 255:
                    x = 255
                elif x < 0:
                    x = 0
                new_image[i, j, l] = x  
    return new_image


images = []
for a in np.linspace(0.5, 1.5, 5):
    for b in np.linspace(-50, 50, 4):
        images.append(adjust_pixel(image, a, b))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./video/test.mp4', fourcc, 1, (width, height))

for img in images:
    out.write(img)
out.release()
