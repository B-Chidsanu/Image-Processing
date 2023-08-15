import cv2
import numpy as np


def adjust_gamma(image, gamma, a=1, b=0):
    image = image / 255
    image = cv2.pow(image, gamma)
    out = a * image + b
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)
    return out


image_path = './dog.jpg'
image = cv2.imread(image_path)

gammas = np.concatenate((np.linspace(0.1, 1, 10), np.linspace(1.1, 2, 10)))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
height, width = image.shape[:2]
out = cv2.VideoWriter('./video/test1.mp4', fourcc, 1, (width, height))

for gamma in gammas:
    adjusted_image = adjust_gamma(image, gamma)
    out.write(adjusted_image)

out.release()
