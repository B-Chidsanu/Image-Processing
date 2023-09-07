import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from numpy import expand_dims
from scipy import signal

# Load VGG16 model
model = VGG16()

# Load your image
img = cv2.imread('./view.jpg')
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = expand_dims(img, axis=0)
img = preprocess_input(img)
img_mean = np.array([123.68, 116.779, 103.939])
shifted_img = img - img_mean

kernels, bias = model.layers[1].get_weights()

print(kernels, bias)

img_result = np.zeros((224, 224, 64))


for i in range(kernels.shape[-1]):
    for channel in range(3):
        img_result[:, :, i] += signal.convolve2d(
            shifted_img[0, :, :, channel], kernels[:, :, channel, i], mode='same')


img_result[img_result < 0] = 0

square = 8
plt.figure(figsize=(10, 10))
for i in range(64):
    plt.subplot(square, square, i + 1)
    plt.imshow(img_result[:, :, i], cmap="gray")
    plt.axis('off')

plt.show()
