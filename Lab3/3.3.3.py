import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from scipy import signal

# # Load the pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Load and preprocess the image
img = cv2.imread('./view.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img_reshape = img.reshape(1, img.shape[0], img.shape[1], 3)
img_copy = img.copy()
img_mean = np.array([123.68, 116.779, 103.939])
img_sub = img_reshape - img_mean


# Get the kernel and bias of the first convolutional layer
kernel, bias = model.layers[1].get_weights()

# Create a list to store the feature maps
feature_maps = []

# Convolve the preprocessed image with each kernel
for i in range(kernel.shape[-1]):
    conv_map = np.zeros_like(img_sub[0, :, :, 0])
    for j in range(3):
        conv_map += signal.convolve2d(img_sub[0, :, :, j], kernel[:, :, j, i], mode='same')
    conv_map = np.maximum(conv_map, 0)  # ReLU activation
    feature_maps.append(conv_map)

# Display the feature maps
plt.figure(figsize=(10, 10))
for i in range(len(feature_maps)):
    plt.subplot(8, 8, i + 1)
    plt.imshow(feature_maps[i], cmap='gray')
    plt.axis('off')
plt.show()