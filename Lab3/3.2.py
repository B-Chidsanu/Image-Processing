import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from numpy import expand_dims

# Load your image
model = VGG16()
img = cv2.imread('./bird.png')
img = cv2.resize(img, (224, 224))  # Fix the resize function
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_mean = np.array([123.68, 116.779, 103.939])
shifted_img = img - img_mean


plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original')

plt.subplot(2, 2, 2)
plt.imshow(shifted_img)
plt.title('Preprocess')

plt.show()
