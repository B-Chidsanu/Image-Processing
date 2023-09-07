import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from scipy import signal

model = VGG16()
model.layers[1].get_config()
model = Model(inputs=model.inputs, outputs=model.layers[1].output)
model.summary()


img = cv2.imread('./view.jpg')
img = cv2.resize(img, (224, 224))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)
feature_maps = model.predict(img)

square = 8
x = 1
plt.figure(figsize=(10, 10))
for i in range(square):
    for j in range(square):
        ax = plt.subplot(square, square, x)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(feature_maps[0, :, :, x-1], cmap="gray")
        x += 1
plt.show()
