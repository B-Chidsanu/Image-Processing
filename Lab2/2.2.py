import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = cv2.imread('./cat.jpg')
image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)


r, g, b = cv2.split(image)


r_eq = cv2.equalizeHist(r)
g_eq = cv2.equalizeHist(g)
b_eq = cv2.equalizeHist(b)


equalized_image = cv2.merge((b_eq, g_eq, r_eq))


hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

hist_r_eq = cv2.calcHist([r_eq], [0], None, [256], [0, 256])
hist_g_eq = cv2.calcHist([g_eq], [0], None, [256], [0, 256])
hist_b_eq = cv2.calcHist([b_eq], [0], None, [256], [0, 256])

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image)


plt.subplot(2, 2, 3)
plt.title('Equalized Image')
plt.imshow(equalized_image)


plt.subplot(2, 2, 2)
plt.title('Histogram - Red Channel')
plt.plot(hist_r, color='red')
plt.plot(hist_g, color='green')
plt.plot(hist_b, color='blue')


plt.subplot(2, 2, 4)
plt.title('Histogram - Green Channel')
plt.plot(hist_r_eq, color='red')
plt.plot(hist_g_eq, color='green')
plt.plot(hist_b_eq, color='blue')


plt.tight_layout()
plt.show()
