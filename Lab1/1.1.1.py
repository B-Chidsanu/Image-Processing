import cv2
import matplotlib.pyplot as plt

#bgr image 
image = cv2.imread('apple.jpg')
b_color = image[:,:,0]
g_color = image[:,:,1]
r_color = image[:,:,2]

#rgb_image
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rr_color = rgb_image[:,:,0]
gg_color = rgb_image[:,:,1]
bb_color = rgb_image[:,:,2]
#can't
plt.figure(figsize=(15, 6))


plt.subplot(2,4,1)
plt.imshow(image)
plt.title('BGR')

plt.subplot(2, 4, 2)
plt.imshow(b_color, cmap="gray")
plt.title('B')

plt.subplot(2, 4, 3)
plt.imshow(g_color, cmap="gray")
plt.title('G')

plt.subplot(2, 4, 4)
plt.imshow(r_color, cmap="gray")
plt.title('R')

plt.subplot(2, 4, 5)
plt.imshow(rgb_image)
plt.title('RGB')

plt.subplot(2, 4, 6)
plt.imshow(rr_color, cmap="gray")
plt.title('R')

plt.subplot(2, 4, 7)
plt.imshow(gg_color, cmap="gray")
plt.title('G')

plt.subplot(2, 4, 8)
plt.imshow(bb_color, cmap="gray")
plt.title('B')


plt.show()