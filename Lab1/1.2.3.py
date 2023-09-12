import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img_1 = cv.imread("nature.jpg")
img2 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)

# ปรับขนาดภาพ
resize_img2 = cv.resize(img2, (1980, 1024))

plt.figure(figsize=(12, 3))

# แสดงภาพต้นฉบับ
plt.subplot(1, 3, 1)
plt.imshow(resize_img2)
plt.title('Original')

height, width = resize_img2.shape[:2]

# สร้างกรอบแมสก์ขนาดเดียวกับภาพที่อยู่ในกรอบสี่เหลี่ยม
mask = np.zeros((height, width), dtype=np.uint8)
cv.rectangle(mask, (1700, 200), (1000, 700), 255, -1)

# แสดงกรอบแมสก์
plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Image Mask')

# ประมวลผลภาพด้วยการกำหนดแมสก์
masked_image = cv.bitwise_and(resize_img2, resize_img2, mask=mask)

# แสดงภาพที่เสร็จแล้ว
plt.subplot(1, 3, 3)
plt.imshow(masked_image)
plt.title('Bitwise_AND()')

plt.show()
