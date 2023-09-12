import cv2
import matplotlib.pyplot as plt

# อ่านรูปภาพ
img = cv2.imread("nature.jpg")
# แปลงรูปภาพไปเป็นRGB และ slicing สี
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
RGB_img_red = img2[:, :, 0]
RGB_img_green = img2[:, :, 1]
RGB_img_blue = img2[:, :, 2]

plt.figure(figsize=(12, 8))
# แสดงรูปภาพ RGB ที่ slicing
plt.subplot(4, 4, 1)
plt.imshow(img2)
plt.title('RGB')

plt.subplot(4, 4, 2)
plt.imshow(RGB_img_red, cmap="gray")
plt.title('R')

plt.subplot(4, 4, 3)
plt.imshow(RGB_img_green, cmap="gray")
plt.title('G')

plt.subplot(4, 4, 4)
plt.imshow(RGB_img_blue, cmap="gray")
plt.title('B')

# แปลงรูปภาพไปเป็น HSV และ slicing สี
HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
HSV_img_h = HSV_img[:, :, 0]
HSV_img_s = HSV_img[:, :, 1]
HSV_img_v = HSV_img[:, :, 2]

# แสดงรูปภาพ HSV ที่ slicing
plt.subplot(4, 4, 5)
plt.imshow(HSV_img)
plt.title('HSV')

plt.subplot(4, 4, 6)
plt.imshow(HSV_img_h, cmap="gray")
plt.title('H')

plt.subplot(4, 4, 7)
plt.imshow(HSV_img_s, cmap="gray")
plt.title('S')

plt.subplot(4, 4, 8)
plt.imshow(HSV_img_v, cmap="gray")
plt.title('V')

# แปลงรูปภาพไปเป็น HLS และ slicing สี
HLS_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
HLS_img_h = HLS_img[:, :, 0]
HLS_img_l = HLS_img[:, :, 1]
HLS_img_s = HLS_img[:, :, 2]

# แสดงรูปภาพ HLS ที่ slicing
plt.subplot(4, 4, 9)
plt.imshow(HLS_img)
plt.title('HLS')

plt.subplot(4, 4, 10)
plt.imshow(HLS_img_h, cmap="gray")
plt.title('H')

plt.subplot(4, 4, 11)
plt.imshow(HLS_img_l, cmap="gray")
plt.title('L')

plt.subplot(4, 4, 12)
plt.imshow(HLS_img_s, cmap="gray")
plt.title('S')

# แปลงรูปภาพไปเป็น YCrCb และ Slicing สี
YCrCb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
YCrCb_img_Y = YCrCb_img[:, :, 0]
YCrCb_img_Cr = YCrCb_img[:, :, 1]
YCrCb_img_Cb = YCrCb_img[:, :, 2]

# แสดงรูปภาพ YCrCb ที่ slicing
plt.subplot(4, 4, 13)
plt.imshow(YCrCb_img)
plt.title('YCrCb')

plt.subplot(4, 4, 14)
plt.imshow(YCrCb_img_Y, cmap="gray")
plt.title('Y')

plt.subplot(4, 4, 15)
plt.imshow(YCrCb_img_Cr, cmap="gray")
plt.title('Cr')

plt.subplot(4, 4, 16)
plt.imshow(YCrCb_img_Cb, cmap="gray")
plt.title('Cb')

plt.show()
