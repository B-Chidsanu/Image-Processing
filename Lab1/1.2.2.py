import cv2
# import numpy as np

img_1 = cv2.imread("nature.jpg")
img_2 = cv2.imread("natural1.jpg")

img_01 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
img_02 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
resize_img1 = cv2.resize(img_1, (1280, 823))
resize_img2 = cv2.resize(img_2, (1280, 823))

output_file = "video/video_test.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 1 frame / 1s
fps = 1

# สร้างวิดีโอที่จะเป็นไฟล์
out = cv2.VideoWriter(output_file, fourcc, fps,
                      (resize_img1.shape[1], resize_img1.shape[0]))

# สร้างลิสต์ของค่า weight ที่ใช้ในการผสมภาพ โดยเริ่มจาก 1.0 ลดลงจนถึง 0
w = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]


# สร้างวิดีโอที่ภาพเปลี่ยนแปลงตามค่า weight ของภาพทั้งสอง
for w1, w2 in zip(w, w[::-1]):
    image_result = cv2.addWeighted(resize_img1, w1, resize_img2, w2, 0)
    out.write(image_result)


# สร้างวิดีโอที่ภาพเปลี่ยนแปลงตามค่า weight ของภาพทั้งสองใหม่อีกครั้ง
for w2, w1 in zip(w, w[::-1]):
    image_result = cv2.addWeighted(resize_img1, w1, resize_img2, w2, 0)
    out.write(image_result)
    print(w1,w2)
    


out.release()
