import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image_path
original_image = cv2.imread('./nature.jpg')
template_image = cv2.imread('./waterfall.jpg')


def Normalized_Histogram(image):
    pdf = []
    for channel in range(3):
        hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
        pdf.append(hist/hist.sum())
    return pdf


original_pdf = Normalized_Histogram(original_image)
template_pdf = Normalized_Histogram(template_image)


def Cummulative_Histogram(pdf):
    return [np.cumsum(p) for p in pdf]


original_cdf = Cummulative_Histogram(original_pdf)
template_cdf = Cummulative_Histogram(template_pdf)


def histogram_matching(original_cdf, template_cdf, image):
    matched_image = np.zeros_like(image)

    for channel in range(3):
        for i in range(256):
            diff = np.abs(original_cdf[channel][i] - template_cdf[channel])
            min_diff_index = np.argmin(diff)
            matched_image[:, :, channel][image[:,
                                               :, channel] == i] = min_diff_index

    return matched_image


matched_image = histogram_matching(original_cdf, template_cdf, original_image)
matched_pdf = Normalized_Histogram(matched_image)
matched_cdf = Cummulative_Histogram(matched_pdf)

plt.figure(figsize=(12, 6))

plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('original image')

plt.subplot(3, 3, 2)
plt.plot(original_pdf[0], color='red')
plt.plot(original_pdf[1], color='green')
plt.plot(original_pdf[2], color='blue')
plt.title('orginal PDF')

plt.subplot(3, 3, 3)
plt.plot(original_cdf[0], color='red')
plt.plot(original_cdf[1], color='green')
plt.plot(original_cdf[2], color='blue')
plt.title('original CDF')

plt.subplot(3, 3, 4)
plt.imshow(cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB))
plt.title('template image')

plt.subplot(3, 3, 5)
plt.plot(template_pdf[0], color='red')
plt.plot(template_pdf[1], color='green')
plt.plot(template_pdf[2], color='blue')
plt.title('template PDF')

plt.subplot(3, 3, 6)
plt.plot(template_cdf[0], color='red')
plt.plot(template_cdf[1], color='green')
plt.plot(template_cdf[2], color='blue')
plt.title('original CDF')

plt.subplot(3, 3, 7)
plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
plt.title('Matched Image')


plt.subplot(3, 3, 8)
plt.plot(matched_pdf[0], color='red')
plt.plot(matched_pdf[1], color='green')
plt.plot(matched_pdf[2], color='blue')
plt.title('matched PDF')

plt.subplot(3, 3, 9)
plt.plot(matched_cdf[0], color='red')
plt.plot(matched_cdf[1], color='green')
plt.plot(matched_cdf[2], color='blue')
plt.title('matched CDF')

plt.show()
