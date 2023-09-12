import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to quantize the image by reducing the bit depth
def quantize_image(image, bits):
    # Calculate the quantization step
    Qlevel = 256 / (2 ** bits)
    
    # Quantize the image by mapping pixel values to the quantized levels
    quantized_image = np.floor(image / Qlevel) * Qlevel
    
    return quantized_image.astype(np.uint8)

# Read the input image
image = cv2.imread('origami.jpg', cv2.IMREAD_GRAYSCALE)


bits = 4
quantized_image = quantize_image(image, bits)

# Display the original and quantized images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(quantized_image, cmap='gray')
plt.title(f'Quantized Image ({bits} bits)')
plt.axis('off')

plt.show()

