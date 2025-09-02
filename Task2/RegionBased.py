#import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

# Create synthetic multi-digit image
def create_multi_digit_image(digits, spacing=5):
    imgs = []
    for d in digits:
        img, _ = mnist_dataset[d]
        img = img.squeeze().numpy() * 255  # to numpy, scale [0,255]
        imgs.append(img)
    combined = np.concatenate(
        [np.pad(img, ((0, 0), (0 if i == 0 else spacing, 0)), 'constant') 
         for i, img in enumerate(imgs)], axis=1
    )
    return combined.astype(np.uint8)

# Multi-digit image 
multi_digit_img = create_multi_digit_image([0, 10, 20, 30])

# Thresholding (prepare for region segmentation)
_, binary = cv2.threshold(multi_digit_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
binary = 255 - binary   # invert (digits=white, background=black)

# Region-based segmentation (Connected Components)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Extract digit regions
digit_images = []
segmented_output = cv2.cvtColor(multi_digit_img, cv2.COLOR_GRAY2BGR)

# For loop to extract each digit
for i in range(1, num_labels):  # skip background (label 0)
    x, y, w, h, area = stats[i]
    # If statement to filter out small regions
    if w > 5 and h > 5:  # filter out noise
        digit = binary[y:y+h, x:x+w]
        digit_resized = cv2.resize(digit, (28, 28))
        digit_images.append(digit_resized)
        cv2.rectangle(segmented_output, (x, y), (x+w, y+h), (0, 255, 0), 2)


# Visualise Results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1) # Original Multi-digit
plt.title("Original Multi-digit")
plt.imshow(multi_digit_img, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2) # Binary Image
plt.title("Binary Image")
plt.imshow(binary, cmap="gray") # Make binary image gray
plt.axis("off")

plt.subplot(1, 3, 3) # Region-based Segmentation
plt.title("Region-based Segmentation")
plt.imshow(segmented_output)
plt.axis("off")

plt.show()

# Show segmented digits
if digit_images:
    plt.figure(figsize=(12, 2))
    for i, d in enumerate(digit_images):
        plt.subplot(1, len(digit_images), i + 1)
        plt.imshow(d, cmap="gray")
        plt.axis("off")
    plt.suptitle("Extracted Digits (28x28 each)")
    plt.show()
