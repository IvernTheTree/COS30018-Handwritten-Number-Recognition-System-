import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from Task_1.Preprocessing import MNISTPreprocessor
from scipy.ndimage import label  # Added missing import


# Run from the root directory:
# python -m Task2.Clustering

# Function to estimate number of digits using connected components
def estimate_digit_count(image):
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, ncomponents = label(image > 0, structure)
    return ncomponents

# Function to segment digits in an image using K-Means clustering
# Takes in a binary image and number of clusters (default 2)
# Separates the digits into clusters using K-Means
def segment_digits_kmeans(image, n_clusters=None):
    if n_clusters is None:
        n_clusters = estimate_digit_count(image)
        print(f"Estimated digit count: {n_clusters}")

    coords = np.column_stack(np.where(image > 0))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
    labels = kmeans.labels_

    plt.imshow(image, cmap='gray')
    for i in range(n_clusters):
        cluster_coords = coords[labels == i]
        plt.scatter(cluster_coords[:, 1], cluster_coords[:, 0], s=1, label=f'Cluster {i}')
    plt.legend()
    plt.title('K-Means Clustering of Digit Pixels')

    masks = []
    for i in range(n_clusters):
        mask = np.zeros_like(image)
        mask[coords[labels == i][:, 0], coords[labels == i][:, 1]] = 1
        masks.append(mask)
    return masks


mnist_processor = MNISTPreprocessor(resize_shape=(28, 28), binarize_threshold=0.5)
mnist_processor.load_datasets(download=False)

# Get two sample images from the training dataset
img, label = mnist_processor.train_dataset[0]
img = img.squeeze().numpy()
img2, label2 = mnist_processor.train_dataset[1]
img2 = img2.squeeze().numpy()

# Combine the two images side by side
multi_digit_image = np.concatenate((img, img2), axis=1)

masks = segment_digits_kmeans(multi_digit_image, n_clusters=2)
for i, mask in enumerate(masks):
    plt.imshow(mask, cmap='gray')
    plt.title(f'Segmented Digit {i+1}')
    plt.show()