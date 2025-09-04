import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from Task_1.Preprocessing import MNISTPreprocessor

# Function to segment digits in an image using K-Means clustering
#Takes in a binary image and number of clusters (default 2)
#and seperates the digits into clusters using K-Means
def segment_digits_kmeans(image, n_clusters=2):

    #Finds digit pixel coordinates, applies K-Means
    coords = np.column_stack(np.where(image > 0))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
    labels = kmeans.labels_

    plt.imshow(image, cmap='gray')
    for i in range(n_clusters):
        cluster_coords = coords[labels == i]
        plt.scatter(cluster_coords[:, 1], cluster_coords[:, 0], s=1, label=f'Cluster {i}')
    plt.legend()
    plt.title('K-Means Clustering of Digit Pixels')

    #placeholder for number so like each cluster is a digit 
    masks = []
    for i in range(n_clusters):
        mask = np.zeros_like(image)
        mask[coords[labels == i][:, 0], coords[labels == i][:, 1]] = 1
        masks.append(mask)
    return masks


mnist_processor = MNISTPreprocessor(resize_shape=(28,28), binarize_threshold=0.5)
mnist_processor.load_datasets(download=False)

#Get a sample image from the training dataset
img, label = mnist_processor.train_dataset[0]
img = img.squeeze().numpy()  
img2, label2 = mnist_processor.train_dataset[1]
img2 = img2.squeeze().numpy()

#combine the two images
multi_digit_image = np.concatenate((img, img2), axis=1)