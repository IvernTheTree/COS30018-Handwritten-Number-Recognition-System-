import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torchvision import datasets, transforms

def segment_digits_kmeans(image, n_clusters=2):

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
 