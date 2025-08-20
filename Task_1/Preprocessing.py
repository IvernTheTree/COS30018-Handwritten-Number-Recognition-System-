from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Lambda, Resize
import matplotlib.pyplot as plt


#Flow of the code:
# 1. Import necessary libraries.
# 2. Define a transformation to convert images to binary format.
# 3. Load the MNIST dataset with the defined transformation.
# 4. Visualize a sample image from the training dataset.


class MNISTPreprocessor:
    def __init__(self, resize_shape=None, binarize_threshold=0.5):
        self.train_dataset = None
        self.test_dataset = None
        self.transform = None
        self.binarize_threshold = binarize_threshold
        self.resize_shape = resize_shape
        self.transform = self.create_transform()

    
    #Image transformation
    def create_transform(self):
        transform_list = []
        if self.resize_shape:
            transform_list.append(Resize(self.resize_shape))
        transform_list.append(ToTensor())
        transform_list.append(Lambda(lambda x: (x > self.binarize_threshold).float()))
        return Compose(transform_list)

    # Download the MNIST dataset, dont run again if already downloaded
    def load_datasets(self, download=False, data_dir='data'):
        self.train_dataset = datasets.MNIST(
            root='data',
            train=True,
            transform=self.transform,
            download=True
        )

        self.test_dataset = datasets.MNIST(
            root='data',
            train=False,
            transform=self.transform,
            download=True
        )

    # Visualize a sample image from the dataset
    # train = True for training dataset, False for test dataset
    def visualize_sample(self, train=True, index=0):
        dataset = self.train_dataset if train else self.test_dataset
        sample_img, sample_label = dataset[index]
        plt.imshow(sample_img.squeeze(), cmap='gray')
        plt.title(f'Label: {sample_label}')
        plt.show()
        


mnist_processor = MNISTPreprocessor(resize_shape=(28,28))
mnist_processor.load_datasets(download=False)
mnist_processor.visualize_sample(train=True, index=0)
mnist_processor.visualize_sample(train=False, index=0)
