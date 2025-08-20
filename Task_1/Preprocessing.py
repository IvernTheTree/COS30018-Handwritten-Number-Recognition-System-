from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Lambda, Resize
import matplotlib.pyplot as plt


#Flow of the code:
# 1. Import necessary libraries.
# 2. Define a transformation to convert images to binary format.
# 3. Load the MNIST dataset with the defined transformation.
# 4. Visualize a sample image from the training dataset.



# Download the MNIST dataset, dont run again if already downloaded
# train_dataset = datasets.MNIST(
#     root='data',
#     train=True,
#     transform=ToTensor(),
#     download=True
# )

# test_dataset = datasets.MNIST(
#     root='data',
#     train=False,
#     transform=ToTensor(),
#     download=True
# )


#Image transformation

#resize = Resize((28, 28))  # height, width. if needed

binarize = Lambda(lambda x: (x > 0.5).float())

transform = Compose([
    ToTensor(),
    binarize  
])

# Load MNIST dataset, (takes raw data from dir 
#                     ,if true then take train data else take test data
#                     ,takes PIL image and return transfromed
#                     ,if true then download the data if not already downloaded)
train_dataset = datasets.MNIST(
    root='data',
    train=True,
    transform=transform,
    download=False         
)

test_dataset = datasets.MNIST(
    root='data',
    train=False,
    transform=transform,
    download=False
)

# Visualize a sample image from the training dataset
sample_img, sample_label = train_dataset[0]
plt.imshow(sample_img.squeeze(), cmap='gray')
plt.title(f'Number: {sample_label}')
plt.show()
