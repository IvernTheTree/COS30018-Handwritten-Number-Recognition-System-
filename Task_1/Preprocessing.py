from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Lambda
import matplotlib.pyplot as plt


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


# Binarization function
binarize = Lambda(lambda x: (x > 0.5).float())

# Binary transformation
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
    download=False         # Already downloaded
)

test_dataset = datasets.MNIST(
    root='data',
    train=False,
    transform=transform,
    download=False
)

sample_img, sample_label = train_dataset[0]
plt.imshow(sample_img.squeeze(), cmap='gray')
plt.title(f'Label: {sample_label}')
plt.show()
