from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Lambda
import matplotlib.pyplot as plt

# Binarization function
binarize = Lambda(lambda x: (x > 0.5).float())

# Compose transforms
transform = Compose([
    ToTensor(),
    binarize  # Optional: comment out if you don't want binarization
])

# Load MNIST dataset
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
