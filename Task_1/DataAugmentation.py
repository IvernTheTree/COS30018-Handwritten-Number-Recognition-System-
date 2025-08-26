import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

"""
transform_train = transforms.Compose([
    transforms.Resize((28, 28)),          # Ensure images are 28x28
    transforms.Grayscale(num_output_channels=1),  # Force grayscale
    transforms.RandomRotation(15),        # Rotate randomly up to ±15 degrees
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random shift (10%)
    transforms.RandomHorizontalFlip(),    # Random horizontal flip
    transforms.ToTensor(),                # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean=0.5, std=0.5
])

# For test/validation set we usually don’t augment
transform_test = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
"""
# Updated transformations with interpolation method specified
transform_train = transforms.Compose([
    transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomAffine(0, translate=(0.05, 0.05), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# For test/validation set we usually don’t augment
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST Dataset with augmentation for training set
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform_train,
    download=True
)

# Load MNIST Test Dataset without augmentation 
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform_test,
    download=True
)

# DataLoader for batching and shuffling
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Visualize some samples
def imshow(img):
    img = img / 2 + 0.5     # Unnormalize
    np_img = img.numpy() # Convert to numpy
    plt.imshow(np_img[0], cmap='gray') # Show the first channel
    plt.axis('off') # Turn off axis
    plt.show()

# Get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Show 8 examples/numbers
fig, axes = plt.subplots(1, 8, figsize=(15, 2))
for idx in range(8):
    img = images[idx] / 2 + 0.5  # unnormalize
    np_img = img.numpy()
    axes[idx].imshow(np_img[0], cmap='gray')
    axes[idx].set_title(f"{labels[idx].item()}")
    axes[idx].axis('off')
plt.show()