''' This python file segments the mnist dataset into segements so the image can be recognised'''
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np 
import random 

# Step 1 is to pick 3 random digits from MNIST dataset
def create_segments(mnist_dataset, num_digits=3):
    # Pick 3 random indicies (positions) in the mnist test dataset
    indicies = random.sample(range(len(mnist_dataset)), num_digits)
    digits = []
    labels = []

    # For each 'pick' of the 3 indicies we get the image and the label and add it to the respective lists
    for idx in indicies:
        img, label = mnist_dataset[idx]
        # Gets rid of colour channels in img and converts to numpy array
        img_np = img.squeeze().numpy()
        digits.append(img_np)
        labels.append(label)
    
    # Show the 3 random digits selected 'actual number shown in image'
    print(f'Selected digits {labels}')

    # Create a canvas the same height as images but wider to fit all digits 
    canvas_height = 28
    canvas_width = 90
    spacing = 5

    canvas = np.zeros((canvas_height, canvas_width))
    x_position = 10 

    for i, digit in enumerate(digits):
        # Lets find the number of each column that contains a value > 0.1 within it for image content
        digit_columns = np.where(digit.max(axis=0) > 0.1)[0] 


        if len(digit_columns) > 0:
            # Extract ONLY the digit content 
            digit_start = digit_columns[0]
            digit_end = digit_columns[-1]
            digit_content = digit[:, digit_start:digit_end+1]
            digit_width = digit_end - digit_start + 1

            canvas[:, x_position:x_position+digit_width] = digit_content
            x_position += digit_width + spacing 
    
    return canvas, labels 

def main():
    print("Loading the MNIST dataset.....")
    '''Input a greyscale image with pixel values from 0-255, we want to change the value between 0 and 1 as that 
    classifies as a "greyscale" image'''
    transform = torchvision.transforms.ToTensor()
    test_dataset = torchvision.datasets.MNIST(
        root='../data',
        train=False,
        transform=transform,
        download=False
    )

    multi_digit_image, true_labels = create_segments(test_dataset, num_digits=3)

    #Visualise
    plt.figure(figsize=(10, 4))
    plt.imshow(multi_digit_image, cmap='grey')
    plt.title(f'Multi-digit image: {true_labels}')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main() 


