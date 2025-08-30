import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np 
import random 

# Pick 3 random digits from MNIST dataset
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
    print(f'\nSelected digits {labels}')

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


def threshold_segmentation(image, threshold=0.1):
    print("Beginning threshold segmentation processing")
    print(f"Original image has pixels ranged from {image.min():.3f} to {image.max():.3f}!")
    print(f"If the image cantains a pixel with value under {threshold} it will be WHITE else BLACK")
    # Convert to binary (black and white)
    binary_image = (image > threshold).astype(np.uint8)

    # Sum of all units in each column 
    print("\nCreating vertical projection (sum of all pixels in each column)")
    vertical_projection = np.sum(binary_image, axis=0)
    print(f"Projection shape: {vertical_projection.shape}")
    print("\nFinding digit boundaries")
    segments = []
    in_digit = False
    start_pos = 0 

    # Enumerate turns into (digit, value)
    for i, proj_val in enumerate(vertical_projection):
        # Find the start of the digit 
        if proj_val > 0 and not in_digit:
            start_pos = i 
            in_digit = True
            print(f"Digit starts at column {i}")
        # FInd the end of the digit
        elif proj_val == 0 and in_digit:
            segments.append((start_pos, i))
            in_digit = False 
            print(f"Digit ends at column {i} width: {i - start_pos} pixels")
        
    # If the digit is in the LAST column of the canvas 
    if in_digit:
        segments.append((start_pos, len(vertical_projection)))
        
    # Extracting individual digits
    segmented_digits = []
    for i, (start, end) in enumerate(segments):
        width = end - start 
        # If less then 3 most likely not part of a digit!
        if width > 3:
            digit_segment = image[:, start:end]
            segmented_digits.append(digit_segment)
            print(f"Digit {i + 1} is in columns {start}-{end} and is {width} pixels wide")
        else:
            print(f"Skipping segment {start}-{end} because {width} pixels wide is too narrow")

    print(f"\nFinal Result: {len(segmented_digits)} digits extracted!")

    return segmented_digits, segments, vertical_projection, binary_image


def visualize_threshold_process(multi_digit_image, binary_image, vertical_projection, 
                                 segmented_digits, true_labels):
    # Visualise the 3 digits selected, what the binary image looks like and the vertical projection
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # The orginal image with three digits
    axes[0, 0].imshow(multi_digit_image, cmap='gray')
    axes[0, 0].set_title(f'1. Original Image\nSelected Digits: {true_labels}')
    axes[0, 0].axis('off')
    
    # The image after being coverted to binary values
    axes[0, 1].imshow(binary_image, cmap='gray')
    axes[0, 1].set_title('2. Binary Image\n(After thresholding)')
    axes[0, 1].axis('off')
    
    # The vertical projection
    axes[0, 2].plot(vertical_projection)
    axes[0, 2].set_title('3. Vertical Projection')
    axes[0, 2].set_xlabel('Column (x-position)')
    axes[0, 2].set_ylabel('Sum of pixels')
    axes[0, 2].grid(True)
    
    # Row 2 will be the segmented digits 
    for i, digit in enumerate(segmented_digits[:3]):
        axes[1, i].imshow(digit, cmap='gray')
        axes[1, i].set_title(f'Digit {i+1}')
        axes[1, i].axis('off')
    
    for i in range(len(segmented_digits[:3]), 3):
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

# Main function
def main():
    print("Loading the MNIST dataset.....")
    '''Input a greyscale image with pixel values from 0-255, we want to change the value between 0 and 1 as that 
    classifies as a "greyscale" image'''
    transform = torchvision.transforms.ToTensor()
    test_dataset = torchvision.datasets.MNIST(
        root='../data',
        train=False,
        transform=transform,
        download=True
    )

    multi_digit_image, true_labels = create_segments(test_dataset, num_digits=3)
    segmented_digits, segments, vertical_projection, binary_image = threshold_segmentation(multi_digit_image, threshold=0.1)

    # Show the results
    print("\nRESULTS SUMMARY")
    print(f"\nThe three digits: {true_labels}")
    print(f"Segments found: {len(segments)}")
    print(f"Digits extracted: {len(segmented_digits)}")
    
    # Visualize the plots 
    visualize_threshold_process(multi_digit_image, binary_image, vertical_projection, 
                                 segmented_digits, true_labels)

if __name__ == "__main__":
    main() 


