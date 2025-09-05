import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# folders
input_folder = "images"
output_folder = "grayscale"
os.makedirs(output_folder, exist_ok=True)

# lists
data = []
labels = []

def convert_to_grayscale(image_path):
    # to gray
    img = Image.open(image_path).convert("L")
    return np.array(img, dtype=np.float32)

# loop files
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(input_folder, filename)
        gray_img = convert_to_grayscale(filepath)

        # save gray img
        out_path = os.path.join(output_folder, filename)
        Image.fromarray((gray_img * 255).astype(np.uint8)).save(out_path)

        data.append(gray_img)
        # label from filename
        label = filename.split("_")[0]
        labels.append(label)

# to numpy
data = np.array(data)
labels = np.array(labels)

# save arrays
np.save("dataset_gray.npy", data)
np.save("labels.npy", labels)

print(f"Processed {len(data)} images, saved to {output_folder}/ and dataset_gray.npy")

# show one
plt.subplot(1,2,1)
plt.title("Example Grayscale")
plt.imshow(data[0], cmap="gray")
plt.axis("off")
plt.show()
