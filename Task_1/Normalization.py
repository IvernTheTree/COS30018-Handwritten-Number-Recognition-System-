import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# I just want 0..1 values
def normalize_0_1(imgs):
    imgs = imgs.astype(np.float32)       # make sure division works as floats 
    imgs = imgs / 255.0                  # scale to 0..1
    return imgs

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("train shape:", x_train.shape)
print("test shape :", x_test.shape)
print("original range (train):", x_train.min(), "to", x_train.max())

# normalize
x_train_norm = normalize_0_1(x_train)
x_test_norm = normalize_0_1(x_test)

# check new range (just to be sure)
print("after normalization (train):", x_train_norm.min(), "to", x_train_norm.max())

# save files (hardcoded names)
np.save("x_train_norm.npy", x_train_norm)
np.save("x_test_norm.npy", x_test_norm)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
print("saved .npy files!")

# show one example before/after
idx = 3  # you can change this to see other images

plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(x_train[idx], cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Normalized (0..1)")
plt.imshow(x_train_norm[idx], cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

