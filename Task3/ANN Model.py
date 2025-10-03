import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values (0-255 → 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten 28x28 images → 784 vector
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Build ANN Model
model = models.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),   # hidden layer 1
    layers.Dense(64, activation="relu"),                        # hidden layer 2
    layers.Dense(10, activation="softmax")                      # output layer (10 classes: 0-9)
])

# Compile model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train the Model
history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test))

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n Test Accuracy: {test_acc*100:.2f}%")

# Plot Training Curves
plt.figure(figsize=(12,5))

# Plot accuracy
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Accuracy") # Training accuracy labels
plt.plot(history.history["val_accuracy"], label="Validation Accuracy") # Validation accuracy labels
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Plot loss
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss") # Training loss labels
plt.plot(history.history["val_loss"], label="Validation Loss") # Validation loss labels
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

# Predict on test images
predictions = model.predict(x_test)

# Pick 10 random test samples to visualize
indices = np.random.choice(len(x_test), 10)
x_sample = x_test[indices] # Sample images for display 
y_true = y_test[indices] # True labels for sample images
y_pred = np.argmax(predictions[indices], axis=1)# Predicted labels for sample images

# Reshape images back to 28x28 for display
x_sample_images = x_sample.reshape(-1, 28, 28)

# Show results
plt.figure(figsize=(12,4))
for i in range(10):
    plt.subplot(2,5,i+1) # 2 rows, 5 columns for 10 images
    plt.imshow(x_sample_images[i], cmap="gray")
    plt.title(f"Pred: {y_pred[i]}, True: {y_true[i]}") # Display predicted and true labels
    plt.axis("off") # Hide axes for clarity

plt.suptitle("Handwritten Number Recognition - ANN Predictions", fontsize=14)
plt.show()