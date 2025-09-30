import numpy as np
from scipy import spatial
from Task_1.Preprocessing import MNISTPreprocessor


# Run from the root directory:
# python -m Task3.K-NN

#The algorithim works by comparing the smilarity of a test image to all training images
#and finding the most similar one (or k most similar ones) and assigning the label of
#the most similar training image (or majority label among the k most similar ones)
#to the test image.
def faster_KNN(X_test, X_train, y_train, k=1):
    predictions_list = []
    tree = spatial.cKDTree(X_train)
    for i in range(len(X_test)):
        dists, idxs = tree.query(X_test[i], k=k)
        if k == 1:
            prediction = y_train[idxs]
        else:
            neighbor_labels = y_train[idxs]
            prediction = np.bincount(neighbor_labels).argmax()
        predictions_list.append(prediction)
    return predictions_list

def preprocess_digit_for_knn(digit_img):
    digit_flat = digit_img.reshape(-1)
    return digit_flat

if __name__ == "__main__":
    mnist_processor = MNISTPreprocessor(resize_shape=(28,28), binarize_threshold=0.5)
    mnist_processor.load_datasets(download=False)

    #Reduce training set size for speed
    N = 5000 
    # Prepare training data
    X_train = np.array([mnist_processor.train_dataset[i][0].squeeze().numpy().reshape(-1) for i in range(len(mnist_processor.train_dataset))])
    y_train = np.array([mnist_processor.train_dataset[i][1] for i in range(len(mnist_processor.train_dataset))])

    # Example: Use first 5 test digits as X_test
    X_test = np.array([mnist_processor.test_dataset[i][0].squeeze().numpy().reshape(-1) for i in range(5)])
    true_labels = [mnist_processor.test_dataset[i][1] for i in range(5)]

    # Predict using K-NN
    predicted_labels = faster_KNN(X_test, X_train, y_train, k=1)

    print("True labels:     ", true_labels)
    print("Predicted labels:", predicted_labels)
    #Check commit