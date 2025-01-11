import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

def loadDataset():
    # Load the MNIST dataset (split into train and test sets)
    return tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True)

def extractValidationSet(train_set, valid_size=10000):
    # Extract the first `valid_size` samples for the validation set
    valid_set = train_set.take(valid_size)
    # Keep the remaining samples for the training set
    train_set = train_set.skip(valid_size)
    return train_set, valid_set

def splitData(dataset):
    # Split images and labels into separate arrays
    images = []
    labels = []

    for image, label in dataset:
        # Append image data
        images.append(image.numpy())
        # Append label data
        labels.append(label.numpy())
    
    # Convert lists to numpy arrays
    X = np.array(images)
    Y = np.array(labels)

    return X, Y

def getData():
    # Load the train and test datasets
    train_set, test_set = loadDataset()

    # Extract validation set from the train set
    train_set, valid_set = extractValidationSet(train_set)

    # Split images (X) and labels (Y) for the training set
    X_train, Y_train = splitData(train_set)

    # Split images (X) and labels (Y) for the validation set
    X_valid, Y_valid = splitData(valid_set)

    # Split images (X) and labels (Y) for the test set
    X_test, Y_test = splitData(test_set)

    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid
