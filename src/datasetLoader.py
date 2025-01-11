import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

def loadDataset():
    # Load train and test dataset
    return tfds.load('mnist', split=['train', 'test'], shuffle_files=True)

def extractValidationSet(train_set):
    # Extract first 10000 images from train
    valid_set = train_set[:10000]

    # Remove valid set from train set
    train_set = train_set[10000:]

    return train_set, valid_set

def splitData(dataset):
    # Split images from labels 
    images = []
    labels = []

    for data in dataset:
        # Extract images
        images.append(data['image'])
        # Extract labels
        labels.append(data['label'])
    
    # Create numpy array:
    X = np.array(images)
    Y = np.array(labels)

    return X, Y

def getData():
    # Load train and test dataset
    train_set, test_set = loadDataset()

    # Extract validationn set from train set
    train_set, valid_set = extractValidationSet(train_set)

    # Split X, Y from train
    X_train, Y_train = splitData(train_set)

    # Split X, Y from validation
    X_valid, Y_valid = splitData(valid_set)

    # Split X, Y from test
    X_test, Y_test = splitData(test_set)

    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid



        