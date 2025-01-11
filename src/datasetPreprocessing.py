import numpy as np
import tensorflow_datasets as tfds

NUM_CLASSESS = 10
IMG_SET_DIM = 28
FLOAT32 = "float32"
PIXEL_RANGE = 255


# Preprocesses the training, test, and validation datasets:
# - Flattens images into 1D vectors of size IMG_SET_DIM * IMG_SET_DIM.
# - Converts pixel values to FLOAT32.
# - Normalizes pixel values to the range [0, 1] by dividing by PIXEL_RANGE.
# In case of an error (e.g., inconsistent dimensions), returns None.
def preprocessDataset(X_train, X_test, X_valid):

    try:
        X_train = X_train.reshape((X_train.shape[0], IMG_SET_DIM * IMG_SET_DIM))
        X_train = X_train.astype(FLOAT32) / PIXEL_RANGE

        X_test = X_test.reshape((X_test.shape[0], IMG_SET_DIM * IMG_SET_DIM))
        X_test = X_test.astype(FLOAT32) / PIXEL_RANGE

        X_valid = X_valid.reshape((X_valid.shape[0], IMG_SET_DIM * IMG_SET_DIM))
        X_valid = X_valid.astype(FLOAT32) / PIXEL_RANGE

        return X_train, X_test, X_valid

    except ValueError as e:
        print(f"Error during preprocessing: {e}")
        return None


# Performs one-hot encoding for the label datasets of training, test, and validation:
# - Converts each label into a binary vector of length NUM_CLASSESS.
# - For example, a label 2 with NUM_CLASSESS = 4 becomes [0, 0, 1, 0].
# - Used to make the labels compatible with classification models.
# In case of an error, returns None.
def oneHotLabelEncoding(Y_train, Y_test, Y_valid):
     try:
        Y_train = to_categorical(Y_train, NUM_CLASSESS)
        Y_test = to_categorical(Y_test, NUM_CLASSESS)
        Y_valid = to_categorical(Y_valid, NUM_CLASSESS)
        return Y_train, Y_test, Y_valid
    except Exception as e:
        print(f"Error during one-hot encoding: {e}")
        return None