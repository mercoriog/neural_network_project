import torch
import numpy as np

NUM_CLASSES = 10
IMG_SET_DIM = 28
FLOAT32 = torch.float32  # Tipo di dato PyTorch
PIXEL_RANGE = 255

# Preprocesses the training, test, and validation datasets:
# - Flattens images into 1D vectors of size IMG_SET_DIM * IMG_SET_DIM.
# - Converts pixel values to FLOAT32.
# - Normalizes pixel values to the range [0, 1] by dividing by PIXEL_RANGE.
# In case of an error (e.g., inconsistent dimensions), returns None.
def preprocessDataset(X_train, X_test, X_valid):
    try:
        # Riformatta i dati e converte il tipo
        X_train = X_train.reshape((X_train.shape[0], IMG_SET_DIM * IMG_SET_DIM))
        X_train = X_train.astype(np.float32) / PIXEL_RANGE  # Usa .astype() per la conversione

        X_test = X_test.reshape((X_test.shape[0], IMG_SET_DIM * IMG_SET_DIM))
        X_test = X_test.astype(np.float32) / PIXEL_RANGE

        X_valid = X_valid.reshape((X_valid.shape[0], IMG_SET_DIM * IMG_SET_DIM))
        X_valid = X_valid.astype(np.float32) / PIXEL_RANGE

        # Converti i dati in tensori PyTorch
        X_train = torch.tensor(X_train, dtype=FLOAT32)
        X_test = torch.tensor(X_test, dtype=FLOAT32)
        X_valid = torch.tensor(X_valid, dtype=FLOAT32)

        return X_train, X_test, X_valid

    except ValueError as e:
        print(f"Error during preprocessing: {e}")
        return None

# Performs one-hot encoding for the label datasets of training, test, and validation:
# - Converts each label into a binary vector of length NUM_CLASSES.
# - For example, a label 2 with NUM_CLASSES = 4 becomes [0, 0, 1, 0].
# - Used to make the labels compatible with classification models.
# In case of an error, returns None.
def oneHotLabelEncoding(Y_train, Y_test, Y_valid):
    try:
        # Converti le etichette in tensori PyTorch (se non lo sono già)
        Y_train = torch.tensor(Y_train, dtype=torch.long)
        Y_test = torch.tensor(Y_test, dtype=torch.long)
        Y_valid = torch.tensor(Y_valid, dtype=torch.long)

        # Applica one-hot encoding
        Y_train = torch.nn.functional.one_hot(Y_train, num_classes=NUM_CLASSES).type(FLOAT32)
        Y_test = torch.nn.functional.one_hot(Y_test, num_classes=NUM_CLASSES).type(FLOAT32)
        Y_valid = torch.nn.functional.one_hot(Y_valid, num_classes=NUM_CLASSES).type(FLOAT32)

        return Y_train, Y_test, Y_valid
    except Exception as e:
        print(f"Error during one-hot encoding: {e}")
        return None