import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split

def loadDataset():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converte in tensore e normalizza in [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))  # Normalizza con media e deviazione standard di MNIST
    ])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_set, test_set

def extractValidationSet(train_set, valid_size=10000):
    train_size = len(train_set) - valid_size
    train_set, valid_set = random_split(train_set, [train_size, valid_size])
    return train_set, valid_set

def splitData(dataset):
    images = []
    labels = []

    for image, label in dataset:
        images.append(image)
        labels.append(label)
    
    # Stack tensors along the first dimension
    X = torch.stack(images)
    Y = torch.tensor(labels, dtype=torch.long)

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

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, Y_train)
    valid_dataset = TensorDataset(X_valid, Y_valid)
    test_dataset = TensorDataset(X_test, Y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, valid_loader, test_loader