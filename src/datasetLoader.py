import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split

# Funzione per caricare il dataset MNIST
def loadDataset():
    # Definisci una pipeline di trasformazioni da applicare alle immagini
    transform = transforms.Compose([
        # Converte le immagini in tensori PyTorch e le normalizza nell'intervallo [0, 1]
        transforms.ToTensor(),
        # Normalizza i valori dei pixel utilizzando la media (0.1307) e la deviazione standard (0.3081) di MNIST
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Carica il training set di MNIST con le trasformazioni definite
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Carica il test set di MNIST con le trasformazioni definite
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Restituisce il training set e il test set
    return train_set, test_set

# Funzione per estrarre un validation set dal training set
def extractValidationSet(train_set, valid_size=10000):
    # Calcola la dimensione del training set dopo aver rimosso il validation set
    train_size = len(train_set) - valid_size
    # Divide il training set in due parti: training set ridotto e validation set
    train_set, valid_set = random_split(train_set, [train_size, valid_size])
    # Restituisce il training set ridotto e il validation set
    return train_set, valid_set

# Funzione per separare immagini e etichette da un dataset
def splitData(dataset):
    # Lista per memorizzare le immagini
    images = []
    # Lista per memorizzare le etichette
    labels = []

    # Itera su ogni elemento del dataset (coppia immagine-etichetta)
    for image, label in dataset:
        # Aggiungi l'immagine alla lista delle immagini
        images.append(image)
        # Aggiungi l'etichetta alla lista delle etichette
        labels.append(label)
    
    # Concatena le immagini in un unico tensore lungo la prima dimensione (batch)
    X = torch.stack(images)
    # Converte le etichette in un tensore PyTorch di tipo long (intero)
    Y = torch.tensor(labels, dtype=torch.long)

    # Restituisce le immagini (X) e le etichette (Y) come tensori
    return X, Y

def getData():
    # Carica i dataset di training e test
    train_set, test_set = loadDataset()

    # Estrai un set di validazione dal dataset di training
    train_set, valid_set = extractValidationSet(train_set)

    # Suddivide il set di training in immagini (X) e etichette (Y)
    X_train, Y_train = splitData(train_set)

    # Suddivide il set di validazione in immagini (X) e etichette (Y)
    X_valid, Y_valid = splitData(valid_set)

    # Suddivide il set di test in immagini (X) e etichette (Y)
    X_test, Y_test = splitData(test_set)

    # Crea i dataset TensorDataset per training, validazione e test
    train_dataset = TensorDataset(X_train, Y_train)
    valid_dataset = TensorDataset(X_valid, Y_valid)
    test_dataset = TensorDataset(X_test, Y_test)

    # Crea i DataLoader con un solo batch (batch_size = numero totale di campioni)
    # Il set di training ha shuffle=True per mescolare i dati ad ogni epoca
    # Il set di validazione e test non vengono mescolati (shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # Restituisce i DataLoader per training, validazione e test
    return train_loader, valid_loader, test_loader
