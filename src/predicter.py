import torch

def predict(model, test_loader):
    """
    Effettua previsioni sul test set utilizzando il modello addestrato.

    Args:
        model (nn.Module): Il modello addestrato.
        test_loader (DataLoader): Il DataLoader contenente il dataset di test.

    Returns:
        predictions (list): Lista delle previsioni (classi predette).
        labels (list): Lista delle etichette vere (ground truth).
        accuracy (float): Accuratezza complessiva sul test set.
    """
    # Imposta il modello in modalità di valutazione
    model.eval()

    # List per memorizzare le previsioni e le etichette vere
    all_predictions = []
    all_labels = []

    # Disabilita il calcolo dei gradienti per l'inferenza
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Effettua le previsioni
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Ottieni la classe predetta

            # Aggiungi le previsioni e le etichette vere alle liste
            all_predictions.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    # Calcola l'accuratezza complessiva
    correct = sum([1 if pred == label else 0 for pred, label in zip(all_predictions, all_labels)])
    accuracy = correct / len(all_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Restituisci le previsioni, le etichette vere e l'accuratezza
    return all_predictions, all_labels, accuracy