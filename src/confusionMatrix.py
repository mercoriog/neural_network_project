import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

def confusionMatrix(model, test_loader):
    """
    Calcola e visualizza la matrice di confusione per un modello su un dataset di test.
    
    Args:
        model (nn.Module): Il modello addestrato.
        test_loader (DataLoader): Il DataLoader contenente il dataset di test.
    """
    # Imposta il modello in modalità di valutazione
    model.eval()

    # List per memorizzare le etichette vere e le previsioni
    all_true_labels = []
    all_predicted_labels = []

    # Disabilita il calcolo dei gradienti per l'inferenza
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Effettua le previsioni
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Ottieni la classe predetta

            # Aggiungi le etichette vere e le previsioni alle liste
            all_true_labels.extend(labels.tolist())
            all_predicted_labels.extend(predicted.tolist())

    # Crea la matrice di confusione
    num_classes = len(torch.unique(torch.tensor(all_true_labels)))  # Numero di classi
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int32)

    for true_label, pred_label in zip(all_true_labels, all_predicted_labels):
        cm[true_label, pred_label] += 1

    # Converti la matrice di confusione in un array NumPy
    cm = cm.numpy()

    # Visualizza la matrice di confusione
    plt.figure(figsize=(15, 8))
    sn.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 14}, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.title("Confusion Matrix")
    plt.show()