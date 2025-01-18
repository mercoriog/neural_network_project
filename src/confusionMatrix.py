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

    metrics = calculate_metrics(cm)

    # Visualizza la matrice di confusione
    plt.figure(figsize=(15, 8))
    sn.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 14}, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.title("Confusion Matrix")
    plt.show()



def calculate_metrics(cm):
    """
    Calcola le metriche di classificazione da una matrice di confusione.
    
    Args:
        cm (numpy.ndarray): La matrice di confusione (2D array).
        
    Returns:
        dict: Un dizionario contenente tutte le metriche.
    """
    # Calcola i valori TP, TN, FP, FN per ciascuna classe
    num_classes = cm.shape[0]
    metrics = {}
    
    # Inizializza variabili globali per metrica aggregata
    total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0
    
    for i in range(num_classes):
        TP = cm[i, i]  # Veri positivi
        FP = cm[:, i].sum() - TP  # Falsi positivi
        FN = cm[i, :].sum() - TP  # Falsi negativi
        TN = cm.sum() - (TP + FP + FN)  # Veri negativi

        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_TN += TN

        # Precision, Recall e F1 per la classe corrente
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f"Class {i}"] = {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "True Positives": TP,
            "False Positives": FP,
            "False Negatives": FN,
            "True Negatives": TN,
        }

    # Global metrics
    accuracy = total_TP / cm.sum() if cm.sum() > 0 else 0
    macro_precision = np.mean([metrics[f"Class {i}"]["Precision"] for i in range(num_classes)])
    macro_recall = np.mean([metrics[f"Class {i}"]["Recall"] for i in range(num_classes)])
    macro_f1 = np.mean([metrics[f"Class {i}"]["F1-Score"] for i in range(num_classes)])
    mcc = (
        (total_TP * total_TN - total_FP * total_FN) /
        np.sqrt((total_TP + total_FP) * (total_TP + total_FN) * (total_TN + total_FP) * (total_TN + total_FN))
        if (total_TP + total_FP) > 0 and (total_TP + total_FN) > 0 and (total_TN + total_FP) > 0 and (total_TN + total_FN) > 0
        else 0
    )

    metrics["Global"] = {
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1-Score": macro_f1,
        "Matthews Correlation Coefficient (MCC)": mcc,
    }

    print("TRUE POSITIVE: " + str(total_TP))
    print("FALSE POSITIVE: " + str(total_FP))
    print("FALSE NEGATIVE: " + str(total_FN))
    print("TRUE NEGATIVE: " + str(total_TN))
    print("PRECISION: " + str(precision))
    print("RECALL: " + str(recall))
    print("F1: " + str(f1))
    print("ACCURACY: " + str(accuracy))
    print("MACRO PRECISION: " + str(macro_precision))
    print("MACRO RECALL: " + str(macro_recall))
    print("MACRO F1: " + str(macro_f1))
    print("MCC: " + str(mcc))


    return metrics