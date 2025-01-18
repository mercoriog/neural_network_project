import torch
from .plotsResults import plotResults

def predict(model, test_loader):
    """
    Effettua previsioni sul test set utilizzando il modello addestrato e restituisce l'andamento dell'accuratezza.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    accuracies = []  # Lista per memorizzare l'accuratezza per ogni batch

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

            # Calcola l'accuratezza per il batch corrente
            correct = sum([1 if pred == label else 0 for pred, label in zip(predicted.tolist(), labels.tolist())])
            batch_accuracy = correct / len(labels)
            accuracies.append(batch_accuracy)  # Aggiungi l'accuratezza del batch alla lista

    # Calcola l'accuratezza complessiva
    correct = sum([1 if pred == label else 0 for pred, label in zip(all_predictions, all_labels)])
    accuracy = correct / len(all_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Restituisci le previsioni, le etichette vere, l'accuratezza complessiva e l'andamento dell'accuratezza
    return all_predictions, all_labels, accuracies


def showPlotsPredict(predictions, labels, accuracies):
    # Visualizza l'andamento dell'accuratezza come grafico
    plotResults(
        metrics=[accuracies],  # Passa la lista di accuratezze
        reps=100,
        title="Test Accuracy over Batches",
        ylabel="Accuracy",
        ylim=[0.0, 1.0],
        metric_name=["Test Accuracy"],
        color=["b"]
    )