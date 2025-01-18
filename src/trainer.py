import torch
import torch.nn as nn
import torch.optim
from .plotsResults import plotResults

def startTraining(model, train_loader, valid_loader, epochs=21):
    # Definisci la funzione di loss
    criterion = nn.CrossEntropyLoss()  # Usa CrossEntropyLoss per problemi di classificazione

    # Usa RProp come ottimizzatore con parametri personalizzati
    optimizer = torch.optim.Rprop(
        model.parameters(),
        lr=0.01,  # Learning rate
        etas=(0.5, 1.2),  # Fattori di incremento/decremento del learning rate
        step_sizes=(1e-06, 10)  # Limiti per la dimensione del passo
    )

    # List per memorizzare i risultati
    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []

    # Ciclo di addestramento
    for epoch in range(epochs):
        model.train()  # Imposta il modello in modalità di training
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Azzera i gradienti
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calcola la loss
            loss.backward()  # Backward pass
            optimizer.step()  # Aggiorna i pesi con RProp

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calcola la loss e l'accuracy per il training
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        # Validazione
        model.eval()  # Imposta il modello in modalità di valutazione
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0

        with torch.no_grad():  # Disabilita il calcolo dei gradienti per la validazione
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()

        # Calcola la loss e l'accuracy per la validazione
        valid_loss /= len(valid_loader)
        valid_acc = correct_valid / total_valid
        valid_loss_history.append(valid_loss)
        valid_acc_history.append(valid_acc)

        # Stampa i risultati per ogni epoca
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}")
        print("-" * 20)

    # Restituisci i risultati
    training_results = {
        "train_loss": train_loss_history,
        "train_accuracy": train_acc_history,
        "valid_loss": valid_loss_history,
        "valid_accuracy": valid_acc_history,
    }
    return training_results

def showTrainingResults(training_results, epochs):
    # Recupera i risultati
    train_loss = training_results["train_loss"]
    train_acc = training_results["train_accuracy"]
    valid_loss = training_results["valid_loss"]
    valid_acc = training_results["valid_accuracy"]

    # Visualizza i risultati
    plotResults(
        metrics=[train_loss, valid_loss],
        reps=epochs,
        ylabel="Loss",
        ylim=[1.5, 2.0],
        metric_name=["Training Loss", "Validation Loss"],
        color=["g", "b"],
    )

    plotResults(
        metrics=[train_acc, valid_acc],
        reps=epochs,
        ylabel="Accuracy",
        ylim=[0.7, 1.0],
        metric_name=["Training Accuracy", "Validation Accuracy"],
        color=["g", "b"],
    )