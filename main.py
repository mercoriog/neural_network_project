from src import predicter
from src import datasetLoader as dsLoad
from src import trainer
from src import confusionMatrix as confMx
from src import modelCompiler as compiler


if __name__ == "__main__":
    # Carica i DataLoader
    train_loader, valid_loader, test_loader = dsLoad.getData()

    # Definisci il modello
    model = compiler.DynamicModel(hidden_num_layers=1, num_neurons=128, func_activation="relu", input_shape=784)

    print(model)

    # Imposta epoche
    epochs=40

    # Addestra il modello
    training_results = trainer.startTraining(model, train_loader, valid_loader, epochs)

    # Mostra i risultati dell'addestramento
    trainer.showTrainingResults(training_results, epochs)

    # Effettua le previsioni sul test set
    predictions, labels, accuracies = predicter.predict(model, test_loader)

    # Visualizza la matrice di confusione
    metrics = confMx.confusionMatrix(model, test_loader)

    print(metrics)

    print('Done.')