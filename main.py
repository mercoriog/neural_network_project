from src import predicter
from src import datasetLoader as dsLoad
from src import trainer
from src import confusionMatrix as confMx
from src import modelCompiler as compiler
from src import plotsResults

if __name__ == "__main__":
    # Carica i DataLoader
    train_loader, valid_loader, test_loader = dsLoad.getData()

    # Definisci il modello
    model = compiler.DynamicModel(hidden_num_layers=1, num_neurons=64, func_activation="relu", input_shape=784)

    print(model)

    # Addestra il modello
    training_results = trainer.startTraining(model, train_loader, valid_loader, epochs=21)

    # Mostra i risultati dell'addestramento
    trainer.showTrainingResults(training_results)

    # Effettua le previsioni sul test set
    predictions, labels, accuracy = predicter.predict(model, test_loader)

    # Visualizza l'accuratezza come grafico
    plotsResults(
        [accuracy],
        title="Test Accuracy",
        ylabel="Accuracy",
        ylim=[0.0, 1.0],
        metric_name=["Test Accuracy"],
        color=["b"]
    )

    # Visualizza la matrice di confusione
    confMx.confusionMatrix(model, test_loader)

    print('Done.')