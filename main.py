from src import predicter
from src import datasetLoader as dsLoad
from src import datasetPreprocessing as dsPrep
from src import trainer
from src import confusionMatrix as confMx
from src import modelCompiler as compiler

if __name__ == "__main__":
    # Load Datatset:
    X_train, y_train, X_test, y_test, X_valid, y_valid = dsLoad.getData()

    # Dataset Images: transformation and normalization
    X_train, X_test, X_valid = dsPrep.preprocessDataset(X_train, X_test, X_valid)

    # Dataset Labels: One-hot encoding
    y_train, y_test, y_valid = dsPrep.oneHotLabelEncoding(y_train, y_test, y_valid)

    # Esempio di utilizzo
    input_shape = 128  # Dimensione dell'input
    layer_sizes = [64, 32, 16, 8]  # Dimensioni dei layer
    activation_functions = [F.relu, F.relu, F.relu, torch.sigmoid]  # Funzioni di attivazione

    # Crea il modello
    model = compiler.DynamicModel(input_shape, layer_sizes, activation_functions)

    # Stampo il modello
    print(model)

    # Define model:
    model = compiler.model(1, 64, "relu", X_train.shape[1])

    # Compile model:
    model = compiler.modelCompile(model)

    # Train model:
    training_results = trainer.startTraining(model, X_train, y_train, X_valid, y_valid)

    # Show training results:
    trainer.showTrainingResults(training_results)

    # Predict test set:
    predicter.predict(model, X_test, y_test)

    # Print confusion matrix:
    confMx.confusionMatrix()

    print('Done.')