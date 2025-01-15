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

    # Define model
    model = compiler.DynamicModel(hidden_num_layers=1, num_neurons=64, func_activation="relu", input_shape=X_train.shape[1])

    print(model)

    # Train model:
    training_results = trainer.startTraining(model, X_train, y_train, X_valid, y_valid)

    # Show training results:
    trainer.showTrainingResults(training_results)

    # Predict test set:
    predicter.predict(model, X_test, y_test)

    # Print confusion matrix:
    confMx.confusionMatrix()

    print('Done.')