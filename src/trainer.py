from plotsResults import plotResults

def startTraining(model, X_train, y_train, X_valid, y_valid):
    # To train the model, we call the fit() method in Keras.
    # Parameters:
    #   - X_train: Training dataset (input features)
    #   - y_train: Labels for the training dataset (target values)
    #   - epochs: Number of times the model will iterate over the entire training dataset
    #   - batch_size: Number of samples processed before the model updates its weights
    #   - validation_data: Tuple (X_valid, y_valid) used to evaluate the model's performance during training
    training_results = model.fit(X_train,  
                                 y_train,   
                                 epochs=21,  
                                 batch_size=64,  
                                 validation_data=(X_valid, y_valid));  

    # Return the training results, which include loss and metrics for each epoch
    return training_results

def showTrainingResults(training_results):
    # Retrieve training results.
    train_loss = training_results.history["loss"]
    train_acc  = training_results.history["accuracy"]
    valid_loss = training_results.history["val_loss"]
    valid_acc  = training_results.history["val_accuracy"]

    plotResults(
        [train_loss, valid_loss],
        ylabel="Loss",
        ylim=[0.0, 0.5],
        metric_name=["Training Loss", "Validation Loss"],
        color=["g", "b"],
    )

    plotResults(
        [train_acc, valid_acc],
        ylabel="Accuracy",
        ylim=[0.9, 1.0],
        metric_name=["Training Accuracy", "Validation Accuracy"],
        color=["g", "b"],
    )

