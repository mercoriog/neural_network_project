
# Define a function to train the model
def training(model):
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