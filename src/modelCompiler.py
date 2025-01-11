from tensorflow.python.keras import Sequential
from tensorflow.python.layers import Dense, Dropout, LeakyReLU

# Define a function to create the model
def model(hidden_num_layers, num_neurons, func_activation, X_train):
    # Instantiate the Sequential model
    model = Sequential()
    
    # 1) Add the first Dense layer with 128 neurons and ReLU activation
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    
     # Add hidden layers with the specified activation function
    for i in range(hidden_num_layers):
        if func_activation == 'leaky_relu':
            model.add(Dense(num_neurons))
            model.add(LeakyReLU(alpha=0.1))  # Leaky ReLU with alpha=0.1
        else:
            model.add(Dense(num_neurons, activation=func_activation))
    
    # Add the output Dense layer with 10 neurons (for 10 classes) and softmax activation
    model.add(Dense(10, activation="softmax"))
    model.add(LeakyReLU())
    # Display the model architecture summary
    model.summary()

    # Return the created model
    return model


# Compile the model
def modelCompile(model):
    # Compile the model with RMSprop optimizer, categorical crossentropy loss, and accuracy metric
    model.compile(
        optimizer="rmsprop",  # Optimizer for weight updates
        loss="categorical_crossentropy",  # Loss function for multi-class classification
        metrics=["accuracy"],  # Metric to monitor during training
    )
    return model