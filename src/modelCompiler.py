import tensorflow as tf
from .RProp import RProp

# Define a function to create the model
def model(hidden_num_layers, num_neurons, func_activation, X_train):
    # Instantiate the Sequential model
    model = tf.keras.Sequential()
    
    # 1) Add the first Dense layer with 128 neurons and ReLU activation
    model.add(tf.keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)))
    
    # Add hidden layers with the specified activation function
    for i in range(hidden_num_layers):
        if func_activation == "leaky_relu":
            model.add(tf.keras.layers.Dense(num_neurons))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))  # Leaky ReLU with alpha=0.1
        else:
            model.add(tf.keras.layers.Dense(num_neurons, activation=func_activation))
    
    # Add the output Dense layer with 10 neurons (for 10 classes) and softmax activation
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    # Display the model architecture summary
    model.summary()

    # Return the created model
    return model


# Compile the model
def modelCompile(model):
    # Create rprop obj
    rprop_optimizer = RProp(init_alpha=1e-3, scale_up=1.2, scale_down=0.5, min_alpha=1e-6, max_alpha=50.0, learning_rate=1e-3)
    # Compile the model with RMSprop optimizer, categorical crossentropy loss, and accuracy metric
    model.compile(
        optimizer=rprop_optimizer,  # Optimizer for weight updates
        loss="categorical_crossentropy",  # Loss function for multi-class classification
        metrics=["accuracy"],  # Metric to monitor during training
    )
    return model