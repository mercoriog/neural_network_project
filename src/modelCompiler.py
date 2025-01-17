import torch
import torch.nn as nn
import torch.nn.functional as F

def getFunc(function):
    if function == "relu":
        return F.relu
    elif function == "leaky_relu":
        return F.leaky_relu
    elif function == "tanh":
        return F.tanh
    else:
        raise ValueError(f"Unsupported activation function: {function}")

class DynamicModel(nn.Module):
    def __init__(self, hidden_num_layers, num_neurons, func_activation, input_shape):
        super(DynamicModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Aggiungi il primo layer con input_shape
        self.layers.append(nn.Linear(input_shape, num_neurons))
        
        # Aggiungi i layer intermedi
        for _ in range(hidden_num_layers):
            self.layers.append(nn.Linear(num_neurons, num_neurons))
        
        # Ultimo layer
        self.layers.append(nn.Linear(num_neurons, 10))

        # Funzione di attivazione
        self.activation_function = getFunc(func_activation)

    def forward(self, x):
        # Appiattisci l'input in un vettore 1D
        x = x.view(x.size(0), -1)  # Forma: [batch_size, input_shape]
        
        # Passa attraverso i layer
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Applica la funzione di attivazione a tutti i layer tranne l'ultimo
                x = self.activation_function(x)
        
        # Applica softmax all'ultimo layer
        x = F.softmax(x, dim=1)
        
        return x