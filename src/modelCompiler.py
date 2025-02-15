import torch
import torch.nn as nn
import torch.nn.functional as F

# Restituisce la funzione di attivazione da utilizzare
def getFunc(function):
    if function == "relu":
        return F.relu
    elif function == "leaky_relu":
        return F.leaky_relu
    elif function == "tanh":
        return F.tanh
    else:
        raise ValueError(f"Unsupported activation function: {function}")

# Classe per definire la rete neurale
class NN(nn.Module):
    def __init__(self, input_size, num_neurons, hidden_num_layers, func_activation, num_classes):
        super(NN, self).__init__()
        
        # Lista per memorizzare i layer
        self.layers = nn.ModuleList()
        
        # Primo layer (input -> primo hidden layer)
        self.layers.append(nn.Linear(input_size, num_neurons))
        
        # Aggiungi i layer intermedi (hidden layers)
        for _ in range(hidden_num_layers):
            self.layers.append(nn.Linear(num_neurons, num_neurons))
        
        # Ultimo layer (ultimo hidden layer -> output)
        self.layers.append(nn.Linear(num_neurons, num_classes))
        
        # Funzione di attivazione
        self.activation_function = getFunc(func_activation)

    def forward(self, x):
         # Appiattisci l'input in un vettore 1D
        x = x.view(x.size(0), -1)  # Forma: [batch_size, input_shape]

        # Itera sui layer e applica la funzione di attivazione
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Applica la funzione di attivazione solo ai layer nascosti
            if i < len(self.layers) - 1:  # Non applicare alla fully connected finale
                x = self.activation_function(x)
        
        # Applica softmax all'ultimo layer
        x = F.softmax(x, dim=1)
        
        return x
    
