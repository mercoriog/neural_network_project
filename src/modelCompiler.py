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
        raise

class DynamicModel(nn.Module):
    def __init__(self, hidden_num_layers, num_neurons, func_activation, input_shape):
        super(DynamicModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Aggiungi il primo layer con input_shape
        self.layers.append(nn.Linear(input_shape, 128))
        
        # Aggiungi i layer intermedi
        for i in range(1, hidden_num_layers):
            self.layers.append(nn.Linear(num_neurons, num_neurons))
        
        # Lista delle funzioni di attivazione
        self.activation_function = getFunc(func_activation)
    
    def forward(self, x):
        i = 0
        for layer in self.layers:
            x = layer(x)
            if i > 0:
                x = F.relu(x)
            elif i == (len(self.layers) - 1):
                x = F.sigmoid(x)
            else:
                x = self.activation_function(x)
        return x