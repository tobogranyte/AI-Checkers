import torch
import torch.nn as nn

class Torch:
    def __init__(self):
		self.batch_num = 0
		self.layers_dims = [397, 1024, 512, 256, 128, 96] #  5-layer model
        self.learning_rate = 0.001
        checkpoint = False
        self.network = Network(layers_dims)
    
    

    
class Network(nn.Module):
    
    def __init__(self, layers_dims):
        super().__init__()
        layers = []
        for n in range(len(layers_dims) - 2):
            layers.append(nn.Linear(layers_dims[n], layers_dims[n+1]), nn.ReLU())
        layers.append(nn.Linear(layers_dims[n+1],layers_dims[n+2]), nn.Softmax())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        
        return self.model(x)

        