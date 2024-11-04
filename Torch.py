import torch
import torch.nn as nn

class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.batch_num = 0
        self.layers_dims = [397, 1024, 512, 256, 128, 96] #  5-layer model
        self.learning_rate = 0.001
        checkpoint = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        layers = []
        for n in range(len(self.layers_dims) - 2):
            layers.append(nn.Linear(self.layers_dims[n], self.layers_dims[n+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.layers_dims[n+1], self.layers_dims[n+2]))
        layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.activations = {}  # Dictionary to store activations
        self.hook_handles = []  # List to store hook handles
        self.model = self.model.to(self.device)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def add_hooks(self):
        # Register hooks and store handles
        for name, layer in self.named_modules():
            handle = layer.register_forward_hook(self._save_activation(name))
            self.hook_handles.append(handle)

    def remove_hooks(self):
        # Remove all hooks using stored handles
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()  # Clear the list after removal

    def forward(self, x, nograd = True):
        x = self.convert(x)
        x = x.to(self.device)
        if nograd:
            with torch.no_grad():
                x = self.model(x)
                x = self.deconvert(x)
                return x
        else:
            return self.model(x)

    def _save_activation(self, name):
        # Hook function to save activations
        def hook(model, input, output):
            self.activations[name] = output.detach().cpu()
        return hook

    def convert(self, x):
        x = torch.from_numpy(np.array(x, dtype=np.float32)).transpose(0,1)
        return x

    def deconvert(self, x):
        if x.requires_grad:
            x = x.transpose(0,1).detach().numpy().astype(np.float64)
        else:
            x = x.transpose(0,1).numpy().astype(np.float64)
        return x
    