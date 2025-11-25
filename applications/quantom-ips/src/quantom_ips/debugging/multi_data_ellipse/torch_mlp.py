import torch
from collections import OrderedDict

class TorchMLP(torch.nn.Module):

    def __init__(self,n_inputs,n_outputs,n_hidden,n_neurons,dropouts):
        super().__init__()
        n_previous = n_inputs
        layers = OrderedDict()
        for i in range(n_hidden):
            current_layer = torch.nn.Linear(n_previous,n_neurons)
            w = current_layer.weight
            b = current_layer.bias

            torch.nn.init.kaiming_normal_(w)
            torch.nn.init.zeros_(b)

            layers[f'layer{i}'] = current_layer
            layers[f'activation{i}'] = torch.nn.LeakyReLU(0.2)

            if dropouts > 0.0:
                layers[f'dropout{i}'] = torch.nn.Dropout(dropouts)

            n_previous = n_neurons

        output_layer = torch.nn.Linear(n_previous,n_outputs)
        w = output_layer.weight
        b = output_layer.bias

        torch.nn.init.xavier_normal_(w)
        torch.nn.init.zeros_(b)

        layers['output'] = output_layer
        layers['output_activation'] = torch.nn.Sigmoid()

        self.model = torch.nn.Sequential(layers)

    def forward(self,x):
        return self.model(x)