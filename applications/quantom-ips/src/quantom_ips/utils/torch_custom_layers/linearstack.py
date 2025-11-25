import torch

# import quantom_ips.utils.torch_nn_registry as torch_nn_registry
from quantom_ips.utils import torch_nn_registry
from collections import OrderedDict


# Defaults for Conv2dTranspose are stride=1, padding=0, dilation=1 in the first layer
# and stride = 2 and padding = 1 in all other layers
class LinearStack(torch.nn.Module):
    def __init__(
        self,
        n_inputs,
        n_neurons=None,
        n_layers=1,
        activation="ReLU",
        weight_init="kaiming_normal",
        bias_init="normal",
        dropout=0.0,
    ) -> None:
        super().__init__()
        if n_neurons is None:
            n_neurons = n_inputs

        internal_activation = self.make_activation(activation)
        
        self.layers = OrderedDict()
        n_prev_neurons = n_inputs
        for i in range(n_layers):
            # Define the layer:
            current_layer = torch.nn.Linear(n_prev_neurons,n_neurons)
            # Initialize it:
            self.init_layer(current_layer,weight_init,bias_init)
            # Register it:
            self.layers[f'Linear_{i}'] = current_layer
            # Register the activation:
            self.layers[f'Linear_act{i}'] = internal_activation
            # Add dropouts, if requested:
            if dropout > 0.0:
                self.layers[f'Dropout_{i}'] = torch.nn.Dropout(dropout)

            n_prev_neurons = n_neurons
        
        self.model = torch.nn.Sequential(self.layers)

    def forward(self, x):
        return self.model(x)

    def make_activation(self, activation):
        try:
            if isinstance(activation, dict):
                act = torch_nn_registry.get_activation(
                    activation["class"], **activation["config"]
                )
            else:
                act = torch_nn_registry.get_activation(activation)
        except Exception as e:
            err_msg = (
                "Activation must be a string or a dictionary with "
                "keys 'class' and 'config'"
            )
            raise Exception(err_msg) from e

        return act
    
        
    def init_layer(self,layer,weight_init,bias_init):
        torch_nn_registry.get_initializer(weight_init,layer.weight)
        torch_nn_registry.get_initializer(bias_init,layer.bias)
        
    
