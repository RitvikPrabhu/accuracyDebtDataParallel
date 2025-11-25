import torch

# import quantom_ips.utils.torch_nn_registry as torch_nn_registry
from quantom_ips.utils import torch_nn_registry
from collections import OrderedDict


# Defaults for Conv2dTranspose are stride=1, padding=0, dilation=1 in the first layer
# and stride = 2 and padding = 1 in all other layers
class ConvTranspose2dStack(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        internal_channels=None,
        n_layers=1,
        activation="ReLU",
        weight_init="kaiming_normal",
        bias_init="normal",
        out_activation=None,
        weight_out_init="xavier_normal",
        bias_out_init="normal"
    ) -> None:
        super().__init__()
        if internal_channels is None:
            internal_channels = out_channels

        internal_activation = self.make_activation(activation)
        if out_activation is None:
            out_activation = internal_activation
        else:
            out_activation = self.make_activation(out_activation)

        self.layers = OrderedDict()
        self.layers["ConvTranspose2d_0"] = torch.nn.ConvTranspose2d(
            in_channels, internal_channels, 3, padding=0
        )
        self.layers["ConvTranspose2d_act0"] = internal_activation
        self.initialize_layer(self.layers["ConvTranspose2d_0"],weight_init,bias_init)

        for i in range(1, n_layers - 1):
            self.layers[f"ConvTranspose2d_{i}"] = torch.nn.ConvTranspose2d(
                internal_channels, internal_channels, kernel_size, stride=2, padding=1
            )
            self.layers[f"ConvTranspose2d_act{i}"] = internal_activation
            self.initialize_layer(self.layers[f"ConvTranspose2d_{i}"],weight_init,bias_init)

        self.layers[f"ConvTranspose2d_{n_layers}"] = torch.nn.ConvTranspose2d(
            internal_channels, out_channels, kernel_size, stride=2, padding=1
        )
        self.layers[f"ConvTranspose2d_act{n_layers}"] = out_activation
        self.initialize_layer(self.layers[f"ConvTranspose2d_{n_layers}"],weight_out_init,bias_out_init)

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
    
    def initialize_layer(self,layer,w_init,b_init):
        w = layer.weight
        b = layer.bias
        
        if w_init is not None and w_init != "":
           torch_nn_registry.get_initializer(w_init,w)
        if b_init is not None and b_init != "":
           torch_nn_registry.get_initializer(b_init,b)
       