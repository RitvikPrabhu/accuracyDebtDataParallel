import torch
from collections import OrderedDict

def get_init(init_str):
    if init_str.lower() == "kaiming_normal":
        return torch.nn.init.kaiming_normal_
    if init_str.lower() == "kaiming_uniform":
        return torch.nn.init.kaiming_uniform_
    if init_str.lower() == "xavier_normal":
        return torch.nn.init.xavier_normal_
    if init_str.lower() == "xavier_uniform":
        return torch.nn.init.xavier_uniform_
    if init_str.lower() == "normal":
        return torch.nn.init.normal_
    if init_str.lower() == "uniform":
        return torch.nn.init.uniform_
    if init_str.lower() == "zeros":
        return torch.nn.init.zeros_

def get_activation(act_str):
    if act_str.lower() == "relu":
        return torch.nn.ReLU()
    if act_str.lower() == "leaky_relu":
        return torch.nn.LeakyReLU(0.2)
    if act_str.lower() == "tanh":
        return torch.nn.Tanh()
    if act_str.lower() == "sigmoid":
        return torch.nn.Sigmoid()
    if act_str.lower() == "hardtanh":
        return torch.nn.Hardtanh(0.0,1.0)
    
def mlp_block(n_inputs, n_layers, n_neurons, activation, weight_init, bias_init, identifier):
    n_prev_neurons = n_inputs
    layers = OrderedDict()

    for i in range(n_layers):
        current_layer = torch.nn.Linear(n_prev_neurons,n_neurons)
        get_init(weight_init)(current_layer.weight)
        get_init(bias_init)(current_layer.bias)

        layers[f'{identifier}_layer{i}'] = current_layer
        layers[f'{identifier}_activation{i}'] = get_activation(activation)

        n_prev_neurons = n_neurons

    return layers