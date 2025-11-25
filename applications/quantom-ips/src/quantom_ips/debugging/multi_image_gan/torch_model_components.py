import torch
import numpy as np

def conv2d_block(in_channels,out_channels,kernel_size,stride,padding,activation,weight_init,bias_init):
    layer = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    activation = get_activation(activation)
    get_init(weight_init)(layer.weight)
    get_init(bias_init)(layer.bias)

    return layer, activation

def convtranspose2d_block(in_channels,out_channels,kernel_size,stride,padding,activation,weight_init,bias_init):
    layer = torch.nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    activation = get_activation(activation)
    get_init(weight_init)(layer.weight)
    get_init(bias_init)(layer.bias)

    return layer, activation

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
    

# Concrete dropout layer:
class ConcreteDropout(torch.nn.Module):
    def __init__(self, module, weight_regularizer=1e-6, dropout_regularizer=1e-5, temp=0.1):
        """
        module: nn.Module (e.g. nn.Conv2d) to wrap
        weight_regularizer: λ1 for weight decay term
        dropout_regularizer: λ2 for dropout rate regularizer
        """
        super().__init__()
        self.module = module
        # ρ parameter, initialized so dropout ~0.1
        init_p = 0.1
        self.rho = torch.nn.Parameter(torch.logit(torch.tensor(1-init_p)))
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.temp = temp

    def forward(self, x):
        # compute p from rho
        p = 1 - torch.sigmoid(self.rho)
        # sample concrete dropout mask
        eps = 1e-7
        u = torch.rand_like(x)
        # stretch+shift to get logistic noise
        s = torch.log(u + eps) - torch.log(1 - u + eps)
        drop_prob = (torch.log(p + eps) - torch.log(1 - p + eps) + s) / self.temp
        z = torch.sigmoid(drop_prob)
        x = x * (1-z) / (1-p)
        out = self.module(x)
        # compute regularizer (to be added to your loss)
        sum_of_squares = torch.sum(self.module.weight ** 2)
        # weight decay term:
        wd = self.weight_regularizer * sum_of_squares / (1 - p)
        # dropout term:
        dp = p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps)
        dp = self.dropout_regularizer * x.numel() * dp
        # stash regularization losses
        self.regularization = wd + dp
        return out
    
# Monitor the trainable parameter of the concrete layer:
def get_dropout_probs(model):
    probs = []
    for n,p in model.named_parameters():
        if "rho" in n:
            prob = 1 - torch.sigmoid(p)
            probs.append(prob.detach().cpu().numpy())
    
    if len(probs) > 0:
       return np.array(probs)
    return None
    