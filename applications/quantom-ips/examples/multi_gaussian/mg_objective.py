import torch
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Any
from quantom_ips.utils.registration import register_with_hydra
from model_utils import mlp_block, get_init, get_activation

logger = logging.getLogger(__name__)

@dataclass
class MGObjectiveDefaults:
    id: str = "MGObjective"
    n_inputs: int = 2
    n_layers: int = 4
    n_neurons: int = 128
    activation: str = 'leaky_relu'
    weight_init: str = 'kaiming_normal'
    bias_init: str = 'zeros'
    output_weight_init: str = 'xavier_normal'
    output_bias_init: str = 'zeros'
    loss_fn: str = "mse"

@register_with_hydra(
    group="environment/objective",
    defaults=MGObjectiveDefaults,
    name="mg_objective",
)
class MGObjective:

    def __init__(self,config,devices="cpu",dtype=torch.float32):
        self.config = config
        self.devices = devices
        self.dtype = dtype

        self.model = Discriminator(
            n_inputs=config.n_inputs,
            n_layers=config.n_layers,
            n_neurons=config.n_neurons,
            activation=config.activation,
            weight_init=config.weight_init,
            bias_init=config.bias_init,
            output_weight_init=config.output_weight_init,
            output_bias_init=config.output_bias_init
        ).to(self.devices,self.dtype)

        loss_fn_str = config.loss_fn
        self.loss_norm = 1.0
        if loss_fn_str.lower() == "mse":
            self.loss_fn = torch.nn.MSELoss()
            self.loss_norm = 0.25
        if loss_fn_str.lower() == "bce":
            self.loss_fn = torch.nn.BCELoss()
            self.loss_norm= -np.log(0.5)

    def forward(self,x_fake,x_real,training=False):
        losses = {}
        y_fake = self.predict(x_fake)
        losses['fake'] = self.loss_fn(y_fake,torch.ones_like(y_fake))

        if training:
            y_real = self.predict(x_real)
            losses['real'] = self.loss_fn(y_real,torch.ones_like(y_real))
            losses['fake'] = self.loss_fn(y_fake,torch.zeros_like(y_fake))

        return losses

    def predict(self,x):
        return self.model.forward(x)



class Discriminator(torch.nn.Module):

    def __init__(self, n_inputs, n_layers, n_neurons, activation, weight_init, bias_init, output_weight_init, output_bias_init):
        super().__init__()
        
        # Define base: 
        self.base = torch.nn.Sequential(mlp_block(
            n_inputs,
            n_layers,
            n_neurons,
            activation,
            weight_init,
            bias_init,
            'd'
        ))
        # Output :
        self.output = torch.nn.Linear(n_neurons,1)
        self.output_act = get_activation("sigmoid")
        get_init(output_weight_init)(self.output.weight)
        get_init(output_bias_init)(self.output.bias)

    def forward(self,x):
        x_base = self.base(x)
        return self.output_act(self.output(x_base))
