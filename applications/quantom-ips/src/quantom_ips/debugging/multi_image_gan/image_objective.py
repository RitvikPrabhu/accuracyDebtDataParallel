import torch
import numpy as np
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Any
from quantom_ips.utils.registration import register_with_hydra
from torch_model_components import conv2d_block

logger = logging.getLogger(__name__)

@dataclass
class ImageObjectiveDefaults:
    id: str = "ImageObjective"
    nc: int = 1
    loss_fn: str = "MSE"
    ndfs: List[Any] = field(default_factory=lambda: [32,64,256,512,1])
    kernels: List[Any] = field(default_factory=lambda: [4,4,4,4,4])
    strides: List[Any] = field(default_factory=lambda: [2,2,2,2,1])
    paddings: List[Any] = field(default_factory=lambda: [1,1,1,1,0])
    activations: List[Any] = field(default_factory=lambda: ['leaky_relu','leaky_relu','leaky_relu','leaky_relu','sigmoid'])
    weight_inits: List[Any] = field(default_factory=lambda: ['kaiming_normal','kaiming_normal','kaiming_normal','kaiming_normal','xavier_normal'])
    bias_inits: List[Any] = field(default_factory=lambda: ['normal']*5)
    dropouts: List[Any] = field(default_factory=lambda: [0.0]*5)
    global_dropout: float = 0.0

@register_with_hydra(
    group="environment/objective",
    defaults=ImageObjectiveDefaults,
    name="image_objective",
)
class ImageObjective:

    def __init__(self,config,devices="cpu",dtype=torch.float32):
        self.config = config
        self.devices = devices
        self.dtype = dtype
        self.model = Discriminator(
            config.nc,
            config.ndfs,
            config.kernels,
            config.strides,
            config.paddings,
            config.activations,
            config.weight_inits,
            config.bias_inits,
            config.dropouts,
            config.global_dropout,
            devices,
            dtype
        )

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
        y_fake = self.predict(x_fake).view(-1)
        losses['fake'] = self.loss_fn(y_fake,torch.ones_like(y_fake))

        if training:
            y_real = self.predict(x_real).view(-1)
            losses['real'] = self.loss_fn(y_real,torch.ones_like(y_real))
            losses['fake'] = self.loss_fn(y_fake,torch.zeros_like(y_fake))

        return losses

    def predict(self,x):
        return self.model.forward(x)

class Discriminator(torch.nn.Module):
    def __init__(self,nc,ndfs,kernels,strides,paddings,activations,weight_inits,bias_inits,dropouts,global_dropout,devices,dtype):
        super(Discriminator, self).__init__()
        n_layers = len(ndfs)
        layers = OrderedDict()
        n_prev_filters = nc
        for i in range(n_layers):
            current_layer, current_act = conv2d_block(
                n_prev_filters,
                ndfs[i],
                kernels[i],
                strides[i],
                paddings[i],
                activations[i],
                weight_inits[i],
                bias_inits[i]
            )
            layers[f'layer_{i}'] = current_layer
            layers[f'activation_{i}'] = current_act
            
            if global_dropout > 0.0:
                layers[f'dropout_{i}'] = torch.nn.Dropout2d(global_dropout)
            elif dropouts[i] > 0.0:
                layers[f'dropout_{i}'] = torch.nn.Dropout2d(dropouts[i])

            n_prev_filters = ndfs[i]

        self.model = torch.nn.Sequential(layers).to(device=devices,dtype=dtype)
        

    def forward(self, input):
        return self.model(input)