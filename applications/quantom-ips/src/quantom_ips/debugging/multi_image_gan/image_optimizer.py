import torch
import logging
from dataclasses import dataclass, field
from typing import List, Any
from collections import OrderedDict
from quantom_ips.utils.registration import register_with_hydra
from torch_model_components import convtranspose2d_block, ConcreteDropout

logger = logging.getLogger(__name__)

@dataclass
class ImageOptimizerDefaults:
    id: str = "ImageOptimizer"
    nz: int = 100
    ngfs: List[Any] = field(default_factory=lambda: [521,256,64,32,2])
    kernels: List[Any] = field(default_factory=lambda: [4,4,4,4,4])
    strides: List[Any] = field(default_factory=lambda: [1,2,2,2,2])
    paddings: List[Any] = field(default_factory=lambda: [0,1,1,1,1])
    activations: List[Any] = field(default_factory=lambda: ['leaky_relu','leaky_relu','leaky_relu','leaky_relu','sigmoid'])
    weight_inits: List[Any] = field(default_factory=lambda: ['kaiming_normal','kaiming_normal','kaiming_normal','kaiming_normal','xavier_normal'])
    bias_inits: List[Any] = field(default_factory=lambda: ['normal']*5)
    dropouts: List[Any] = field(default_factory=lambda: [0.0]*5)
    global_dropout: float = 0.0
    concrete_dropout_weight_regs: List[Any] = field(default_factory=lambda: [0.0]*5)
    concrete_dropout_drop_regs: List[Any] = field(default_factory=lambda: [0.0]*5)
    concrete_temp: float = 0.1

@register_with_hydra(
    group="optimizer",
    defaults=ImageOptimizerDefaults,
    name="image_optimizer",
)
class ImageOptimizer:
    def __init__(self,config,devices="cpu",dtype=torch.float32):
        self.config = config
        self.devices = devices
        self.dtype = dtype

        self.model = Generator(
            config.nz,
            config.ngfs,
            config.kernels,
            config.strides,
            config.paddings,
            config.activations,
            config.weight_inits,
            config.bias_inits,
            config.dropouts,
            config.global_dropout,
            config.concrete_dropout_weight_regs,
            config.concrete_dropout_drop_regs,
            config.concrete_temp,
            devices,
            dtype
        )
    
    def forward(self,batch_size):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(batch_size,self.config.nz,1,1),
            device=self.devices,
            dtype=self.dtype
        )
        return self.model.forward(noise)

    def predict(self,noise,T=0):
        if T > 0:
            response = []
            for _ in range(T):
                response.append(self.model.forward(noise))
            return torch.stack(response)

        return self.model.forward(noise)
    
    
class Generator(torch.nn.Module):
    def __init__(self,nz,ngfs,kernels,strides,paddings,activations,weight_inits,bias_inits,dropouts,global_dropout,concrete_drop_weight_regs,concrete_drop_drop_regs,concrete_temp,devices,dtype):
        super(Generator, self).__init__()
        n_layers = len(ngfs)
        layers = OrderedDict()
        n_prev_filters = nz
 
        self.use_concrete_dropout = False
        
        for i in range(n_layers):
            current_layer, current_act = convtranspose2d_block(
                n_prev_filters,
                ngfs[i],
                kernels[i],
                strides[i],
                paddings[i],
                activations[i],
                weight_inits[i],
                bias_inits[i]
            )
            
            # Use concrete dropout, but then no other dropout anywhere else:
            if concrete_drop_weight_regs[i] > 0.0 and concrete_drop_drop_regs[i] > 0.0:
               self.use_concrete_dropout = True
               layers[f'layer_{i}'] = ConcreteDropout(current_layer,concrete_drop_weight_regs[i],concrete_drop_drop_regs[i],concrete_temp)
               layers[f'activation_{i}'] = current_act
            else:
               layers[f'layer_{i}'] = current_layer
               layers[f'activation_{i}'] = current_act
               if global_dropout > 0.0:
                  layers[f'dropout_{i}'] = torch.nn.Dropout2d(global_dropout)
               elif dropouts[i] > 0.0:
                  layers[f'dropout_{i}'] = torch.nn.Dropout2d(dropouts[i])

            n_prev_filters = ngfs[i]

        self.model = torch.nn.Sequential(layers).to(device=devices,dtype=dtype)

    def forward(self, input):
        return self.model(input)
    
    def add_regularization_loss(self,loss):
        if self.use_concrete_dropout:
          return loss + sum(module.regularization for module in self.model.modules() if hasattr(module, 'regularization'))
        return loss