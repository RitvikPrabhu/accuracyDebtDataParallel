import torch
import math
import logging
from dataclasses import dataclass, field
from typing import List, Any
from quantom_ips.utils.registration import register_with_hydra
from model_utils import mlp_block, get_init, get_activation

logger = logging.getLogger(__name__)

@dataclass
class MGOptimizerDefaults:
    id: str = "MGOptimizer"
    n_inputs: int = 100
    n_layers: int = 4
    n_neurons: int = 128
    activation: str = 'leaky_relu'
    weight_init: str = 'kaiming_normal'
    bias_init: str = 'zeros'
    output_weight_init: str = 'xavier_normal'
    output_bias_init: str = 'zeros'
    min_scalers: List[Any] = field(default_factory=lambda: [[-0.5,0.1],[0.1,0.05]])
    max_scalers: List[Any] = field(default_factory=lambda: [[-2.5,1.2],[2.8,1.0]])

@register_with_hydra(
    group="optimizer",
    defaults=MGOptimizerDefaults,
    name="mg_optimizer",
)
class MGOptimizer:

    def __init__(self,config,devices="cpu",dtype=torch.float32):
        self.config = config
        self.devices = devices
        self.dtype = dtype

        loc_min_scalers = torch.as_tensor(config.min_scalers,device=devices,dtype=dtype)
        loc_max_scalers = torch.as_tensor(config.max_scalers,device=devices,dtype=dtype)


        self.model = Generator(
            n_inputs=config.n_inputs,
            n_layers=config.n_layers,
            n_neurons=config.n_neurons,
            activation=config.activation,
            weight_init=config.weight_init,
            bias_init=config.bias_init,
            output_weight_init=config.output_weight_init,
            output_bias_init=config.output_bias_init,
            min_scalers=loc_min_scalers,
            max_scalers=loc_max_scalers
        ).to(self.devices,self.dtype)

    def forward(self,batch_size):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(batch_size,self.config.n_inputs),
            device=self.devices,
            dtype=self.dtype
        )
        return self.model.forward(noise)

    def predict(self,noise):
        return self.model.forward(noise)

class Generator(torch.nn.Module):

    def __init__(self, n_inputs, n_layers, n_neurons, activation, weight_init, bias_init, output_weight_init, output_bias_init, min_scalers, max_scalers):
        super().__init__()
        self.min_scalers = min_scalers
        self.max_scalers = max_scalers
      
        # Define base: 
        self.base = torch.nn.Sequential(mlp_block(
            n_inputs,
            n_layers,
            n_neurons,
            activation,
            weight_init,
            bias_init,
            'g'
        ))
        # Output for instance 1:
        self.output1 = torch.nn.Linear(n_neurons,2)
        self.output_act1 = get_activation("sigmoid")
        get_init(output_weight_init)(self.output1.weight)
        get_init(output_bias_init)(self.output1.bias)

        # Output for instance 2:
        self.output2 = torch.nn.Linear(n_neurons,2)
        self.output_act2 = get_activation("sigmoid")
        get_init(output_weight_init)(self.output2.weight)
        get_init(output_bias_init)(self.output2.bias)

    def forward(self,x):
        x_base = self.base(x)
        p1 = self.output_act1(self.output1(x_base))
        p2 = self.output_act2(self.output2(x_base))
       
        p10 = self.scale_tensor(p1[:,0],self.min_scalers[0][0],self.max_scalers[0][0])
        p11 = self.scale_tensor(p1[:,1],self.min_scalers[0][1],self.max_scalers[0][1])
        
        p20 = self.scale_tensor(p2[:,0],self.min_scalers[1][0],self.max_scalers[1][0])
        p21 = self.scale_tensor(p2[:,1],self.min_scalers[1][1],self.max_scalers[1][1])

        return torch.stack([
            torch.cat([p10[:,None],p11[:,None]],1),
            torch.cat([p20[:,None],p21[:,None]],1),
        ])
    
    def scale_tensor(self,tensor,min_s,max_s):
        return (max_s - min_s) * tensor + min_s



