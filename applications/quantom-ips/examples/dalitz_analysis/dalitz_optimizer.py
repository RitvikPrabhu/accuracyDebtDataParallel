import torch
from collections import OrderedDict
from dataclasses import dataclass, field
import quantom_ips.utils.torch_nn_registry as reg
from quantom_ips.utils.registration import register_with_hydra

@dataclass
class DalitzOptimizerDefaults:
    id: str = "DalitzGenerator"
    in_features: int = 100
    n_block_neurons: int = 128
    n_block_layers: int = 4
    n_final_neurons: int = 64
    n_final_layers: int = 2
    activation: str = "LeakyReLU"
    weight_init: str = "kaiming_normal"
    bias_init: str = "zeros"
    scalers: list = field(default_factory=lambda: [[2.0,0.7,0.1,0.2,0.1,0.5],[0.1]*6,[0.7,0.2,0.0,0.0,0.0,0.0]])

@register_with_hydra(group="optimizer", defaults=DalitzOptimizerDefaults, name="dalitz_optimizer")
class DalitzGenerator:
    def __init__(self, config, devices, dtype):
        self.id = config.id
        self.config = config
        self.devices = devices
        self.dtype = dtype
        self.model = Generator(
            n_inputs=config.in_features,
            n_block_layers=config.n_block_layers,
            n_block_neurons=config.n_block_neurons,
            n_final_layers=config.n_final_layers,
            n_final_neurons=config.n_final_neurons,
            activation=config.activation,
            weight_init=config.weight_init,
            bias_init=config.bias_init,
            scalers=config.scalers,
            devices=self.devices,
            dtype=self.dtype
        )


    def forward(self,batch_size):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(batch_size, self.config.in_features),
            device=self.devices,
            dtype=self.dtype,
        )
        return self.model.forward(noise)
    
    def predict(self,x):
        return self.model.forward(x)
    
class Generator(torch.nn.Module):

    def __init__(self,n_inputs,n_block_neurons,n_block_layers,n_final_neurons,n_final_layers,activation,weight_init,bias_init,scalers,devices,dtype):
        super().__init__()
        self.devices = devices
        self.dtype = dtype
        self.scales = [
            torch.as_tensor(s,device=devices,dtype=dtype) for s in scalers
        ]
        self.n_block_layers = n_block_layers
        self.n_block_neurons = n_block_neurons
        self.n_final_neurons = n_final_neurons
        self.n_final_layers = n_final_layers
        self.activation = activation
        self.weight_init = weight_init
        self.bias_init = bias_init

        # Linear block:
        self.linear_block = reg.get_layer("LinearStack",
            n_inputs=n_inputs,
            n_neurons=n_block_neurons,
            n_layers=n_block_layers,
            activation=activation,
            weight_init=weight_init,
            bias_init=bias_init
        ).to(self.devices,self.dtype)
        
        # Final layers for individual parameter prediction:
        self.final_layers = []
        for i in range(len(self.scales)):
            identifier = f'final{i}'
            self.final_layers.append(
                self.build_final_block(identifier,len(self.scales[i]))
            )

    def build_final_block(self,identifier,n_outputs):
      layer = OrderedDict()
      layer[f'{identifier}_block'] = reg.get_layer("LinearStack",
        n_inputs=self.n_block_neurons,
        n_neurons=self.n_final_neurons,
        n_layers=self.n_final_layers,
        activation=self.activation,
        weight_init=self.weight_init,
        bias_init=self.bias_init
      )

      output_layer = torch.nn.Linear(self.n_final_neurons,n_outputs)
      torch.nn.init.xavier_normal_(output_layer.weight)
      torch.nn.init.normal_(output_layer.bias)
      layer[f'{identifier}_output'] = output_layer
      layer[f'{identifier}_activation'] = torch.nn.Tanh()
      return torch.nn.Sequential(layer).to(self.devices,self.dtype)
    
    def forward(self,x):
        x_block = self.linear_block(x)
        predictions = []
        for l,s in zip(self.final_layers,self.scales):
            predictions.append(l(x_block)*s)
        return torch.stack(predictions)


    