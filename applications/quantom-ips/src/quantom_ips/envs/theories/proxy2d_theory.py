import numpy as np
import torch
from quantom_ips.utils.registration import register_with_hydra
from dataclasses import dataclass, field
from dataclasses import dataclass

@dataclass
class Proxy2DTheoryDefaults:
    id: str = "Proxy2DTheory"
    a_min: list = field(default_factory=lambda: [0.0,0.0])
    a_max: list = field(default_factory=lambda: [1.0,1.0])
    x_min: float = 0.001
    x_max: float = 0.999
    y_min: float = 0.001
    y_max: float = 0.999
    ratios: list = field(default_factory=lambda: [0.9,0.75])
    average: bool = True

@register_with_hydra(
    group="environment/theory",
    defaults=Proxy2DTheoryDefaults,
    name="proxy2d",
)
class Proxy2DTheory:
    """
    Theory module that ingests a 2D array A mimicing a density and produces "x-sections" that are fed into LOITS. This module is an extension of the 
    Proxy 2D application that was used for the LOITS paper. 
    """

    # Initialize:
    # *************************
    def __init__(self, config, devices="cpu", dtype=torch.float32):
        self.devices = devices
        self.dtype = dtype

        # Basic settings in config:
        self.a_min = config.a_min
        self.a_max = config.a_max
        self.ratios = config.ratios
        self.x_min = config.x_min
        self.x_max = config.x_max
        self.y_min = config.y_min
        self.y_max = config.y_max
        self.average = config.average
    # *************************
    
    # Build a linear combination between the input densities
    # *************************
    def linear_combination(self,A):
        A_0 = A[0]*(self.a_max[0] - self.a_min[0]) + self.a_min[0]
        A_1 = A[1]*(self.a_max[1] - self.a_min[1]) + self.a_min[1]
        combos = [
            self.ratios[0]*A_0+(1.0-self.ratios[0])*A_1,
            (1.0-self.ratios[1])*A_0 + self.ratios[1]*A_1
        ]

        if len(self.ratios) > 2:
            combos.append(
                self.ratios[2]*A_0+(1.0-self.ratios[2])*A_1
            )

        return torch.stack(combos,dim=0)
    # *************************
    
    # Forward:
    # *************************
    def forward(self,A):
        # Get the dimensions from the predictions:
        n_points_x = A.size()[2]
        n_points_y = A.size()[3]

        # Compute axes:
        x_axis = torch.linspace(self.x_min,self.x_max,n_points_x,device=self.devices,dtype=self.dtype)
        y_axis = torch.linspace(self.y_min,self.y_max,n_points_y,device=self.devices,dtype=self.dtype)
        axes = [x_axis,y_axis]

        # Get the x-sections:
        x_secs = torch.vmap(self.linear_combination,in_dims=0)(A)

        if self.average:
            x_secs = x_secs.mean(dim=0,keepdim=True)

        # Provide loss as one output argument:
        zero_loss = torch.zeros(
            size=(1,), dtype=self.dtype, device=self.devices, requires_grad=True
        )
        
        return x_secs, axes, zero_loss
    # *************************
