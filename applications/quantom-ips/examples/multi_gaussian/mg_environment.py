import torch
import logging
from dataclasses import dataclass, field
from typing import List, Any
from quantom_ips.utils.registration import register_with_hydra, make
from mg_pipeline import MGPipeline

logger = logging.getLogger(__name__)

defaults = [
    {"parser": "mg_parser"},
    {"objective": "mg_objective"},
    "_self_",
]

@dataclass
class MGEnvironmentDefaults:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    id: str = "MGEnvironment"
    alpha: List[Any] = field(default_factory=lambda: [[1.0,0.0,0.0],[0.0,1.0,0.0]])
    n_samples: int = 100
    mode: str = "baseline"
    min_scale: List[float] = field(default_factory=lambda: [0.0,0.0])
    max_scale: list[float] = field(default_factory=lambda: [1.0,1.0])

@register_with_hydra(
    group="environment",
    defaults=MGEnvironmentDefaults,
    name="mg_environment",
)
class MGEnvironment:

    def __init__(self, config, mpi_comm=None, devices="cpu", dtype=torch.float32):
        self.config = config
        self.alpha = config.alpha
        self.devices = devices
        self.dtype = dtype

        self.min_scale = torch.as_tensor(config.min_scale,device=devices,dtype=dtype)
        self.max_scale = torch.as_tensor(config.max_scale,device=devices,dtype=dtype)

        # Get the current rank (if mpi is existent):
        self.data_idx = None
        if mpi_comm is not None and config.mode.lower() == "multidata":
           current_rank = mpi_comm.Get_rank()
           self.data_idx = 0
           if current_rank % 2 == 0:
              self.data_idx = 1

           self.min_scale = self.min_scale[self.data_idx]
           self.max_scale = self.max_scale[self.data_idx]

           # Overwrite settings in data parser and objective:
           self.config.parser.data_idx = self.data_idx
           self.config.objective.n_inputs = 1

        # Data Parser
        self.data_parser = make(
            self.config.parser.id,
            config=self.config.parser,
            devices=self.devices,
            dtype=self.dtype,
        )

        # Objective:
        self.objective = make(
            self.config.objective.id,
            config=self.config.objective,
            devices=self.devices,
            dtype=self.dtype,
        )

        # Set up the pipeline:
        self.pipeline = MGPipeline(config.n_samples, config.alpha)
        
        self.x_min = self.data_parser.data.min(0)[0]
        self.x_max = self.data_parser.data.max(0)[0]

       
    def scale_data(self,x):
        x_std = (x - self.x_min) / (self.x_max - self.x_min)
        return (self.max_scale - self.min_scale) * x_std + self.min_scale

    def step(self, params):
        # Create fake data:
        fake_data = self.pipeline.forward(params,self.data_idx)
        # Make loss accessible:
        fake_data_norm = self.scale_data(fake_data)

        return self.objective.forward(fake_data_norm,None,False), fake_data
        
    def get_objective_losses(self,fake_data):
        real_data = self.data_parser.forward(fake_data.size()[0])
        fake_data_norm = self.scale_data(fake_data)
        real_data_norm = self.scale_data(real_data)

        loss = self.objective.forward(fake_data_norm.detach(),real_data_norm.detach(),training=True)

        del real_data
        del fake_data
        return loss

        