import torch
import logging
from dataclasses import dataclass, field
from typing import List, Any
from quantom_ips.utils.registration import register_with_hydra, make
from quantom_ips.debugging.multi_image_gan.pipeline import pipeline

logger = logging.getLogger(__name__)

defaults = [
    {"parser": "image_parser"},
    {"objective": "image_objective"},
    "_self_",
]

@dataclass
class ImageEnvironmentDefaults:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    id: str = "ImageEnvironment"
    alphas: List[Any] = field(default_factory=lambda: [[0.5,0.5],[0.5,0.5]])
    
@register_with_hydra(
    group="environment",
    defaults=ImageEnvironmentDefaults,
    name="image_environment",
)
class ImageEnvironment:
    def __init__(self, config, mpi_comm=None, devices="cpu", dtype=torch.float32):
        self.config = config
        self.alphas = config.alphas
        self.devices = devices
        self.dtype = dtype
        
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

        # Get the current rank (if mpi is existent):
        self.data_idx = -1
        if mpi_comm is not None:
           current_rank = mpi_comm.Get_rank()
           self.data_idx = 0
           if current_rank % 2 == 0:
              self.data_idx = 1
    
    def step(self, params):
        # Create fake data:
        fake_data = torch.vmap(lambda p: pipeline(p,self.alphas,True),in_dims=0)(params)
        # Add noise:
        fake_data = self.data_parser.add_noise(fake_data)

        if self.data_idx >= 0:
            fake_data = fake_data[:,self.data_idx,:,:][:,None,:,:]
            # Get the objective score / loss:
            return self.objective.forward(fake_data,None,False), fake_data
        
        return None, fake_data
    
    def get_objective_losses(self,fake_data):
        real_data = self.data_parser.get_samples(fake_data.size()[0])[:,self.data_idx,:,:][:,None,:,:]
        loss = self.objective.forward(fake_data.detach(),real_data.detach(),training=True)

        del real_data
        del fake_data
        return loss


        

        

        