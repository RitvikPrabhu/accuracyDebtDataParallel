import numpy as np
from image_environment import ImageEnvironment
from image_objective import ImageObjective
from image_parser import ImageParser
from image_trainer import ImageTrainer
from image_optimizer import ImageOptimizer
from image_analysis import ImageAnalysis
from quantom_ips.gradient_transport.torch_arar_pcgrad import TorchARARPCGrad
from quantom_ips.gradient_transport.torch_arar import TorchARAR
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from quantom_ips.utils.registration import register_with_hydra, make
from typing import List, Any
import hydra
import torch
import os

defaults = [
    {"environment": "image_environment"},
    {"optimizer": "image_optimizer"},
    {"trainer": "image_trainer"},
    {"analysis": "image_analysis"},
    "_self_",
]

@dataclass
class RunImageGANDefaults:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    id: str = "RunImageGAN"

cs = ConfigStore.instance()
cs.store(name="run_image_gan", node=RunImageGANDefaults)

@hydra.main(version_base=None, config_name="run_image_gan")
def run(config) -> None:
  
  # Get the trainer first:
  trainer = make(
     id=config.trainer.id,
     config=config.trainer
  )

  comm = trainer.comm
  devices = trainer.devices
  dtype = trainer.dtype
  
  # Get the environment:
  env = make(
     id=config.environment.id,
     config=config.environment,
     mpi_comm=comm,
     devices=devices,
     dtype=dtype
  )
  
  # Get the optimizer: 
  opt = make(
     id=config.optimizer.id,
     config=config.optimizer,
     devices=devices,
     dtype=dtype
  )

  # Last, but not least, get the analysis:
  analysis = make(
     id=config.analysis.id,
     config=config.analysis,
     mpi_comm=comm,
     devices=devices,
     dtype=dtype
  )

  # Now run everything:
  trainer.run(opt,env,analysis)

if __name__ == "__main__":
    run()