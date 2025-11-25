from mg_parser import MGParser
from mg_objective import MGObjective
from mg_environment import MGEnvironment
from mg_trainer import MGTrainer
from mg_analysis import MGAnalysis
from mg_optimizer import MGOptimizer
from quantom_ips.gradient_transport.torch_arar_chunk import TorchARARChunk
from quantom_ips.gradient_transport.torch_arar import TorchARAR
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from quantom_ips.utils.registration import register_with_hydra, make
from typing import List, Any
import hydra

defaults = [
    {"environment": "mg_environment"},
    {"optimizer": "mg_optimizer"},
    {"trainer": "mg_trainer"},
    {"analysis": "mg_analysis"},
    "_self_",
]

@dataclass
class RunMGDefaults:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    id: str = "RunMG"

cs = ConfigStore.instance()
cs.store(name="run_mg", node=RunMGDefaults)

@hydra.main(version_base=None, config_name="run_mg")
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