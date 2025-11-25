import mpi4py
mpi4py.rc.thread_level = "serialized"
from mpi4py import MPI
import logging
import hydra
import torch
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import List, Any, Optional
from hydra.core.config_store import ConfigStore
from quantom_ips.analyses.distributed_base_analysis import DistributedBaseAnalysis
from dalitz_theory import DalitzTheory
from dalitz_optimizer import DalitzGenerator
from dalitz_environment import DalitzEnvironment
from quantom_ips.envs.sample_transformers.identity import IdentitySampleTransformer
from quantom_ips.envs.parsers.multi_numpy_parser import MultiNumpyParser
from quantom_ips.envs.samplers.loits_2d import LOInverseTransformSampler2D
from quantom_ips.trainers.distributed_gan_trainer import DistributedGANTrainer
from quantom_ips.gradient_transport.torch_arar import TorchARAR
from quantom_ips.envs.objectives.mlp_discriminator_v2 import MLPDiscriminatorV2
from quantom_ips.utils import list_registered_modules, make

logger = logging.getLogger("gan_training_workflow")

defaults = [
    {"trainer": "distributed_gan_trainer"},
    {"environment": "dalitz_environment"},
    {"optimizer": "dalitz_optimizer"},
    {"analysis": "distributed_base_analysis"},
    "_self_",
]

@dataclass
class DalitzAnalysisWorkflow:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    trainer: Any = MISSING
    environment: Any = MISSING
    optimizer: Any = MISSING
    analysis: Any = MISSING
    dtype: str = "float32"

cs = ConfigStore.instance()
cs.store(
    name="run_dalitz_analysis", node=DalitzAnalysisWorkflow
)
@hydra.main(version_base=None, config_name="run_dalitz_analysis")
def run(config) -> None:
    logger.debug(f"Registered modules: {list_registered_modules()}")

    # Register the trainer first, because it gives us access to MPI:
    trainer = make(config.trainer.id, config=config.trainer)
    comm = trainer.comm
    devices = trainer.devices
    dtype = trainer.dtype
    # Get the environment:
    env = make(
        config.environment.id,
        config=config.environment,
        mpi_comm=comm,
        devices=devices,
        dtype=dtype,
    )
    # Get the optimizer:
    opt = make(
        config.optimizer.id,
        config=config.optimizer,
        devices=devices,
        dtype=dtype,
    )
    # Get the analysis module:
    analysis = make(
        config.analysis.id,
        config=config.analysis,
        mpi_comm=comm,
        devices=devices,
        dtype=dtype,
    )
    # Run everything:
    trainer.run(opt, env, analysis)

if __name__ == "__main__":
    run()

