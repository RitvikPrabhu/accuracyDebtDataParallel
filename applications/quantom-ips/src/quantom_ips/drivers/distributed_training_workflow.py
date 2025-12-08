import mpi4py

mpi4py.rc.thread_level = "serialized"
from mpi4py import MPI
import logging
import hydra
import torch
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import List, Any, Optional
import numpy as np
from hydra.core.config_store import ConfigStore
from quantom_ips.analyses.distributed_base_analysis import DistributedBaseAnalysis
from quantom_ips.envs.distributed_base_environment_v1 import (
    DistributedBaseEnvironmentV1,
)
from quantom_ips.envs.parsers.distributed_numpy_parser import DistributedNumpyParser
from quantom_ips.envs.sample_transformers.identity import IdentitySampleTransformer
from quantom_ips.envs.samplers.loits_2d import LOInverseTransformSampler2D
from quantom_ips.envs.samplers.loits_1d import LOInverseTransformSampler1D
from quantom_ips.envs.theories.identity import IdentityTheory
from quantom_ips.envs.theories.proxy2d_theory import Proxy2DTheory
from quantom_ips.envs.theories.duke_and_owens_theory import DukeAndOwensTheory
from quantom_ips.optimizers.proxy_optimizer import ProxyOptimizer
from quantom_ips.optimizers.conv2dtranspose_optimizer_v2 import (
    Conv2DTransposeOptimizerV2,
)
from quantom_ips.optimizers.dense1d_optimizer import Dense1DOptimizer
from quantom_ips.trainers.distributed_gan_trainer import DistributedGANTrainer
from quantom_ips.gradient_transport.torch_arar import TorchARAR
from quantom_ips.envs.objectives.mlp_discriminator import MLPDiscriminator
from quantom_ips.utils import list_registered_modules, make

logger = logging.getLogger("gan_training_workflow")

defaults = [
    {"trainer": "distributed_gan_trainer"},
    {"environment": "distributed_base"},
    {"optimizer": "dense1D"},
    {"analysis": "distributed_base_analysis"},
    {"hydra/run": "scratch_run_dir"},
    "_self_",
]


@dataclass
class DistributedGANWorkflow:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    trainer: Any = MISSING
    environment: Any = MISSING
    optimizer: Any = MISSING
    analysis: Any = MISSING
    dtype: str = "float32"


cs = ConfigStore.instance()
cs.store(name="distributed_gan_workflow", node=DistributedGANWorkflow)
cs.store(
    group="hydra/run",
    name="scratch_run_dir",
    node={"dir": "/scratch/ritvik_quantom_outputs/${now:%Y-%m-%d}_ARAR/${now:%H-%M-%S}"},
)


@hydra.main(version_base=None, config_name="distributed_gan_workflow")
def run(config) -> None:
    logger.debug(f"Registered modules: {list_registered_modules()}")

    # Register the trainer first, because it gives us access to MPI:
    trainer = make(config.trainer.id, config=config.trainer)
    # Seed all RNGs identically across MPI ranks for reproducibility.
    import random

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
