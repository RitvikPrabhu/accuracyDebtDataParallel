import logging
from dataclasses import dataclass, field
from typing import Any, List

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from quantom_ips.analyses.ddp_base_analysis import DDPBaseAnalysis
from quantom_ips.envs.ddp_base_environment_v1 import DDPBaseEnvironmentV1
from quantom_ips.envs.parsers.ddp_numpy_parser import DDPDistributedNumpyParser
from quantom_ips.envs.sample_transformers.identity import IdentitySampleTransformer
from quantom_ips.envs.samplers.its_2d import InverseTransformSampler2D
from quantom_ips.envs.samplers.loits_2d import LOInverseTransformSampler2D
from quantom_ips.envs.theories.identity import IdentityTheory
from quantom_ips.optimizers.proxy_optimizer import ProxyOptimizer
from quantom_ips.optimizers.conv2dtranspose_optimizer import Conv2DTransposeOptimizer
from quantom_ips.optimizers.conv2dtranspose_optimizer_v2 import (
    Conv2DTransposeOptimizerV2,
)
from quantom_ips.trainers.ddp_gan_trainer import DDPGANTrainer
from quantom_ips.envs.objectives.mlp_discriminator import MLPDiscriminator
from quantom_ips.utils import list_registered_modules, make

logger = logging.getLogger("ddp_gan_mpi_training_workflow")


defaults = [
    {"trainer": "ddp_gan"},
    {"environment": "ddp_base"},
    {"optimizer": "dense1D"},
    {"analysis": "ddp_base_analysis"},
    "_self_",
]


@dataclass
class DDPGANMPIWorkflow:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    trainer: Any = MISSING
    environment: Any = MISSING
    optimizer: Any = MISSING
    analysis: Any = MISSING
    dtype: str = "float32"


cs = ConfigStore.instance()
cs.store(name="ddp_gan_mpi_workflow", node=DDPGANMPIWorkflow)


def get_dtype(dtype_str):
    if "float32" in dtype_str:
        return torch.float32
    elif "float64" in dtype_str:
        return torch.float64
    else:
        raise ValueError("dtype must be either 'float32' or 'float64'")


def set_seeds(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed_mpi():
    """
    Initialize torch.distributed using the MPI backend. Rank/world_size are
    inferred from the MPI environment. local_rank is derived from rank.
    """
    dist.init_process_group(backend="mpi")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else rank
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


@hydra.main(version_base=None, config_name="ddp_gan_mpi_workflow")
def run(config) -> None:
    rank, world_size, local_rank = init_distributed_mpi()
    set_seeds(42)

    dtype = get_dtype(config.dtype)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    if rank == 0:
        logger.info(f"Registered modules: {list_registered_modules()}")
        logger.info(f"Using device: {device} | world_size={world_size}")

    trainer = make(
        config.trainer.id,
        config=config.trainer,
        device=device,
        dtype=dtype,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    env = make(
        config.environment.id,
        config=config.environment,
        devices=device,
        dtype=dtype,
    )

    opt = make(
        config.optimizer.id,
        config=config.optimizer,
        devices=device,
        dtype=dtype,
    )

    analysis = make(
        config.analysis.id,
        config=config.analysis,
        devices=device,
        dtype=dtype,
    )

    trainer.run(opt, env, analysis)

    dist.destroy_process_group()


if __name__ == "__main__":
    run()
