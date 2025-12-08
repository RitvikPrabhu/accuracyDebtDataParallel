import logging
import os
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
import horovod.torch as hvd
import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from quantom_ips.analyses.base_analysis import BaseAnalysis
from quantom_ips.analyses.hvd_base_analysis import HVDBaseAnalysis
from quantom_ips.envs.base_environment_v1 import BaseEnvironmentV1
from quantom_ips.envs.hvd_base_environment_v1 import HVDBaseEnvironmentV1
from quantom_ips.envs.parsers.proxy_parser import ProxyParser
from quantom_ips.envs.parsers.hvd_numpy_parser import HVDDistributedNumpyParser
from quantom_ips.envs.sample_transformers.identity import IdentitySampleTransformer
from quantom_ips.envs.samplers.its_2d import InverseTransformSampler2D
from quantom_ips.envs.samplers.loits_2d import LOInverseTransformSampler2D
from quantom_ips.envs.theories.identity import IdentityTheory
from quantom_ips.envs.theories.duke_and_owens_theory import DukeAndOwensTheory
from quantom_ips.optimizers.proxy_optimizer import ProxyOptimizer
from quantom_ips.optimizers.conv2dtranspose_optimizer import Conv2DTransposeOptimizer
from quantom_ips.optimizers.conv2dtranspose_optimizer_v2 import (
    Conv2DTransposeOptimizerV2,
)
from quantom_ips.optimizers.dense1d_optimizer import Dense1DOptimizer
from quantom_ips.trainers.hvd_gan_trainer import HVDGANTrainer
from quantom_ips.envs.objectives.mlp_discriminator import MLPDiscriminator
from quantom_ips.utils import list_registered_modules, make

logger = logging.getLogger("hvd_gan_training_workflow")


defaults = [
    {"trainer": "hvd_gan"},
    {"environment": "hvd_base"},
    {"optimizer": "dense1D"},
    {"analysis": "hvd_base_analysis"},
    {"hydra/run": "scratch_run_dir"},
    "_self_",
]


@dataclass
class HVDGANWorkflow:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    trainer: Any = MISSING
    environment: Any = MISSING
    optimizer: Any = MISSING
    analysis: Any = MISSING
    dtype: str = "float32"


cs = ConfigStore.instance()
cs.store(name="hvd_gan_workflow", node=HVDGANWorkflow)
cs.store(
    group="hydra/run",
    name="scratch_run_dir",
    node={"dir": "/scratch/ritvik_quantom_outputs/${now:%Y-%m-%d}_HVD/${now:%H-%M-%S}"},
)


def get_dtype(dtype_str):
    if "float32" in dtype_str:
        return torch.float32
    elif "float64" in dtype_str:
        return torch.float64
    else:
        raise ValueError("dtype must be either 'float32' or 'float64'")


def init_hvd():
    hvd.init()
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, hvd.size(), local_rank


def set_seeds(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_name="hvd_gan_workflow")
def run(config) -> None:
    rank, world_size, local_rank = init_hvd()
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


if __name__ == "__main__":
    run()
