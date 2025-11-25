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
from quantom_ips.envs.distributed_multi_data_environment_v1 import (
    DistributedMultiDataEnvironmentV1,
)
from quantom_ips.envs.sample_transformers.identity import IdentitySampleTransformer
from quantom_ips.envs.parsers.multi_numpy_parser import MultiNumpyParser
from quantom_ips.envs.samplers.loits_2d import LOInverseTransformSampler2D
from quantom_ips.envs.theories.duke_and_owens_theory import DukeAndOwensTheory
from quantom_ips.envs.theories.identity import IdentityTheory
from quantom_ips.envs.theories.proxy2d_theory import Proxy2DTheory
from quantom_ips.optimizers.proxy_optimizer import ProxyOptimizer
from quantom_ips.optimizers.conv2dtranspose_optimizer_v2 import (
    Conv2DTransposeOptimizerV2,
)
from quantom_ips.optimizers.conv2dtranspose_optimizer import (
    Conv2DTransposeOptimizer,
)
from quantom_ips.optimizers.dense1d_optimizer import Dense1DOptimizer
from quantom_ips.trainers.distributed_gan_trainer import DistributedGANTrainer
from quantom_ips.trainers.distributed_wgan_trainer_v2 import DistributedWGANTrainerV2
from quantom_ips.trainers.distributed_gan_trainer_v2 import DistributedGANTrainerV2
from quantom_ips.gradient_transport.torch_arar import TorchARAR
from quantom_ips.gradient_transport.torch_arar_v2 import TorchARARV2
from quantom_ips.envs.objectives.mlp_discriminator_v2 import MLPDiscriminatorV2
from quantom_ips.envs.objectives.sa_discriminator import SADiscriminator
from quantom_ips.envs.objectives.resnet_discriminator import ResNetDiscriminator
from quantom_ips.utils import list_registered_modules, make

logger = logging.getLogger("gan_training_workflow")

defaults = [
    {"trainer": "distributed_wgan_trainer_v2"},
    {"environment": "distributed_multi_data"},
    {"optimizer": "conv2D_v2"},
    {"analysis": "distributed_base_analysis"},
    "_self_",
]


@dataclass
class DistributedMultiDataGANWorkflowV2:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    trainer: Any = MISSING
    environment: Any = MISSING
    optimizer: Any = MISSING
    analysis: Any = MISSING
    dtype: str = "float32"


cs = ConfigStore.instance()
cs.store(
    name="distributed_multi_data_gan_workflow_v2", node=DistributedMultiDataGANWorkflowV2
)


@hydra.main(version_base=None, config_name="distributed_multi_data_gan_workflow_v2")
def run(config) -> None:
    logger.debug(f"Registered modules: {list_registered_modules()}")

    # Register the trainer first, because it gives us access to MPI:
    trainer = make(config.trainer.id, config=config.trainer)
    comm = trainer.comm
    devices = trainer.devices
    dtype = trainer.dtype

    # This is nasty, but I just want to test things:
    if config.trainer.n_data_sets > 1:
       config.environment.parser.chosen_data_set_idx = [trainer.pred_rank]

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
