import hydra
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import List, Any, Optional
from hydra.core.config_store import ConfigStore
from quantom_ips.envs.distributed_multi_data_environment_v1 import DistributedMultiDataEnvironmentV1
from quantom_ips.envs.base_environment_v1 import BaseEnvironmentV1
from quantom_ips.envs.samplers.its_2d import InverseTransformSampler2D
from quantom_ips.envs.samplers.loits_2d import LOInverseTransformSampler2D
from quantom_ips.envs.objectives.mlp_discriminator import MLPDiscriminator
from quantom_ips.envs.objectives.mlp_discriminator_v2 import MLPDiscriminatorV2
from quantom_ips.envs.theories.identity import IdentityTheory
from quantom_ips.envs.theories.proxy2d_theory import Proxy2DTheory
from quantom_ips.envs.theories.proxy2d_theory_v2 import Proxy2DTheoryV2
from quantom_ips.envs.parsers.proxy_parser import ProxyParser
from quantom_ips.envs.parsers.multi_numpy_parser import MultiNumpyParser
from quantom_ips.envs.sample_transformers.identity import IdentitySampleTransformer
from quantom_ips.optimizers.conv2dtranspose_optimizer_v2 import (
    Conv2DTransposeOptimizerV2,
)
from quantom_ips.utils import list_registered_modules, make
from quantom_ips.offline_evaluation.base_2d_eval.eval_tools import analyze_trained_generators
from quantom_ips.utils.pixelated_densities import ProxyApplication2DDensities
from quantom_ips.offline_evaluation.duke_and_owens_eval.grad_viz import show_gradients

plt.rcParams.update({'font.size':20})
plt.tight_layout()
logger = logging.getLogger("Run Eval")

defaults = [
    {"environment": "distributed_multi_data"},
    {"optimizer": "conv2D_v2"},
    "_self_",
]


@dataclass
class RunEval:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    environment: Any = MISSING
    optimizer: Any = MISSING
    dtype: str = "float32"
    devices: str = "cpu"
    batch_size: int = 5000
    ensemble_size: int = 1
    n_ranks: int = 8
    n_bins: int = 100
    pdf_cmap: str = 'gist_ncar'
    pdf_delta_r: float = 0.2
    movie_duration: float = 0.1
    data_locs: list = MISSING
    data_names: list = MISSING
    logdir: str = "${hydra:runtime.output_dir}"
    n_densities: int = 3
    
cs = ConfigStore.instance()
cs.store(name="run_eval", node=RunEval)
def get_dtype(dtype_str):
    if "float32" in dtype_str:
        return torch.float32
    elif "float64" in dtype_str:
        return torch.float64
    else:
        raise ValueError("dtype must be either 'float32' or 'float64'")


@hydra.main(version_base=None, config_name="run_eval")
def run(config) -> None:
    logger.debug(f"Registered modules: {list_registered_modules()}")

    dtype = get_dtype(config.dtype)
    devices = config.devices

    env = make(
        config.environment.id,
        mpi_comm=None,
        config=config.environment,
        devices=devices,
        dtype=dtype,
    )

    opt = make(
        config.optimizer.id,
        config=config.optimizer,
        devices=devices,
        dtype=dtype,
    )

    # Create true densities:
    density_generator = ProxyApplication2DDensities(
            devices=devices, batch_size=1, n_targets=config.n_densities
    )
    optimizer_sizes = config.optimizer.layers.Upsample.config.size
    true_densities = density_generator.get_pixelated_density(optimizer_sizes[0],optimizer_sizes[1])
    true_densities = torch.squeeze(true_densities).detach().cpu().numpy()
    
    for loc, name in zip(config.data_locs,config.data_names):
        analyze_trained_generators(loc,config.ensemble_size,config.n_ranks,config.batch_size,true_densities,opt,env,config.n_bins,config.pdf_cmap,config.pdf_delta_r,config.movie_duration,config.logdir)
    #    grads = show_gradients(loc,config.n_ranks,config.logdir)

if __name__ == "__main__":
    run()
    

