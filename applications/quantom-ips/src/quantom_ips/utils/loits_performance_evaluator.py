import torch
import numpy as np
import hydra
from dataclasses import dataclass, field
from typing import List, Any
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from quantom_ips.utils.pixelated_densities import DukeAndOwensDensities
from quantom_ips.utils import make
from quantom_ips.envs.theories.duke_and_owens_theory import DukeAndOwensTheory
from quantom_ips.envs.samplers.loits_2d import LOInverseTransformSampler2D
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import logging
import os
import time

logger = logging.getLogger("2D LOITS Performance Evaluator")

@dataclass
class LOITSPerformanceEvaluator:
    n_events: List[Any] = field(default_factory=lambda: [10000,100000,1000000])
    grid_size: int = 100
    batch_size: int = 1
    n_trials: int = 3
    data_loc: str = "${hydra:runtime.output_dir}"
    parallelize_dim_sampling: bool = False
    split_per_batch: bool = False

cs = ConfigStore.instance()
cs.store(name="loits_performance_evaluator", node=LOITSPerformanceEvaluator)

@hydra.main(version_base=None, config_name="loits_performance_evaluator")
def run(config) -> None:
    # Get the torch device first:
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"

    logger.info(f"Determine torch device: {device}")

     # Create directory to store all the results:
    logger.info("Create directory for toy data and plots")
    loc = config.data_loc
    if config.parallelize_dim_sampling:
        loc += "_parallel_dim"
    os.makedirs(loc,exist_ok=True)

    # Get the densities:
    logger.info("Create densities")
    density_generator = DukeAndOwensDensities(devices=device,batch_size=config.batch_size)
    densities = density_generator.get_pixelated_densities(config.grid_size,config.grid_size)

    # Get the theory module:
    logger.info("Load theory and sampler module")
    t_cfg = {
        "a_min": [0.0,0.0],
        "a_max": [1.0,1.0],
        "xsec_min":0.0,
        "xsec_max":1.0,
        "fixed_density_index": -1,
        "average": False,
        "acceptance_epsilon": 1e-11,
    }
    t_cfg = OmegaConf.create(t_cfg)
    theory = make("DukeAndOwensTheory", config=t_cfg, devices=device)

    # Sampler:
    s_cfg = {
        "parallelize_dim_sampling": config.parallelize_dim_sampling,
        "split_per_batch": config.split_per_batch,
        "use_threading": False,
        "vmap_randomness": "different",
        "average": False,
        "a_min": 0.0,
        "a_max": 1.0,
        "log_space": True,
        "n_interpolations_x": 10,
        "n_interpolations_y": 10,
    }
    s_cfg = OmegaConf.create(s_cfg)
    sampler = make(
            "LOInverseTransformSampler2D", config=s_cfg, devices=device
    )

    logger.info(f"Generate toy data for {config.n_trials} trials and {len(config.n_events)} different number of events")

    # Pass through the theory first:
    A, grid_axes, _ = theory.forward(densities)

    t_exec = []
    for nevs in config.n_events:
        t_avg = 0.0
        for _ in range(config.n_trials):
           t_start = time.time()
           sampler.forward(A, grid_axes, nevs)
           t_end = time.time()

           t_avg += float((t_end-t_start)/config.n_trials)

        t_exec.append(t_avg)

    logger.info("Write everything to file")
    np.save(loc+"/t_exec.npy",np.array(t_exec))
    np.save(loc+"/n_events.npy",np.array(config.n_events))

if __name__ == "__main__":
    run()
    