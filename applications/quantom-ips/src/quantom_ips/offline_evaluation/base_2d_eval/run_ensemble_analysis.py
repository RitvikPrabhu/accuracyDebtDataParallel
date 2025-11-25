import hydra
import torch
import os
import logging
import numpy as np
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import List, Any, Optional
from hydra.core.config_store import ConfigStore
from quantom_ips.offline_evaluation.base_2d_eval.ensemble_analysis import (
    collect_individual_predictions,
    get_ensemble_predictions,
    get_metrics
)
from quantom_ips.optimizers.conv2dtranspose_optimizer_v2 import (
    Conv2DTransposeOptimizerV2,
)
from quantom_ips.utils import list_registered_modules, make
from quantom_ips.utils.pixelated_densities import ProxyApplication2DDensities

logger = logging.getLogger("Run Ensemble Analysis")

defaults = [
    {"optimizer": "conv2D_v2"},
    "_self_",
]


@dataclass
class RunEnsembleAnalysis:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    optimizer: Any = MISSING
    dtype: str = "float32"
    devices: str = "cpu"
    batch_size: int = 5000
    n_ranks: int = 60
    ensemble_size: int = 1
    t_start: int = 2000
    t_step: int = 500
    i_max: int = 10000
    model_loc : str = "../../../../../testrun_2d_proxyapp"
    result_loc: str = "results_arar"


cs = ConfigStore.instance()
cs.store(name="run_ensemble_analysis", node=RunEnsembleAnalysis)
def get_dtype(dtype_str):
    if "float32" in dtype_str:
        return torch.float32
    elif "float64" in dtype_str:
        return torch.float64
    else:
        raise ValueError("dtype must be either 'float32' or 'float64'")


@hydra.main(version_base=None, config_name="run_ensemble_analysis")
def run(config) -> None:
    logger.debug(f"Registered modules: {list_registered_modules()}")

    dtype = get_dtype(config.dtype)
    devices = config.devices
    
    logger.debug("Set up optimizer and true densities")
    opt = make(
        config.optimizer.id,
        config=config.optimizer,
        devices=devices,
        dtype=dtype,
    )

    # Create true densities:
    density_generator = ProxyApplication2DDensities(
            devices=devices, batch_size=1, n_targets=1
    )
    optimizer_sizes = config.optimizer.layers.Upsample.config.size
    true_densities = density_generator.get_pixelated_density(optimizer_sizes[0],optimizer_sizes[1])
    true_densities = torch.squeeze(true_densities).detach().cpu().numpy()
    
    logger.debug("Run ensemble analysis")
    individual_predictions, times = collect_individual_predictions(
        model_loc = config.model_loc,
        ensemble_size = config.ensemble_size,
        n_ranks = config.n_ranks,
        batch_size = config.batch_size,
        t_start = config.t_start,
        t_step = config.t_step,
        i_max = config.i_max,
        opt=opt
    )
    
    ensemble_mean, ensemble_std = get_ensemble_predictions(individual_predictions)
    
    logger.debug("Write everything to file")
    results= get_metrics(ensemble_mean,ensemble_std,true_densities)
    results['times'] = times

    os.makedirs(config.result_loc,exist_ok=True)
    for key in results:
        np.save(config.result_loc+"/"+key+".npy",results[key])



if __name__ == "__main__":
    run()


