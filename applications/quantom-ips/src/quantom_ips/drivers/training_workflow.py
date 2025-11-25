import logging
import hydra
import torch
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import List, Any, Optional
from hydra.core.config_store import ConfigStore
from quantom_ips.analyses.base_analysis import BaseAnalysis
from quantom_ips.envs.base_environment_v1 import BaseEnvironmentV1
from quantom_ips.envs.parsers.proxy_parser import ProxyParser
from quantom_ips.envs.sample_transformers.identity import IdentitySampleTransformer
from quantom_ips.envs.samplers.its_2d import InverseTransformSampler2D
from quantom_ips.envs.samplers.loits_2d import LOInverseTransformSampler2D
from quantom_ips.envs.theories.identity import IdentityTheory
from quantom_ips.optimizers.proxy_optimizer import ProxyOptimizer
from quantom_ips.optimizers.conv2dtranspose_optimizer import Conv2DTransposeOptimizer
from quantom_ips.trainers.gan_trainer import GANTrainer
from quantom_ips.envs.objectives.mlp_discriminator import MLPDiscriminator
from quantom_ips.utils import list_registered_modules, make

logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)

logger = logging.getLogger("gan_training_workflow")

# Environment Configurations
# BaseEnvironment stored in cfg folder, but still requires import above


defaults = [
    {"trainer": "GAN"},
    {"environment": "base"},
    {"optimizer": "conv2D"},
    {"analysis": "base"},
    "_self_",
]


@dataclass
class GANWorkflow:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    trainer: Any = MISSING
    environment: Any = MISSING
    optimizer: Any = MISSING
    analysis: Any = MISSING
    device: Optional[str] = None
    dtype: str = "float32"


cs = ConfigStore.instance()
cs.store(name="gan_workflow", node=GANWorkflow)


def get_dtype(dtype_str):
    if "float32" in dtype_str:
        return torch.float32
    elif "float64" in dtype_str:
        return torch.float64
    else:
        raise ValueError("dtype must be either 'float32' or 'float64'")


def get_device(device):
    if device is not None:
        return device

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@hydra.main(version_base=None, config_name="gan_workflow")
def run(config) -> None:
    logger.debug(f"Registered modules: {list_registered_modules()}")

    dtype = get_dtype(config.dtype)
    device = get_device(config.device)

    logger.info(f"Using device: {device}")

    trainer = make(config.trainer.id, config=config.trainer, device=device, dtype=dtype)

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
