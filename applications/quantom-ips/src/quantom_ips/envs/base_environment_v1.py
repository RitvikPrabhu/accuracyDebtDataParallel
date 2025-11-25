import torch
import logging
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import List, Any
from quantom_ips.utils.registration import register_with_hydra, make

logger = logging.getLogger(__name__)

defaults = [
    {"parser": "proxy_2D"},
    {"objective": "MLPDiscriminator"},
    {"event_filter": "identity"},
    {"experiment": "identity"},
    {"sampler": "ITS_2D"},
    {"theory": "identity"},
    {"preprocessor": "identity"},
    "_self_",
]


@dataclass
class BaseEnvironmentDefaults:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    id: str = "BaseEnvironmentV1"
    loss_fn: str = "MSE"
    n_samples: int = 10000
    label_noise: float = 0.2
    obj_returns_loss: bool = False


@register_with_hydra(group="environment", defaults=BaseEnvironmentDefaults, name="base")
class BaseEnvironmentV1:
    def __init__(self, config, devices="cpu", dtype=torch.float32):

        self.config = config
        self.devices = devices
        self.dtype = dtype
        self.n_samples = self.config.n_samples

        # Modules:
        self.theory = make(
            self.config.theory.id,
            config=self.config.theory,
            devices=self.devices,
            dtype=self.dtype,
        )
        self.sampler = make(
            self.config.sampler.id,
            config=self.config.sampler,
            devices=self.devices,
            dtype=self.dtype,
        )
        self.experiment = make(
            self.config.experiment.id,
            config=self.config.experiment,
            devices=self.devices,
            dtype=self.dtype,
        )
        self.filter = make(
            self.config.event_filter.id,
            config=self.config.event_filter,
            devices=self.devices,
            dtype=self.dtype,
        )
        self.data_parser = make(
            self.config.parser.id,
            config=self.config.parser,
            devices=self.devices,
            dtype=self.dtype,
        )
        self.objective = make(
            self.config.objective.id,
            config=self.config.objective,
            devices=self.devices,
            dtype=self.dtype,
        )
        self.preprocessor = make(
            self.config.preprocessor.id,
            config=self.config.preprocessor,
            devices=self.devices,
            dtype=self.dtype,
        )

    def step(self, params):
        logger.debug(f"Theory input shape: {params.shape}")

        # Get data from the theory module:
        probabilities, grid_axes, theory_loss = self.theory.forward(params)
        logger.debug(f"x-Section shape: {probabilities.shape}")
        logger.debug(
            f"grid length and shapes: {len(grid_axes)} {[g.shape for g in grid_axes]}"
        )
        logger.debug(f"Theory loss: {theory_loss}")

        # Translate everything to events:
        samples = self.sampler.forward(probabilities, grid_axes, self.n_samples)
        logger.debug(f"Sample shape: {samples.shape}")

        # Pass data through experiment and filter:
        noisy_samples = self.experiment.forward(samples)
        filtered_samples = self.filter.forward(noisy_samples)

        real_samples = self.data_parser.get_samples(filtered_samples.shape)
        real_samples = self.preprocessor.forward(real_samples)
        logger.debug(
            f"Sample shape (real/fake): {real_samples.shape, filtered_samples.shape}"
        )

        gen_loss = self.objective.forward(filtered_samples, real_samples)
        logger.debug(f"Generator loss: {gen_loss}")

        loss = gen_loss | theory_loss
        return loss, filtered_samples

    def get_objective_losses(self, gen_events):

        real_events = self.data_parser.get_samples(gen_events.shape)
        objective_losses = self.objective.forward(
            gen_events, real_events, training=True
        )

        return objective_losses
