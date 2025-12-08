import logging
from dataclasses import dataclass, field
from typing import Any, List

import horovod.torch as hvd
import torch
from omegaconf import MISSING

from quantom_ips.utils.log_sad_score import get_logSAD_from_events
from quantom_ips.utils.registration import make, register_with_hydra

logger = logging.getLogger(__name__)


defaults = [
    {"parser": "hvd_parser"},
    {"objective": "MLPDiscriminator"},
    {"event_filter": "identity"},
    {"experiment": "identity"},
    {"sampler": "LOITS_2D"},
    {"theory": "duke_and_owens"},
    {"preprocessor": "identity"},
    "_self_",
]


@dataclass
class HVDBaseEnvironmentDefaults:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    id: str = "HVDBaseEnvironmentV1"
    loss_fn: str = "MSE"
    n_samples: int = 100000
    label_noise: float = 0.0
    n_log_sad_bins: int = 100


@register_with_hydra(group="environment", defaults=HVDBaseEnvironmentDefaults, name="hvd_base")
class HVDBaseEnvironmentV1:
    """
    Horovod-friendly analogue of DistributedBaseEnvironmentV1.
    """

    def __init__(self, config, devices="cpu", dtype=torch.float32):
        if not hvd.is_initialized():
            raise RuntimeError("HVDBaseEnvironmentV1 requires horovod.init().")

        self.config = config
        self.devices = devices
        self.dtype = dtype
        self.n_samples = self.config.n_samples

        self.data_parser = make(
            self.config.parser.id,
            config=self.config.parser,
            devices=self.devices,
            dtype=self.dtype,
        )
        self.preprocessor = make(
            self.config.preprocessor.id,
            config=self.config.preprocessor,
            devices=self.devices,
            dtype=self.dtype,
        )
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
        self.objective = make(
            self.config.objective.id,
            config=self.config.objective,
            devices=self.devices,
            dtype=self.dtype,
        )

        self.data_parser.data = self.preprocessor.forward(self.data_parser.data)

    def step(self, params):
        probabilities, grid_axes, theory_loss = self.theory.forward(params)
        samples = self.sampler.forward(probabilities, grid_axes, self.n_samples)

        noisy_samples = self.experiment.forward(samples)
        filtered_samples = self.filter.forward(noisy_samples)

        gen_loss = self.objective.forward(filtered_samples, None)

        loss = gen_loss | {"theory_loss": theory_loss}
        return loss, filtered_samples

    def get_objective_losses(self, gen_events):
        real_events = self.data_parser.get_samples(
            gen_events.size(), to_torch_tensor=True
        )
        objective_losses = self.objective.forward(
            gen_events.detach(), real_events, training=True
        )
        log_sad = get_logSAD_from_events(
            real_events.detach().flatten(0, 2).cpu().numpy(),
            gen_events.detach().flatten(0, 2).cpu().numpy(),
            self.config.n_log_sad_bins,
        )

        losses = objective_losses | {"log_sad": log_sad}
        return losses
