import torch
import logging
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import List, Any
from quantom_ips.utils.registration import register_with_hydra, make
from quantom_ips.utils.log_sad_score import get_logSAD_from_events

logger = logging.getLogger(__name__)

defaults = [
    {"parser": "distributed_parser"},
    {"objective": "MLPDiscriminator"},
    {"event_filter": "identity"},
    {"experiment": "identity"},
    {"sampler": "LOITS_2D"},
    {"theory": "duke_and_owens"},
    {"preprocessor": "identity"},
    "_self_",
]


@dataclass
class DistributedBaseEnvironmentDefaults:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    id: str = "DistributedBaseEnvironmentV1"
    loss_fn: str = "MSE"
    n_samples: int = 100000
    label_noise: float = 0.0
    n_log_sad_bins: int = 100


@register_with_hydra(
    group="environment",
    defaults=DistributedBaseEnvironmentDefaults,
    name="distributed_base",
)
class DistributedBaseEnvironmentV1:

    # Initialize:
    # *****************************************************
    def __init__(self, config, mpi_comm, devices="cuda", dtype=torch.float32):
        self.config = config
        self.devices = devices
        self.dtype = dtype
        self.n_samples = self.config.n_samples

        # Modules:
        self.data_parser = make(
            self.config.parser.id,
            config=self.config.parser,
            mpi_comm=mpi_comm,
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
        # Apply the preprocessing on the entire data, so that we do not have to do it later:
        self.data_parser.data = self.preprocessor.forward(self.data_parser.data)

    # *****************************************************

    # Run a single environment step:
    # *****************************************************
    def step(self, params):
        # Get data from the theory module:
        probabilities, grid_axes, theory_loss = self.theory.forward(params)
        # Translate everything to events:
        samples = self.sampler.forward(probabilities, grid_axes, self.n_samples)

        # Pass data through experiment and filter:
        noisy_samples = self.experiment.forward(samples)
        filtered_samples = self.filter.forward(noisy_samples)

        # Compute optimizer loss:
        gen_loss = self.objective.forward(filtered_samples, None)

        loss = gen_loss | {"theory_loss": theory_loss}
        return loss, filtered_samples

    # *****************************************************

    # Get the objective score:
    # *****************************************************
    def get_objective_losses(self, gen_events):
        # Get samples from the real data:
        real_events = self.data_parser.get_samples(
            gen_events.size(), to_torch_tensor=True
        )
        objective_losses = self.objective.forward(
            gen_events.detach(), real_events, training=True
        )
        # Compute the log[SAD] score for real / generated events, just to have another metric:
        log_sad = get_logSAD_from_events(
            real_events.detach().flatten(0, 2).cpu().numpy(),
            gen_events.detach().flatten(0, 2).cpu().numpy(),
            self.config.n_log_sad_bins,
        )
        # Delete what we do not need:
        del real_events
        del gen_events

        # Not ideal, but we add the log_sad score to the objective dictionary, just so that the reponse from get_objective_loses is consistent throughout all modules:
        losses = objective_losses | {"log_sad": log_sad}
        return losses

    # *****************************************************
