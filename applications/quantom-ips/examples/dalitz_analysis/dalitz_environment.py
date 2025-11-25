import torch
import logging
from dataclasses import dataclass, field
from typing import List, Any
from quantom_ips.utils.registration import register_with_hydra, make
from quantom_ips.utils.log_sad_score import get_logSAD_from_events


logger = logging.getLogger(__name__)

defaults = [
    {"parser": "multi_numpy_parser"},
    {"objective": "MLPDiscriminatorV2"},
    {"event_filter": "identity"},
    {"experiment": "identity"},
    {"sampler": "LOITS_2D"},
    {"theory": "dalitz_theory"},
    {"preprocessor": "identity"},
    "_self_",
]

@dataclass
class DalitzEnvironmentDefaults:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    id: str = "DalitzEnvironment"
    loss_fn: str = "MSE"
    n_samples: int = 10000
    n_log_sad_bins: int = 100
    mode: str = "baseline"

@register_with_hydra(
    group="environment",
    defaults=DalitzEnvironmentDefaults,
    name="dalitz_environment",
)
class DalitzEnvironment:

    # Initialize:
    # *****************************************************
    def __init__(self, config, mpi_comm, devices="cpu", dtype=torch.float32):
        self.config = config
        self.devices = devices
        self.dtype = dtype
        self.n_samples = self.config.n_samples
        self.mode = config.mode

        if config.mode.lower() == "multidata" and mpi_comm is not None:
            self.config.parser.use_deterministic_data_idx = True

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

        # Apply the preprocessing on the entire data, so that we do not have to do it later:
        self.data_parser.data = self.preprocessor.forward(self.data_parser.data)
        # Get the indices that have been used to select the subsets:
        self.chosen_sets = self.data_parser.chosen_sets

        if self.mode.lower() == "baseline": 
            self.data_min = torch.min(self.data_parser.data,1).values.flatten(0,1)
            self.data_max = torch.max(self.data_parser.data,1).values.flatten(0,1)
        if self.mode.lower() == "multidata": 
            self.data_min = torch.min(self.data_parser.data,0).values
            self.data_max = torch.max(self.data_parser.data,0).values
            
        self.config.objective.observed_feature_range_min = self.data_min.detach().cpu().tolist()
        self.config.objective.observed_feature_range_max = self.data_max.detach().cpu().tolist()
        
        self.objective = make(
            self.config.objective.id,
            config=self.config.objective,
            devices=self.devices,
            dtype=self.dtype,
        )

    # *****************************************************

    # Run a single environment step:
    # *****************************************************
    def step(self, params):
        #true_densities = self.density_generator.get_pixelated_density(params.size()[2],params.size()[3])
        # Get data from the theory module:
        probabilities, grid_axes, theory_loss = self.theory.forward(params)
        # Translate everything to events:
        samples = self.sampler.forward(
            probabilities, grid_axes, self.n_samples, chosen_targets=self.chosen_sets
        )

        # Pass data through experiment and filter:
        noisy_samples = self.experiment.forward(samples)
        filtered_samples = self.filter.forward(noisy_samples)

        # Compute optimizer loss:
        if self.mode.lower() == "baseline" and len(samples.size()) == 4:
           gen_loss = self.objective.forward(
               filtered_samples.transpose(1,2).flatten(2,3).flatten(0,1), 
               None
           )

        if self.mode.lower() == "multidata":
            gen_loss = self.objective.forward(
               filtered_samples, 
               None
           )

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

        if self.mode.lower() == "baseline":
            real_events = real_events.transpose(1,2).flatten(2,3).flatten(0,1)
            gen_events = gen_events.transpose(1,2).flatten(2,3).flatten(0,1)
        
        objective_losses = self.objective.forward(
            gen_events.detach(), real_events, training=True
        )

        # Compute the log[SAD] score for real / generated events, just to have another metric:
        if self.mode.lower() == "baseline":
          log_sad = -1.0
        else:
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