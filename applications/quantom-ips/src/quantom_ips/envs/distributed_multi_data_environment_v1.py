import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from quantom_ips.utils.pixelated_densities import ProxyApplication2DDensities
from dataclasses import dataclass, field
from omegaconf import MISSING
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
    {"theory": "duke_and_owens"},
    {"preprocessor": "identity"},
    "_self_",
]


@dataclass
class DistributedMultiDataEnvironmentDefaults:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    id: str = "DistributedMultiDataEnvironmentV1"
    loss_fn: str = "MSE"
    n_samples: int = 10000
    label_noise: float = 0.0
    n_log_sad_bins: int = 100
    use_auto_scaling: bool = False


@register_with_hydra(
    group="environment",
    defaults=DistributedMultiDataEnvironmentDefaults,
    name="distributed_multi_data",
)
class DistributedMultiDataEnvironmentV1:

    # Initialize:
    # *****************************************************
    def __init__(self, config, mpi_comm, devices="cpu", dtype=torch.float32):
        self.config = config
        self.devices = devices
        self.dtype = dtype
        self.n_samples = self.config.n_samples

        # self.comm = mpi_comm
        # self.density_generator = ProxyApplication2DDensities(
        #     devices=devices, batch_size=1, n_targets=2
        # )
        
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
        # Determine the minimum and maximum of the data:

        # Let the enviornment determine the range for the min/max scaler --> Only if we have a single data set  
        if self.config.use_auto_scaling and len(self.data_parser.data.size()) == 2:
            if self.config.objective.use_log:
              data_min = torch.min(torch.log(self.data_parser.data+1e-7),0).values
              data_max = torch.max(torch.log(self.data_parser.data+1e-7),0).values
            else:
              data_min = torch.min(self.data_parser.data,0).values
              data_max = torch.max(self.data_parser.data,0).values
            
            self.config.objective.observed_feature_range_min = data_min.detach().cpu().tolist()
            self.config.objective.observed_feature_range_max = data_max.detach().cpu().tolist()

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
        
        # current_rank = self.comm.Get_rank()
        # fig,ax = plt.subplots(2,2)
        # fig.suptitle(f"chosen set: {self.chosen_sets} for rank: {current_rank}")

        # ax[0,0].hist(real_events.detach().cpu().numpy()[0][0][:,0],100)
        # ax[0,1].hist(real_events.detach().cpu().numpy()[0][0][:,1],100)

        # ax[1,0].hist(gen_events.detach().cpu().numpy()[0][0][:,0],100)
        # ax[1,1].hist(gen_events.detach().cpu().numpy()[0][0][:,1],100)

        
        # plt.show()


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
