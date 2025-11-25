import numpy as np
import torch
import logging
from quantom_ips.utils.registration import register_with_hydra
from dataclasses import dataclass, field
from omegaconf import MISSING

logger = logging.getLogger("MultiNumpyParser")

"""
Parsers for multiple numpy data sets...
"""


@dataclass
class MultiNumpyParserDefaults:
    id: str = "MultiNumpyParser"
    paths: list = MISSING
    event_axis: int = 0
    data_fraction: float = 0.5
    data_size: int = -1
    n_randomly_chosen_sets: int = -1
    keep_particle_dim_samples: bool = False
    chosen_data_set_idx: list = field(default_factory=lambda: [])
    keep_data_off_device: bool = False
    use_deterministic_data_idx: bool = False


@register_with_hydra(
    group="environment/parser",
    defaults=MultiNumpyParserDefaults,
    name="multi_numpy_parser",
)
class MultiNumpyParser:

    # Initialize:
    # ***********************************
    def __init__(self, config, mpi_comm=None, devices="cpu", dtype=torch.float32):
        self.devices = devices
        self.data_dtype = dtype
        self.np_dtype = str(dtype).split(".")[1]
        self.data_paths = config.paths
        self.data_axis = config.event_axis
        self.data_fraction = config.data_fraction
        self.data_size = config.data_size
        self.n_randomly_chosen_sets = config.n_randomly_chosen_sets
        self.use_deterministic_data_idx = config.use_deterministic_data_idx
        self.chosen_data_set_idx = config.chosen_data_set_idx
        self.keep_particle_dim_samples = config.keep_particle_dim_samples
        self.chosen_sets = None
        self.keep_data_off_device = config.keep_data_off_device
    
        n_data_sets = len(self.data_paths)

        if len(self.chosen_data_set_idx) > 0:
            self.chosen_sets = self.chosen_data_set_idx
            logger.info(f"Using fixed data set indices {self.chosen_data_set_idx}")
        elif (
            self.n_randomly_chosen_sets > 0
            and self.n_randomly_chosen_sets < n_data_sets
        ):
            self.chosen_sets = np.random.choice(
                n_data_sets, size=self.n_randomly_chosen_sets, replace=False
            )
            logger.info(f"Using randomly chosen sets: {self.chosen_sets}")
        elif (
           self.use_deterministic_data_idx
           and mpi_comm is not None 
        ):
           self.chosen_sets = [
               self.get_data_idx(mpi_comm.Get_rank(),n_data_sets)
           ]
           logger.info(f"Using deterministic sets: {self.chosen_sets}")

        # Get the data:
        self.data = self.load_data()

    # ***********************************
    
    # Get deterministinc data index, if we have multiple ranks:
    # ***********************************
    def get_data_idx(self,current_rank,n_datasets):
        data_idx = current_rank
        accept_rank = False
        while accept_rank == False:
            if data_idx < n_datasets:
              accept_rank = True
            else:
              data_idx -= n_datasets

        return data_idx
    # ***********************************

    # Load .npy file(s):
    # ***********************************
    # Load a single file:
    def load_single_file(self, path_to_file):
        try:
            return np.load(path_to_file).astype(self.np_dtype)
        except Exception:
            logging.exception("File does not exist!")

    # -----------------------------

    # Load multiple files for ONE data set:
    def load_single_data_set(self, current_paths):
        try:

            collected_data = []
            # +++++++++++++++++++++
            for path in current_paths:
                collected_data.append(self.load_single_file(path))
            # +++++++++++++++++++++
            data = np.concatenate(collected_data, axis=self.data_axis)

            # Now reduce the data size according to the user specificaions:
            old_size = data.shape[self.data_axis]
            new_size = old_size

            # Absolute data size preceeds data fraction:
            if self.data_size is not None and self.data_size > 0:
                new_size = int(self.data_size)
            # Use the data fraction if not absolute size is provided:
            if (
                (self.data_size is None or self.data_size < 0)
                and self.data_fraction is not None
                and self.data_fraction > 0
                and self.data_fraction <= 1.0
            ):
                new_size = int(self.data_fraction * old_size)

            # Take samples from the data, according to the data fraction / size:
            s_idx = np.random.choice(old_size, new_size)

            # Turn data into a torch tensor if requested:
            if self.devices is not None and self.keep_data_off_device is False:
                return torch.as_tensor(
                    data[s_idx], dtype=self.data_dtype, device=self.devices
                )
            return data[s_idx]

        except Exception:
            logger.exception("Please check the provided data path which must be a list")

            return None

    # -----------------------------

    # Now load the data:
    def load_data(self):
        data_sets = []
        # ++++++++++++++++++
        for i in range(len(self.data_paths)):
            paths = self.data_paths[i]
            if self.chosen_sets is not None:
                if i in self.chosen_sets:
                    data_sets.append(self.load_single_data_set(paths))
            else:
                data_sets.append(self.load_single_data_set(paths))
        # ++++++++++++++++++

        if len(data_sets) > 1:
            if self.devices is not None and self.keep_data_off_device is False:
                print("TO TORCH TENSOR")
                return torch.stack(data_sets, 0)
            return np.stack(data_sets, 0)
        return data_sets[0]

    # ***********************************

    # Draw random samples from the data:
    # ***********************************
    # Translate data to torch tensor, if requested:
    def get_torch_tensor(self, data):
        return torch.as_tensor(data, device=self.devices, dtype=self.data_dtype)

    # -----------------------------

    def get_samples_from_single_set(self, single_set, sample_shape, to_torch_tensor):
        output_data = single_set
        if sample_shape is not None:
            batch_size, particles, n_samples, _ = sample_shape
            # Get the new shape from the provided shape:
            new_shape = n_samples * batch_size
            if self.keep_particle_dim_samples:
                new_shape *= particles

            idx = np.random.choice(
                single_set.shape[self.data_axis], new_shape
            )
            
            output_data = single_set[idx].reshape((batch_size,1,n_samples,sample_shape[-1]))
            if self.keep_particle_dim_samples:
              output_data = single_set[idx].reshape(sample_shape)

        if to_torch_tensor:
            return self.get_torch_tensor(output_data)
        return output_data

    # -----------------------------

    def get_samples(self, sample_shape=None, to_torch_tensor=False):
        output_data = self.data
        if sample_shape is not None:
            if len(self.data.shape) == 3:
                sampled_sets = []
                # +++++++++++++++++++++
                for i in range(self.data.shape[0]):
                    sampled_sets.append(
                        self.get_samples_from_single_set(
                            self.data[i],
                            sample_shape=sample_shape,
                            to_torch_tensor=to_torch_tensor,
                        )
                    )
                # +++++++++++++++++++++
                if to_torch_tensor:
                    output_data = torch.concat(sampled_sets,1)
                else:
                    output_data = np.concatenate(sampled_sets,1)

            if len(self.data.shape) == 2:
                output_data = self.get_samples_from_single_set(
                    self.data,
                    sample_shape=sample_shape,
                    to_torch_tensor=to_torch_tensor,
                )

        return output_data

    # ***********************************
