import numpy as np
import torch
import logging
from quantom_ips.utils.registration import register_with_hydra

"""
Load .npy files into memory of a specified rank and then broadcast the data accross the other ranks. 
"""

from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class DistributedNumpyParserDefaults:
    id: str = "DistributedNumpyParser"
    path: list = MISSING
    event_axis: int = 0
    master_rank: int = 0
    data_fraction: float = 0.5
    data_size: int = -1


@register_with_hydra(
    group="environment/parser",
    defaults=DistributedNumpyParserDefaults,
    name="distributed_parser",
)
class DistributedNumpyParser:

    # Initialize:
    # ***********************************
    def __init__(self, config, mpi_comm, devices="cpu", dtype=torch.float32):
        module = "data."
        module_name = "numpy_parser"
        self.full_module_name = module + module_name

        self.devices = devices
        self.data_dtype = dtype
        self.np_dtype = str(dtype).split(".")[1]
        self.data_path = config.path
        self.data_axis = config.event_axis
        self.master_rank = config.master_rank
        self.data_fraction = config.data_fraction
        self.data_size = config.data_size
        self.mpi_comm = mpi_comm

        # Get the data:
        self.data = self.load_data()

    # ***********************************

    # Load .npy file(s):
    # ***********************************
    # Load a single file:
    def load_single_file(self, path_to_file):
        try:
            return np.load(path_to_file).astype(self.np_dtype)
        except Exception:
            logging.exception(
                ">>> " + self.full_module_name + ": File does not exist! <<<"
            )

    # -----------------------------

    # Load multiple files which represent the final data:
    def load_data(self):
        try:
            shared_data = None
            # Load data files into the memory of the master rank:
            if self.mpi_comm.Get_rank() == self.master_rank:

                collected_data = []
                # +++++++++++++++++++++
                for path in self.data_path:
                    collected_data.append(self.load_single_file(path))
                # +++++++++++++++++++++
                shared_data = np.concatenate(collected_data, axis=self.data_axis)

            # Distributed the data accross all other ranks:
            shared_data = self.mpi_comm.bcast(shared_data, root=self.master_rank)

            # Now reduce the data size according to the user specificaions:
            old_size = shared_data.shape[self.data_axis]
            new_size = old_size

            # Absolute data size preceeds data fraction:
            if self.data_size is not None and self.data_size > 0:
                new_size = int(self.data_size)
            # Use the data fraction if no absolute size is provided:
            elif (
                self.data_fraction is not None
                and self.data_fraction > 0
                and self.data_fraction < 1.0
            ):
                new_size = int(self.data_fraction * old_size)

            # If requested size equals full size, keep the full dataset to stay
            # deterministic across backends.
            if new_size >= old_size:
                sampled = shared_data
            else:
                s_idx = np.random.choice(old_size, new_size, replace=False)
                sampled = shared_data[s_idx]

            # Turn data into a torch tensor if requested:
            if self.devices is not None:
                return torch.as_tensor(sampled, dtype=self.data_dtype, device=self.devices)
            return sampled

        except Exception:
            logging.exception(
                ">>> "
                + self.full_module_name
                + ": Please check the provided data path which must be a list. <<<"
            )

            return None

    # ***********************************

    # Draw random samples from the data:
    # ***********************************
    # Translate data to torch tensor, if requested:
    def get_torch_tensor(self, data):
        return torch.as_tensor(data, device=self.devices, dtype=self.data_dtype)

    # -----------------------------

    def get_samples(self, sample_shape=None, to_torch_tensor=False):
        output_data = self.data
        if sample_shape is not None:
            batch_size, particles, n_samples, _ = sample_shape
            total = n_samples * batch_size * particles
            # Use torch RNG when data is a tensor to mirror DDP/HVD behavior.
            if torch.is_tensor(self.data):
                idx = torch.randint(
                    low=0,
                    high=self.data.shape[self.data_axis],
                    size=(total,),
                    device=self.data.device,
                )
                output_data = self.data.index_select(self.data_axis, idx).reshape(sample_shape)
            else:
                idx = np.random.choice(self.data.shape[self.data_axis], total)
                output_data = self.data[idx].reshape(sample_shape)

        if to_torch_tensor and not torch.is_tensor(output_data):
            return self.get_torch_tensor(output_data)
        return output_data

    # ***********************************
