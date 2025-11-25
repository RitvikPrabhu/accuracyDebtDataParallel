import logging
from dataclasses import dataclass
from omegaconf import MISSING
from typing import List

import numpy as np
import torch
import torch.distributed as dist

from quantom_ips.utils.registration import register_with_hydra

logger = logging.getLogger(__name__)


@dataclass
class DDPDistributedNumpyParserDefaults:
    id: str = "DDPDistributedNumpyParser"
    path: List[str] = MISSING
    event_axis: int = 0
    master_rank: int = 0
    data_fraction: float = 1.0
    data_size: int = -1


@register_with_hydra(
    group="environment/parser",
    defaults=DDPDistributedNumpyParserDefaults,
    name="ddp_parser",
)
class DDPDistributedNumpyParser:
    """
    Torch DDP-friendly numpy parser. Rank `master_rank` loads the data and
    broadcasts it to all ranks via torch.distributed. Supports subsampling
    via `data_fraction` or `data_size`.
    """

    def __init__(self, config, devices="cpu", dtype=torch.float32):
        if not dist.is_initialized():
            raise RuntimeError(
                "DDPDistributedNumpyParser requires torch.distributed to be initialized."
            )

        self.devices = devices
        self.dtype = dtype
        self.data_path = config.path
        self.data_axis = config.event_axis
        self.master_rank = config.master_rank
        self.data_fraction = config.data_fraction
        self.data_size = config.data_size

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.data = self.load_data()

    def load_single_file(self, path_to_file):
        return np.load(path_to_file)

    def broadcast_data(self, array):
        obj_list = [array]
        dist.broadcast_object_list(obj_list, src=self.master_rank)
        return obj_list[0]

    def load_data(self):
        shared_data = None
        if self.rank == self.master_rank:
            collected = [self.load_single_file(p) for p in self.data_path]
            shared_data = np.concatenate(collected, axis=self.data_axis)
        shared_data = self.broadcast_data(shared_data)

        old_size = shared_data.shape[self.data_axis]
        new_size = old_size
        if self.data_size and self.data_size > 0:
            new_size = int(self.data_size)
        elif (
            self.data_fraction
            and self.data_fraction > 0
            and self.data_fraction < 1.0
        ):
            new_size = int(self.data_fraction * old_size)

        if new_size >= old_size:
            sampled = shared_data
        else:
            idx = np.random.choice(old_size, new_size, replace=False)
            sampled = shared_data[idx]

        tensor_data = torch.as_tensor(sampled, dtype=self.dtype)
        if self.devices is not None:
            tensor_data = tensor_data.to(self.devices)
        return tensor_data

    def get_samples(self, sample_shape=None, to_torch_tensor=True):
        output = self.data
        if sample_shape is not None:
            batch_size, particles, n_samples, _ = sample_shape
            idx = torch.randint(
                low=0,
                high=self.data.shape[self.data_axis],
                size=(n_samples * batch_size * particles,),
                device=self.data.device,
            )
            output = self.data.index_select(self.data_axis, idx).reshape(sample_shape)

        if to_torch_tensor:
            return output
        return output.cpu().numpy()
