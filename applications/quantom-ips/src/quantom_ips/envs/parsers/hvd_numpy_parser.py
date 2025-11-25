import logging
from dataclasses import dataclass
from typing import List

import horovod.torch as hvd
import numpy as np
import torch
from omegaconf import MISSING

from quantom_ips.utils.registration import register_with_hydra

logger = logging.getLogger(__name__)


@dataclass
class HVDDistributedNumpyParserDefaults:
    id: str = "HVDDistributedNumpyParser"
    path: List[str] = MISSING
    event_axis: int = 0
    master_rank: int = 0
    data_fraction: float = 1.0
    data_size: int = -1


@register_with_hydra(
    group="environment/parser",
    defaults=HVDDistributedNumpyParserDefaults,
    name="hvd_parser",
)
class HVDDistributedNumpyParser:
    """
    Horovod-friendly numpy parser: root loads, broadcasts to all ranks.
    Mirrors the distributed/ddp parsers but uses horovod for sync.
    """

    def __init__(self, config, devices="cpu", dtype=torch.float32):
        if not hvd.is_initialized():
            raise RuntimeError("HVDDistributedNumpyParser requires horovod.init().")

        self.devices = devices
        self.dtype = dtype
        self.data_path = config.path
        self.data_axis = config.event_axis
        self.master_rank = config.master_rank
        self.data_fraction = config.data_fraction
        self.data_size = config.data_size

        self.rank = hvd.rank()
        self.world_size = hvd.size()

        self.data = self.load_data()

    def load_single_file(self, path_to_file):
        return np.load(path_to_file)

    def load_data(self):
        shared_data = None
        if self.rank == self.master_rank:
            collected = [self.load_single_file(p) for p in self.data_path]
            shared_data = np.concatenate(collected, axis=self.data_axis)

        # Broadcast via tensors to avoid object-type mismatches.
        if self.rank == self.master_rank:
            data_tensor = torch.as_tensor(shared_data, dtype=self.dtype, device="cpu")
            shape = torch.tensor(data_tensor.shape, dtype=torch.long, device="cpu")
            shape_len = torch.tensor([shape.numel()], dtype=torch.long, device="cpu")
        else:
            data_tensor = None
            shape = None
            shape_len = torch.zeros(1, dtype=torch.long, device="cpu")

        # Broadcast shape length then shape.
        shape_len = hvd.broadcast(shape_len, root_rank=self.master_rank)
        if self.rank != self.master_rank:
            shape = torch.zeros(shape_len.item(), dtype=torch.long, device="cpu")
        shape = hvd.broadcast(shape, root_rank=self.master_rank)
        shape_list = shape.tolist()

        # Broadcast data tensor.
        if self.rank != self.master_rank:
            data_tensor = torch.empty(tuple(shape_list), dtype=self.dtype, device="cpu")
        data_tensor = hvd.broadcast(data_tensor, root_rank=self.master_rank)

        if self.devices is not None:
            data_tensor = data_tensor.to(self.devices)
        tensor_data = data_tensor

        old_size = tensor_data.shape[self.data_axis]
        new_size = old_size
        if self.data_size and self.data_size > 0:
            new_size = int(self.data_size)
        elif self.data_fraction and 0 < self.data_fraction < 1.0:
            new_size = int(self.data_fraction * old_size)

        if new_size >= old_size:
            return tensor_data

        idx = torch.randperm(old_size, device=tensor_data.device)[:new_size]
        sampled = tensor_data.index_select(self.data_axis, idx)

        return sampled

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
