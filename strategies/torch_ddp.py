from __future__ import annotations
import os, torch, torch.distributed as dist
from core.registry import STRATEGIES
from .base import BaseStrategy

@STRATEGIES.register("torch_ddp")
class TorchDDPStrategy(BaseStrategy):
    name = "torch_ddp"
    def __init__(self, backend="nccl", init_method="env://", find_unused_params=False, broadcast_buffers=True):
        self.backend = backend
        self.init_method = init_method
        self.find_unused_params = find_unused_params
        self.broadcast_buffers = broadcast_buffers

    def setup(self, model, optimizer, backend_cfg):
        if not dist.is_initialized():
            dist.init_process_group(backend=self.backend, init_method=self.init_method)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=self.find_unused_params, broadcast_buffers=self.broadcast_buffers
        )
        return model, optimizer

    def barrier(self):
        if dist.is_initialized(): dist.barrier()
    def is_master(self):
        return (not dist.is_initialized()) or dist.get_rank() == 0
