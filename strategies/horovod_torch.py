from __future__ import annotations
import torch
import horovod.torch as hvd
from core.registry import STRATEGIES
from .base import BaseStrategy

@STRATEGIES.register("horovod_torch")
class HorovodTorch(BaseStrategy):
    name = "horovod_torch"
    def __init__(self, allreduce="average", fusion_threshold_mb=64, cycle_time_ms=5):
        self.mode = allreduce
        self.fusion_threshold_mb = fusion_threshold_mb
        self.cycle_time_ms = cycle_time_ms

    def setup(self, model, optimizer, backend_cfg):
        hvd.init()
        if torch.cuda.is_available():
            torch.cuda.set_device(hvd.local_rank())
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        return model, optimizer

    def is_master(self):
        return hvd.rank() == 0
