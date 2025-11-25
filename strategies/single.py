from __future__ import annotations
from core.registry import STRATEGIES
from .base import BaseStrategy

@STRATEGIES.register("single_process")
class SingleProcessStrategy(BaseStrategy):
    name = "single_process"
    def setup(self, model, optimizer, backend_cfg):
        # No-op: leave model/optimizer unchanged for single-process training
        return model, optimizer
