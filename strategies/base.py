from __future__ import annotations
from typing import Any, Dict

class BaseStrategy:
    name = "base"
    def setup(self, model, optimizer, backend_cfg: Dict[str, Any]):
        raise NotImplementedError
    def barrier(self):
        pass
    def broadcast(self, obj):
        return obj
    def allreduce(self, tensor):
        return tensor
    def is_master(self) -> bool:
        return True
