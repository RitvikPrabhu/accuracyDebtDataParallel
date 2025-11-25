from __future__ import annotations
from typing import Callable, Dict

class _Registry:
    def __init__(self):
        self._map: Dict[str, Callable] = {}
    def register(self, name: str):
        def deco(fn_or_cls):
            if name in self._map:
                raise KeyError(f"Registry duplicate: {name}")
            self._map[name] = fn_or_cls
            return fn_or_cls
        return deco
    def get(self, name: str):
        if name not in self._map:
            raise KeyError(f"Unknown component: {name}. Available: {list(self._map)}")
        return self._map[name]

DATASETS = _Registry()
MODELS = _Registry()
STRATEGIES = _Registry()
BACKENDS = _Registry()
