import os, yaml
from pathlib import Path
from typing import Any, Dict

def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    includes = cfg.pop("_include_", [])
    merged: Dict[str, Any] = {}
    for inc in includes:
        incp = (path.parent / inc) if not os.path.isabs(inc) else Path(inc)
        merged = deep_update(merged, load_config(incp))
    return deep_update(merged, cfg)

def deep_update(base: Dict, upd: Dict):
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base
