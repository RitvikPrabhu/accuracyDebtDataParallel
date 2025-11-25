from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import os, sys, shlex, subprocess

class ApplicationAdapter:
    def run(self, app_cfg: Dict[str, Any], output_dir: str) -> int:
        raise NotImplementedError

def _ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def _as_cli_overrides(overrides: Dict[str, Any]) -> List[str]:
    out = []
    for k, v in overrides.items():
        if isinstance(v, list):
            out.append(f"{k}=[{','.join(map(str, v))}]")
        elif isinstance(v, bool):
            out.append(f"{k}={str(v).lower()}")
        else:
            out.append(f"{k}={v}")
    return out
