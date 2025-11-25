# applications/quantom_ips_adapter.py
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import os, sys, shlex, subprocess

def _ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def _as_cli_overrides(overrides: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for k, v in (overrides or {}).items():
        if isinstance(v, list):
            out.append(f"{k}=[{','.join(map(str, v))}]")
        elif isinstance(v, bool):
            out.append(f"{k}={str(v).lower()}")
        else:
            out.append(f"{k}={v}")
    return out

class QuantomIPSAdapter:
    """
    Runs a Hydra module inside the quantom-ips package.
    Supports launcher: 'python' (default), 'torchrun', or 'horovodrun'.
    Optional 'vendor': path to prepend to PYTHONPATH (e.g., 'applications').
    """
    def run(self, app_cfg: Dict[str, Any], output_dir: str) -> int:
        module: str = app_cfg["module"]                      # e.g. quantom_ips.drivers.training_workflow
        hydra_overrides: Dict[str, Any] = app_cfg.get("hydra_overrides", {})
        launcher: str = app_cfg.get("launcher", "python")   # python | torchrun | horovodrun
        nproc: int = int(app_cfg.get("nproc_per_node"))
        vendor: str | None = app_cfg.get("vendor")          # e.g. applications
        extra: List[str] = list(app_cfg.get("extra_args", []))

        _ensure_dir(output_dir)
        env = os.environ.copy()
        if vendor:
            env["PYTHONPATH"] = str(Path(vendor).resolve()) + os.pathsep + env.get("PYTHONPATH", "")

        base_args = _as_cli_overrides(hydra_overrides) + [
            f"hydra.run.dir={output_dir}",
            "hydra.output_subdir=.",
        ]

        if launcher == "torchrun":
            cmd = ["torchrun", f"--nproc_per_node={nproc}", "-m", module] + base_args
        elif launcher == "horovodrun":
            np = str(app_cfg.get("np", nproc))
            cmd = ["horovodrun", "-np", np, sys.executable, "-m", module] + base_args
        else:
            cmd = [sys.executable, "-m", module] + base_args

        if extra:
            cmd += list(map(str, extra))

        print("[applications.quantom_ips] running:", " ".join(shlex.quote(x) for x in cmd))
        return subprocess.call(cmd, env=env)
