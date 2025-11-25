from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import os, sys, shlex, subprocess
from .adapter_base import ApplicationAdapter, _ensure_dir, _as_cli_overrides

class QuantomIPSAdapter(ApplicationAdapter):
    def run(self, app_cfg: Dict[str, Any], output_dir: str) -> int:
        module: str = app_cfg["module"]                         
        hydra_overrides: Dict[str, Any] = app_cfg.get("hydra_overrides", {})
        vendor: str | None = app_cfg.get("vendor")              
        launcher: str = app_cfg.get("launcher", "python")       
        nproc: int = int(app_cfg.get("nproc_per_node", 1))
        extra: List[str] = list(app_cfg.get("extra_args", []))  

        _ensure_dir(output_dir)
        cli = [sys.executable, "-m", module] + _as_cli_overrides(hydra_overrides) + [
            f"hydra.run.dir={output_dir}",
            "hydra.output_subdir=.",
        ]
        env = os.environ.copy()
        if vendor:
            vendor_abs = str(Path(vendor).resolve())
            env["PYTHONPATH"] = vendor_abs + os.pathsep + env.get("PYTHONPATH", "")

        if launcher == "torchrun":
            cmd = ["torchrun", f"--nproc_per_node={nproc}", "-m", module] + _as_cli_overrides(hydra_overrides) + [
                f"hydra.run.dir={output_dir}",
                "hydra.output_subdir=.",
            ]
        elif launcher == "horovodrun":
            np = str(app_cfg.get("np", nproc))
            cmd = ["horovodrun", "-np", np, sys.executable, "-m", module] + _as_cli_overrides(hydra_overrides) + [
                f"hydra.run.dir={output_dir}",
                "hydra.output_subdir=.",
            ]
        else:
            cmd = cli

        if extra:
            cmd += list(map(str, extra))

        print("[apps.quantom_ips] running:", " ".join(shlex.quote(x) for x in cmd))
        return subprocess.call(cmd, env=env)
