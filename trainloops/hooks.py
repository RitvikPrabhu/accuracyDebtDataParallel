from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

class Hook:
    def on_train_start(self, state: Dict[str, Any]): ...
    def on_epoch_start(self, state: Dict[str, Any]): ...
    def on_batch_end(self, state: Dict[str, Any]): ...
    def on_epoch_end(self, state: Dict[str, Any]): ...
    def on_train_end(self, state: Dict[str, Any]): ...

@dataclass
class CheckpointHook(Hook):
    outdir: str
    every_n_epochs: int = 1
    def on_epoch_end(self, state: Dict[str, Any]):
        epoch = state["epoch"]
        if epoch % self.every_n_epochs != 0:
            return
        Path(self.outdir).mkdir(parents=True, exist_ok=True)
        path = Path(self.outdir) / f"epoch{epoch:03d}.pt"
        try:
            import torch
            torch.save(state["model"].state_dict(), path)
        except Exception:
            pass

@dataclass
class CSVLogger(Hook):
    outdir: str
    filename: str = "metrics.csv"
    _fh: Optional[Any] = field(default=None, init=False, repr=False)

    def on_train_start(self, state: Dict[str, Any]):
        Path(self.outdir).mkdir(parents=True, exist_ok=True)
        path = Path(self.outdir) / self.filename
        header = "epoch,step,loss,lr,throughput,samples\n"
        new_file = not path.exists()
        self._fh = open(path, "a", buffering=1)
        if new_file:
            self._fh.write(header)

    def on_batch_end(self, state: Dict[str, Any]):
        if self._fh is None:
            return
        self._fh.write(
            f"{state['epoch']},{state['step']},{state['loss']:.6f},{state['lr']:.6f},{state.get('throughput',0):.3f},{state.get('batch_size',0)}\n"
        )

    def on_train_end(self, state: Dict[str, Any]):
        if self._fh is not None:
            self._fh.close()

@dataclass
class TensorBoardLogger(Hook):
    outdir: str
    def __post_init__(self):
        self._writer = None
    def on_train_start(self, state: Dict[str, Any]):
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=self.outdir)
    def on_batch_end(self, state: Dict[str, Any]):
        if self._writer is None:
            return
        self._writer.add_scalar("train/loss", state["loss"], state["step"])
        self._writer.add_scalar("train/lr", state["lr"], state["step"])
        if "throughput" in state:
            self._writer.add_scalar("train/throughput", state["throughput"], state["step"])
    def on_epoch_end(self, state: Dict[str, Any]):
        if self._writer is not None:
            self._writer.flush()
    def on_train_end(self, state: Dict[str, Any]):
        if self._writer is not None:
            self._writer.close()

@dataclass
class WandbLogger(Hook):
    project: str
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    def __post_init__(self):
        self._wandb = None
    def on_train_start(self, state: Dict[str, Any]):
        import wandb
        self._wandb = wandb
        wandb.init(project=self.project, name=self.name or state.get("experiment_name"), config=self.config or {})
    def on_batch_end(self, state: Dict[str, Any]):
        self._wandb.log({
            "train/loss": state["loss"],
            "train/lr": state["lr"],
            "train/throughput": state.get("throughput", 0),
            "step": state["step"],
            "epoch": state["epoch"],
        })
    def on_train_end(self, state: Dict[str, Any]):
        if self._wandb:
            self._wandb.finish()
