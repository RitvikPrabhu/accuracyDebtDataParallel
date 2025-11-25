from __future__ import annotations
from typing import Any, Dict, List
from time import time
from core.utils import world_info_from_env
import torch


class Trainer:
    def __init__(self, strategy, backend, hooks: List[Any] | None = None):
        self.strategy = strategy
        self.backend = backend
        self.hooks = hooks or []

    def _dispatch(self, method: str, state: Dict[str, Any]):
        for h in self.hooks:
            fn = getattr(h, method, None)
            if callable(fn):
                fn(state)

    def fit(self, model, optimizer, train_loader, num_epochs: int, log_interval: int = 50):
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        model = self.backend.prepare(model, amp=True, dtype="float32")
        model, optimizer = self.strategy.setup(model, optimizer, {})
        rank, world, _ = world_info_from_env()
        loss_fn = torch.nn.CrossEntropyLoss()
        model.train()
        step = 0
        state = {
            "experiment_name": None,
            "model": model,
            "optimizer": optimizer,
            "epoch": 0,
            "step": step,
            "loss": 0.0,
            "lr": next(iter(optimizer.param_groups))['lr'],
            "throughput": 0.0,
            "batch_size": getattr(train_loader, 'batch_size', 0),
        }
        self._dispatch("on_train_start", state)
        for epoch in range(1, num_epochs + 1):
            state["epoch"] = epoch
            self._dispatch("on_epoch_start", state)
            t0 = time(); seen = 0
            for i, (x, y) in enumerate(train_loader):
                dev = 'cuda' if torch.cuda.is_available() else 'cpu'
                x, y = x.to(dev), y.to(dev)
                with torch.cuda.amp.autocast(enabled=True):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)

                seen += x.size(0)
                dt = max(time() - t0, 1e-9)
                state.update({
                    "step": step,
                    "loss": float(loss.item()),
                    "lr": float(optimizer.param_groups[0]['lr']),
                    "throughput": seen / dt,
                    "batch_size": int(x.size(0)),
                })
                if step % log_interval == 0 and self.strategy.is_master():
                    print(f"[E{epoch} S{step}] loss={state['loss']:.4f} lr={state['lr']:.6f} thpt={state['throughput']:.1f} ex/s")
                self._dispatch("on_batch_end", state)
                step += 1
            self._dispatch("on_epoch_end", state)
        self._dispatch("on_train_end", state)
