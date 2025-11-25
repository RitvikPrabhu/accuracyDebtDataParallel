import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from quantom_ips.utils.registration import register_with_hydra

logger = logging.getLogger(__name__)


@dataclass
class DDPGANTrainerDefaults:
    id: str = "DDPGANTrainer"
    n_epochs: int = 10
    outer_update_epochs: int = 2
    gen_lr: float = 1e-5
    gen_beta_1: float = 0.5
    gen_beta_2: float = 0.999
    disc_lr: float = 1e-4
    disc_beta_1: float = 0.5
    disc_beta_2: float = 0.999
    batch_size: int = 10
    logdir: str = "${hydra:runtime.output_dir}"
    train_objective: bool = True
    distribute_disc: bool = False
    progress_bar: bool = False
    group_size: int = 4
    broadcast_buffers: bool = True
    find_unused_parameters: bool = False
    generator_update_frequency: int = 1
    discriminator_update_frequency: int = 1
    warmup_epochs: int = 1


@register_with_hydra(group="trainer", defaults=DDPGANTrainerDefaults, name="ddp_gan")
class DDPGANTrainer:
    """
    Torch DDP variant of the GAN trainer. Expects torch.distributed to be
    initialized before construction (e.g., via torchrun). Uses full-world
    allreduce (no hierarchical inner/outer grouping).
    """

    def __init__(self, config, device, dtype, rank: int, world_size: int, local_rank: int | None = None) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.outer_update_epochs = max(1, int(self.config.outer_update_epochs))
        self.distribute_disc = bool(self.config.distribute_disc)

        if not dist.is_initialized():
            raise RuntimeError("DDPGANTrainer expects torch.distributed to be initialized before construction.")
        self.global_group = dist.group.WORLD
        self.inner_group = self.global_group
        self.inner_group_ranks: list[int] = list(range(self.world_size))
        # Legacy compatibility hook (no-op now that we only use full-world allreduce)
        self._setup_groups()

    def _setup_groups(self) -> None:
        """
        Legacy shim. Previously created inner/outer groups; now a no-op because
        we rely on full-world allreduce.
        """
        return None

    def _wrap_ddp(self, module: torch.nn.Module, process_group=None) -> DDP:
        ddp_kwargs: Dict[str, Any] = {
            "broadcast_buffers": self.config.broadcast_buffers,
            "find_unused_parameters": self.config.find_unused_parameters,
        }
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            if self.local_rank is not None:
                ddp_kwargs["device_ids"] = [self.local_rank]
                ddp_kwargs["output_device"] = self.local_rank
        if process_group is not None:
            ddp_kwargs["process_group"] = process_group
        return DDP(module, **ddp_kwargs)

    def _reduce_mean(self, value: torch.Tensor) -> float:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= float(self.world_size)
        return value.item()

    def _reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        reduced: Dict[str, float] = {}
        for k, v in metrics.items():
            t = torch.tensor(v, device=self.device, dtype=torch.float32)
            reduced[k] = self._reduce_mean(t)
        return reduced

    def _outer_sync_module(self, module: torch.nn.Module | None) -> None:
        """
        Coarse synchronization across the full world. Redundant with gradient
        allreduce but kept to match HVD outer sync behavior.
        """
        if module is None:
            return

        for p in module.parameters():
            if not p.requires_grad:
                continue
            dist.all_reduce(p.data, op=dist.ReduceOp.SUM, group=self.global_group)
            p.data /= float(self.world_size)

    def _get_loss_norm(self) -> float:
        loss_norm = 1.0
        loss_fn = getattr(getattr(self.env, "config", None), "loss_fn", "").lower()
        if loss_fn == "mse":
            loss_norm = 0.25
        elif loss_fn == "bce":
            loss_norm = -np.log(0.5)
        return loss_norm

    def train_step(self, update_gen: bool, update_disc: bool, loss_norm: float) -> Dict[str, float]:
        self.generator.zero_grad(set_to_none=True)

        params = self.opt.forward(self.config.batch_size)
        losses, fake_events = self.env.step(params)
        gen_loss = losses["generator"]

        if update_gen:
            gen_loss.backward()
            self.gen_optimizer.step()

        outputs = {"gen_loss": gen_loss.detach().item()}

        if self.config.train_objective:
            self.discriminator.zero_grad(set_to_none=True)
            disc_losses = self.env.get_objective_losses(fake_events.detach())
            real_loss = disc_losses["real"]
            fake_loss = disc_losses["generator"]
            full_disc_loss = real_loss + fake_loss
            if update_disc:
                full_disc_loss.backward()
                self.disc_optimizer.step()
            outputs["real_loss"] = real_loss.detach().item()
            outputs["fake_loss"] = fake_loss.detach().item()
            if "log_sad" in disc_losses:
                outputs["log_sad_score"] = disc_losses["log_sad"]

        outputs = {k: v / loss_norm for k, v in outputs.items()}

        if dist.is_initialized():
            outputs = self._reduce_metrics(outputs)

        return outputs

    def run(self, opt, env, analysis):
        self.opt = opt
        self.env = env

        self.generator = self._wrap_ddp(self.opt.model, process_group=self.inner_group)
        if self.config.train_objective:
            disc_module = self.env.objective.model
            if self.distribute_disc:
                self.discriminator = self._wrap_ddp(disc_module, process_group=self.inner_group)
            else:
                self.discriminator = disc_module

        self.gen_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            self.config.gen_lr,
            betas=(self.config.gen_beta_1, self.config.gen_beta_2),
        )
        if self.config.train_objective:
            self.disc_optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                self.config.disc_lr,
                betas=(self.config.disc_beta_1, self.config.disc_beta_2),
            )

        loss_norm = self._get_loss_norm()

        start_time = time.perf_counter()
        for epoch in range(1, self.config.n_epochs + 1):
            allow_update = epoch > self.config.warmup_epochs
            update_gen = allow_update and (epoch % self.config.generator_update_frequency == 0)
            update_disc = allow_update and (epoch % self.config.discriminator_update_frequency == 0)

            metrics = self.train_step(update_gen, update_disc, loss_norm)
            if epoch % self.outer_update_epochs == 0:
                self._outer_sync_module(self.generator)
                if self.config.train_objective and self.distribute_disc:
                    self._outer_sync_module(self.discriminator)
            if self.rank == 0:
                analysis.forward(
                    self.opt,
                    epoch=epoch,
                    n_epochs=self.config.n_epochs,
                    loss_history={k: [v] for k, v in metrics.items()},
                    is_online=True,
                )

        end_time = time.perf_counter()
        elapsed_local = torch.tensor([end_time - start_time], device=self.device)
        elapsed_max = elapsed_local.clone()
        if dist.is_initialized():
            dist.all_reduce(elapsed_max, op=dist.ReduceOp.MAX)
        if self.rank == 0:
            logger.info(f"Training finished in {elapsed_max.item():.2f}s")
            analysis.forward(
                self.opt,
                epoch=self.config.n_epochs,
                n_epochs=self.config.n_epochs,
                loss_history=None,
                force=True,
            )
