import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

import horovod.torch as hvd
import numpy as np
import torch

from quantom_ips.utils.registration import register_with_hydra

logger = logging.getLogger(__name__)


@dataclass
class HVDGANTrainerDefaults:
    id: str = "HVDGANTrainer"
    n_epochs: int = 10000
    outer_update_epochs: int = 2
    gen_lr: float = 1e-5
    gen_beta_1: float = 0.5
    gen_beta_2: float = 0.999
    disc_lr: float = 1e-4
    disc_beta_1: float = 0.5
    disc_beta_2: float = 0.999
    batch_size: int = 1024
    logdir: str = "${hydra:runtime.output_dir}"
    train_objective: bool = True
    distribute_disc: bool = False
    progress_bar: bool = False
    generator_update_frequency: int = 1
    discriminator_update_frequency: int = 1
    warmup_epochs: int = 1


@register_with_hydra(group="trainer", defaults=HVDGANTrainerDefaults, name="hvd_gan")
class HVDGANTrainer:
    """
    Horovod variant of the GAN trainer. Uses hvd.DistributedOptimizer and
    broadcasts parameters/optimizer state.
    """

    def __init__(self, config, device, dtype) -> None:
        if not hvd.is_initialized():
            raise RuntimeError("HVDGANTrainer requires horovod.init().")

        self.config = config
        self.dtype = dtype
        self.device = device
        self.outer_update_epochs = max(1, int(self.config.outer_update_epochs))
        self.distribute_disc = bool(self.config.distribute_disc)

    def _reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        reduced: Dict[str, float] = {}
        for k, v in metrics.items():
            t = torch.tensor(v, device=self.device, dtype=torch.float32)
            reduced[k] = hvd.allreduce(t, name=k).item()
        return reduced

    def _outer_sync_module(self, module: torch.nn.Module | None) -> None:
        if module is None:
            return

        for idx, p in enumerate(module.parameters()):
            if not p.requires_grad:
                continue
            synced = hvd.allreduce(p.data, name=f"outer_sync_{idx}")
            p.data.copy_(synced)

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

        outputs = self._reduce_metrics(outputs)
        return outputs

    def run(self, opt, env, analysis):
        self.opt = opt
        self.env = env

        self.generator = self.opt.model
        if self.config.train_objective:
            self.discriminator = self.env.objective.model

        gen_named_params = [(f"gen_{name}", param) for name, param in self.generator.named_parameters()]
        self.gen_optimizer = hvd.DistributedOptimizer(
            torch.optim.Adam(
                self.generator.parameters(),
                self.config.gen_lr,
                betas=(self.config.gen_beta_1, self.config.gen_beta_2),
            ),
            named_parameters=gen_named_params,
            # Generator uses one backward pass, discriminator uses a separate one; allow two to avoid hook assertion.
            backward_passes_per_step=2,
        )
        if self.config.train_objective:
            self.disc_optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                self.config.disc_lr,
                betas=(self.config.disc_beta_1, self.config.disc_beta_2),
            )
            if self.distribute_disc:
                disc_named_params = [
                    (f"disc_{name}", param) for name, param in self.discriminator.named_parameters()
                ]
                self.disc_optimizer = hvd.DistributedOptimizer(
                    self.disc_optimizer,
                    named_parameters=disc_named_params,
                    backward_passes_per_step=2,
                )

        # Broadcast parameters & optimizer state.
        hvd.broadcast_parameters(self.generator.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.gen_optimizer, root_rank=0)
        if self.config.train_objective:
            if self.distribute_disc:
                hvd.broadcast_parameters(self.discriminator.state_dict(), root_rank=0)
                hvd.broadcast_optimizer_state(self.disc_optimizer, root_rank=0)

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

            if hvd.rank() == 0:
                # Match logging cadence to distributed analysis.
                logger.info(
                    " | ".join([f"Epoch: {epoch}"] + [f"{k}: {v}" for k, v in metrics.items()])
                )
                analysis.forward(
                    self.opt,
                    epoch=epoch,
                    n_epochs=self.config.n_epochs,
                    loss_history={k: [v] for k, v in metrics.items()},
                    is_online=True,
                )

        # Synchronize end time across all ranks and report total wall time.
        end_time = time.perf_counter()
        elapsed_local = torch.tensor([end_time - start_time], device=self.device)
        elapsed_max = hvd.allreduce(elapsed_local, name="train_time_max", op=hvd.mpi_ops.Max)
        if hvd.rank() == 0:
            logger.info(f"Training finished in {elapsed_max.item():.2f}s")
            analysis.forward(
                self.opt,
                epoch=self.config.n_epochs,
                n_epochs=self.config.n_epochs,
                loss_history=None,
                force=True,
            )
