import logging
import os
import torch
from tqdm.auto import tqdm
from pathlib import Path

from dataclasses import dataclass

from quantom_ips.utils.registration import register_with_hydra

logger = logging.getLogger(__name__)


@dataclass
class GANTrainerDefaults:
    id: str = "GANTrainer"
    n_epochs: int = 5
    gen_lr: float = 1e-5
    disc_lr: float = 1e-5
    batch_size: int = 10
    logdir: str = "${hydra:runtime.output_dir}"
    noise_dim: int = 100
    train_objective: bool = True
    progress_bar: bool = False


@register_with_hydra(group="trainer", defaults=GANTrainerDefaults, name="GAN")
class GANTrainer:
    def __init__(self, config, device, dtype) -> None:

        self.config = config
        self.dtype = dtype
        self.device = device

    def train_step(self):
        self.generator.zero_grad()

        params = self.opt.forward(self.config.batch_size)
        gen_losses, fake_events = self.env.step(params)
        for gen_loss in gen_losses.values():
            gen_loss.backward()
        self.gen_optimizer.step()

        outputs = {}
        for loss_type in gen_losses:
            outputs[loss_type] = gen_losses[loss_type].detach().cpu().item()

        if self.config.train_objective:
            self.discriminator.zero_grad()
            losses = self.env.get_objective_losses(fake_events.detach())
            for loss in losses.values():
                loss.backward()
            self.disc_optimizer.step()

            for loss_type in losses:
                outputs[loss_type + "_disc"] = losses[loss_type].detach().cpu().item()

        return outputs

    def run(self, opt, env, analysis):
        self.opt = opt
        self.env = env

        os.makedirs(Path(self.config.logdir), exist_ok=True)

        self.generator = self.opt.model
        self.discriminator = self.env.objective.model

        self.gen_optimizer = torch.optim.Adam(
            self.generator.parameters(), self.config.gen_lr
        )
        if self.config.train_objective:
            self.disc_optimizer = torch.optim.Adam(
                self.discriminator.parameters(), self.config.disc_lr
            )

        # Check optimizer outputs:
        opt_out = self.opt.forward(self.config.batch_size)
        if opt_out.shape[0] != self.config.batch_size:
            logger.warning(
                f"Optimizer only returning {opt_out.shape[0]} solutions "
                f"when {self.config.batch_size} were requested. "
                "This may result in undesired results."
            )

        losses = {}
        analysis.forward(self.opt, self.env, epoch=0)
        logger.info(f"Training for {self.config.n_epochs} epochs.")
        for epoch in tqdm(
            range(1, self.config.n_epochs + 1), disable=not self.config.progress_bar
        ):
            train_output = self.train_step()

            if epoch == 1:
                for k in train_output:
                    losses[k] = [train_output[k]]
            else:
                for k in train_output:
                    losses[k].append(train_output[k])

            with torch.no_grad():
                analysis.forward(self.opt, self.env, epoch=epoch)

        logger.info("Finished Training!")

        analysis.forward(
            self.opt,
            self.env,
            epoch=self.config.n_epochs,
            loss_history=losses,
            force=True,
        )
