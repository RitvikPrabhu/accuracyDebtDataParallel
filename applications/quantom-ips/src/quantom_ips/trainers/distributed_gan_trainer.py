import mpi4py

mpi4py.rc.thread_level = "serialized"
from mpi4py import MPI
import logging
import os
import torch
import time
from torch import cuda
import numpy as np
import cProfile
from pathlib import Path
from typing import List, Any
from dataclasses import dataclass, field
from omegaconf import MISSING
from quantom_ips.utils.monitors.gpu_performance_monitor import mon_gpu_usage
from dataclasses import dataclass
from quantom_ips.utils.registration import register_with_hydra, make

logger = logging.getLogger(__name__)

defaults = [
    {"gradient_transport": "torch_arar"},
    "_self_",
]


@dataclass
class DistributedGANTrainerDefaults:
    id: str = "DistributedGANTrainer"
    defaults: List[Any] = field(default_factory=lambda: defaults)
    logdir: str = "${hydra:runtime.output_dir}"
    profile_folder: str = "profiles"
    profile_name: str = "distributed_gan"
    trainer: Any = MISSING
    n_epochs: int = 10000
    outer_update_epochs: int = 2
    gradient_scale: float = 1.0
    gen_lr: float = 1e-5
    gen_beta_1: float = 0.5
    gen_beta_2: float = 0.999
    gen_ema: float = -1.0
    disc_lr: float = 1e-4
    disc_beta_1: float = 0.5
    disc_beta_2: float = 0.999
    batch_size: int = 1
    torch_device: str = "cuda"
    generator_update_frequency: int = 1
    discriminator_update_frequency: int = 1
    enable_analysis: bool = True
    use_theory_loss: bool = False
    disable_gpu_mon: bool = False
    distribute_disc: bool = False


@register_with_hydra(
    group="trainer",
    defaults=DistributedGANTrainerDefaults,
    name="distributed_gan_trainer",
)
class DistributedGANTrainer:

    # Initialize:
    # *********************************************
    def __init__(self, config) -> None:
        # Config:
        self.config = config

        # Gradient Transport: (This one needs to be at first, as it determines the torch device and handles all the mpi-related stuff)
        self.grad_transport = make(
            self.config.gradient_transport.id,
            config=self.config.gradient_transport,
        )
        # Get mpi_comm so that we have access to ranks if needed
        self.comm = self.grad_transport.comm
        # Get the devices and dtype:
        self.devices = self.grad_transport.devices
        self.dtype = self.grad_transport.dtype

        # Create a folder for profiling:
        self.profile_folder = self.config.logdir + "/" + self.config.profile_folder
        if self.comm.Get_rank() == 0:
            os.makedirs(self.profile_folder, exist_ok=True)

    # *********************************************
    
    def mon_grads(self,model,indicator):
        for n, p in model.named_parameters():
            if 'weight' in n or 'bias' in n:
               print(f"{indicator}-grads: {torch.mean(p.grad)}")

    # Define a single training step:
    # *********************************************
    # Define single generator update --> So that we can monitor the gpu utilization:
    def generator_update(
        self,
        loss,
        theory_loss,
        generator,
        generator_optim,
        update_outer_group,
        do_update,
    ):
        if do_update:
            # Backpropagation:
            loss.backward(retain_graph=True)
            # Use theory loss, if required:
            if self.config.use_theory_loss and theory_loss is not None:
                theory_loss.backward(retain_graph=True)
            # Transprot gradients:
            self.grad_transport.forward(
                generator, update_outer_group, gradient_scale=self.config.gradient_scale
            )
            
            #self.mon_grads(generator,'generator')

            # Update generator weights:
            generator_optim.step()

    # ---------------------------

    # Define single discriminator update --> So that we can monitor the gpu utilization:
    def discriminator_update(
        self, discriminator, discriminator_optim, real_loss, fake_loss, do_update, update_outer_group
    ):
        if do_update:
            # Compute full loss:
            full_disc_loss = real_loss + fake_loss
            # Backpropagation:
            full_disc_loss.backward()
            # Transprot gradients for discriminator, if wanted:
            if self.config.distribute_disc:
              self.grad_transport.forward(
                discriminator, update_outer_group, gradient_scale=self.config.gradient_scale
              )

           # self.mon_grads(discriminator,'discriminator')
            # Update discriminator weights:
            discriminator_optim.step()

    # ---------------------------

    # Full training step:
    def train_step(
        self,
        env,
        opt,
        generator,
        generator_optim,
        discriminator,
        discriminator_optim,
        update_gen,
        update_disc,
        update_outer_group,
        loss_norm,
    ):
        # Train generator:

        # Reset generator gradient first:
        generator_optim.zero_grad(set_to_none=True)
        # Determine parameters and determine loss:
        params = opt.forward(self.config.batch_size)
        env_out, env_step_footprint = mon_gpu_usage(
            self.grad_transport.torch_device_id, self.config.disable_gpu_mon
        )(env.step)(params)
        loss, fake_events = env_out
        gen_loss = loss.get("generator")
        theory_loss = loss.get("theory_loss")

        # Update the generator network and check the gpu utilization at the same time:
        _, gen_opt_footprint = mon_gpu_usage(
            self.grad_transport.torch_device_id, self.config.disable_gpu_mon
        )(self.generator_update)(
            gen_loss,
            theory_loss,
            generator,
            generator_optim,
            update_outer_group,
            update_gen
        )

        # Train the discriminator:

        # Reset optimizer:
        discriminator_optim.zero_grad(set_to_none=True)
        # Get losses:
        obj_out, obj_score_footpring = mon_gpu_usage(
            self.grad_transport.torch_device_id, self.config.disable_gpu_mon
        )(env.get_objective_losses)(fake_events)
        real_loss = obj_out.get("real")
        fake_loss = obj_out.get("generator")
        log_sad_score = obj_out.get("log_sad")

        # Use losses to update the discriminator network:
        _, disc_opt_footprint = mon_gpu_usage(
            self.grad_transport.torch_device_id, self.config.disable_gpu_mon
        )(self.discriminator_update)(
            discriminator, discriminator_optim, real_loss, fake_loss, update_disc, update_outer_group
        )

        return {
            "real_loss": real_loss.detach().cpu().numpy() / loss_norm,
            "fake_loss": fake_loss.detach().cpu().numpy() / loss_norm,
            "gen_loss": gen_loss.detach().cpu().numpy() / loss_norm,
            "env_step_gpu_mem_fraction": env_step_footprint["gpu_used_mem_fraction"],
            "env_step_gpu_utilization": env_step_footprint["gpu_utilization"],
            "obj_gpu_mem_fraction": obj_score_footpring["gpu_used_mem_fraction"],
            "obj_gpu_utilization": obj_score_footpring["gpu_utilization"],
            "gen_opt_gpu_mem_fraction": gen_opt_footprint["gpu_used_mem_fraction"],
            "gen_opt_gpu_utilization": gen_opt_footprint["gpu_utilization"],
            "disc_opt_gpu_mem_fraction": disc_opt_footprint["gpu_used_mem_fraction"],
            "disc_opt_gpu_utilization": disc_opt_footprint["gpu_utilization"],
            "log_sad_score": log_sad_score,
        }

    # *********************************************

    # Use EMA (Exponential Moving Average) to smoothen the training a bit:
    # *********************************************
    def get_ema_model(self, ema_decay, model):
        if ema_decay <= 0.0:
            return None

        ema_avg = (
            lambda averaged_model_parameter, model_parameter, num_averaged: ema_decay
            * averaged_model_parameter
            + (1.0 - ema_decay) * model_parameter
        )
        return torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)

    # *********************************************

    # Run everything: (I am not proud of this, because it feels very clunky, but this is something for later...)
    # *********************************************
    def run(self, opt, env, analysis):
        # Define a profiling function:
        def profile():
            def prof_decorator(f):
                def wrap_f(*args, **kwargs):
                    pr = cProfile.Profile()
                    pr.enable()
                    result = f(*args, **kwargs)
                    pr.disable()

                    filename_r = (
                        self.config.logdir
                        + "/"
                        + self.config.profile_folder
                        + "/"
                        + self.config.profile_name
                        + "_r"
                        + str(self.grad_transport.rank)
                    )
                    pr.dump_stats(filename_r + ".prof")
                    pr.dump_stats(filename_r + ".pstats")

                    return result

                return wrap_f

            return prof_decorator

        # Run trainer with profiler:
        @profile()
        def run_with_profiler():
            # Get models and their optimizers:
            total_start = MPI.Wtime()

            # Generator:
            generator = opt.model
            generator_optimizer = torch.optim.Adam(
                generator.parameters(),
                lr=self.config.gen_lr,
                betas=(self.config.gen_beta_1, self.config.gen_beta_2),
            )
            # Discriminator:
            discriminator = env.objective.model
            discriminator_optimizer = torch.optim.Adam(
                discriminator.parameters(),
                lr=self.config.disc_lr,
                betas=(self.config.disc_beta_1, self.config.disc_beta_2),
            )

            # EMA generator, if desired:
            ema_generator = self.get_ema_model(self.config.gen_ema, generator)

            # Synchronize generator models accross ranks:
            self.grad_transport.sync_model(generator, generator_optimizer)
            # Synchronize discriminator if its gradients are shared as well:
            if self.config.distribute_disc:
                self.grad_transport.sync_model(discriminator,discriminator_optimizer)

            # Get the normalization for the loss:
            loss_norm = 1.0
            if env.config.loss_fn.lower() == "mse":
                loss_norm = 0.25
            if env.config.loss_fn.lower() == "bce":
                loss_norm = -np.log(0.5)

            # Clock is ticking:
            self.grad_transport.comm.Barrier()
            t_start = MPI.Wtime()
            # +++++++++++++++++++++++++++++
            for epoch in range(1, self.config.n_epochs + 1):
                update_gen = False
                update_disc = False
                if epoch > 1 and epoch % self.config.generator_update_frequency == 0:
                    update_gen = True
                if epoch > 1 and epoch % self.config.discriminator_update_frequency == 0:
                    update_disc = True

                outer_group_update = False
                if epoch % self.config.outer_update_epochs == 0:
                    outer_group_update = True

                # Run a training step:
                current_results = self.train_step(
                    env,
                    opt,
                    generator,
                    generator_optimizer,
                    discriminator,
                    discriminator_optimizer,
                    update_gen=update_gen,
                    update_disc=update_disc,
                    update_outer_group=outer_group_update,
                    loss_norm=loss_norm,
                )

                # Update the ema_generator, if it does exist:
                if update_gen == True and ema_generator is not None:
                    ema_generator.update_parameters(generator)

                # Record time and register it with the loss history:
                t_end = MPI.Wtime()

                current_results["train_step_time"] = t_end - t_start
                logger.info(
                    " | ".join([f"Epoch: {epoch}"] + [f"{k}: {v}" for k, v in current_results.items()])
                )

                # Run online analysis, i.e. monitor all relevant metrics:
                analysis.forward(
                    generator, epoch, self.config.n_epochs, current_results, True
                )
            # +++++++++++++++++++++++++++++

            # Clean up:
            self.grad_transport.clear()

            # And now write information to file:
            analysis.forward()

            total_end = MPI.Wtime()
            elapsed_local = total_end - total_start
            elapsed_max = self.comm.allreduce(elapsed_local, op=MPI.MAX)
            if self.comm.Get_rank() == 0:
                logger.info(f"Training finished in {elapsed_max:.2f}s")

        # Run with the profiler:
        run_with_profiler()

    # *********************************************
