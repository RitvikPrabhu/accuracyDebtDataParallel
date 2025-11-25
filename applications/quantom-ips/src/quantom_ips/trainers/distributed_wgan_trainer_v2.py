import mpi4py

mpi4py.rc.thread_level = "serialized"
from mpi4py import MPI
import logging
import os
import torch
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
class DistributedWGANTrainerV2Defaults:
    id: str = "DistributedWGANTrainerV2"
    defaults: List[Any] = field(default_factory=lambda: defaults)
    logdir: str = "${hydra:runtime.output_dir}"
    profile_folder: str = "profiles"
    profile_name: str = "distributed_gan"
    trainer: Any = MISSING
    n_epochs: int = 10
    n_data_sets: int = 3
    n_unknowns: int = 2
    outer_update_epochs: int = 2
    gradient_scale: float = 1.0
    gen_lr: float = 1e-5
    gen_beta_1: float = 0.0
    gen_beta_2: float = 0.9
    gen_ema: float = -1.0
    disc_lr: float = 1e-4
    disc_beta_1: float = 0.0
    disc_beta_2: float = 0.9
    optimizer: str = "adam"
    gradient_penalty_scale: float = 1.0
    batch_size: int = 10
    torch_device: str = "cpu"
    generator_update_frequency: int = 1
    discriminator_update_frequency: int = 1
    enable_analysis: bool = True
    use_theory_loss: bool = False
    disable_gpu_mon: bool = False
    distribute_discriminator: bool = False


@register_with_hydra(
    group="trainer",
    defaults=DistributedWGANTrainerV2Defaults,
    name="distributed_wgan_trainer_v2",
)
class DistributedWGANTrainerV2:

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
        # Set everything up for the collection of generator predictions:
        self.pred_comm = None
        self.pred_rank = -1
        self.n_pred_ranks = -1
        self.pred_neighbours = None
        self.freeze_generator = False

        if config.n_data_sets > 1:
          self.pred_comm = self.comm.Split(color=self.grad_transport.rank / config.n_data_sets)
          self.pred_rank = self.pred_comm.Get_rank()
          self.n_pred_ranks = self.pred_comm.Get_size()

          self.pred_neighbours = self.get_neighbours(self.pred_rank,self.n_pred_ranks)

          if self.pred_rank >= config.n_unknowns:
              self.freeze_generator = True
        # Get the devices and dtype:
        self.devices = self.grad_transport.devices
        self.dtype = self.grad_transport.dtype

        # Create a folder for profiling:
        self.profile_folder = self.config.logdir + "/" + self.config.profile_folder
        if self.comm.Get_rank() == 0:
            os.makedirs(self.profile_folder, exist_ok=True)

    # *********************************************
    
    # Collect predictions from all generators:
    # *********************************************
    def get_neighbours(self, current_rank, num_ranks):
        left = ((current_rank - 1) + num_ranks) % num_ranks
        right = (current_rank + 1) % num_ranks

        return left, right
    
    def ring_allreduce(self, data, mpi_comm, num_ranks, neighbours):
        send_data = np.copy(data)
        recv_data = np.copy(data)
        accum_data = np.copy(data)

        # +++++++++++++++++++++++
        for i in range(num_ranks - 1):
            if i % 2 == 0:
                # Send send_buff
                send_req = mpi_comm.Isend(send_data, neighbours[1])
                mpi_comm.Recv(recv_data, neighbours[0])
                accum_data[:] += recv_data[:]
            else:
                # Send recv_buff
                send_req = mpi_comm.Isend(recv_data, neighbours[1])
                mpi_comm.Recv(send_data, neighbours[0])
                accum_data[:] += send_data[:]
            send_req.wait()
        # +++++++++++++++++++++++

        return accum_data
    
    def collect_predictions(self,local_prediction):
        if self.pred_rank >= 0:
           batch_size, _, dim_x, dim_y = local_prediction.size()

           local_predictions = np.zeros((batch_size,self.n_pred_ranks,dim_x,dim_y))
           local_predictions[:,self.pred_rank,:,:] = local_prediction.detach().cpu().numpy()[:,0,:,:]
           predictions_np = self.ring_allreduce(local_predictions,self.pred_comm,self.n_pred_ranks,self.pred_neighbours)
           predictions_torch = torch.as_tensor(predictions_np,dtype=local_prediction.dtype,device=local_prediction.device)
           predictions_torch[:,self.pred_rank,:,:] = local_prediction[:,0,:,:]
           return predictions_torch
        
        return local_prediction
    # *********************************************

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
        if do_update and ~self.freeze_generator:
            # Backpropagation:
            loss.backward(retain_graph=True)
            # Use theory loss, if required:
            if self.config.use_theory_loss and theory_loss is not None:
                theory_loss.backward(retain_graph=True)
            # Transprot gradients:
            if self.pred_neighbours is None:
               self.grad_transport.forward(
                  generator, update_outer_group, gradient_scale=self.config.gradient_scale
               )
            # Update generator weights:
            generator_optim.step()

    # ---------------------------

    # Define single discriminator update --> So that we can monitor the gpu utilization:
    def discriminator_update(
        self, discriminator, discriminator_optim, critic_loss, update_outer_group, do_update
    ):
        if do_update:
            # Backpropagation:
            critic_loss.backward()
            if self.config.distribute_discriminator:
                # Transprot gradients:
                self.grad_transport.forward(
                   discriminator, update_outer_group, gradient_scale=self.config.gradient_scale
                )
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
        if self.pred_neighbours is not None:
          #  print(f"pre-shape: {params.size()}")
            params = self.collect_predictions(params)
         #   print(f"post-shape: {params.size()}")
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
        gp = obj_out.get("gradient_penalty")
        log_sad_score = obj_out.get("log_sad")
        # Compute the critics loss:
        # PLEASE Note: The official definition of the ciritc loss is: mean(fake_samples) - mean(real_samples) + scale*gradient_penalty
        critic_loss = fake_loss - real_loss + self.config.gradient_penalty_scale*gp
        raw_critic_loss = fake_loss - real_loss

        # Use losses to update the discriminator network:
        _, disc_opt_footprint = mon_gpu_usage(
            self.grad_transport.torch_device_id, self.config.disable_gpu_mon
        )(self.discriminator_update)(
            discriminator, discriminator_optim, critic_loss, update_outer_group, update_disc
        )

        return {
            "critic_loss":critic_loss.detach().cpu().numpy(),
            "raw_critic_loss":raw_critic_loss.detach().cpu().numpy(),
            "gen_loss": gen_loss.detach().cpu().numpy(),
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

            # Generator:
            generator = opt.model
            generator_optimizer = None
            if self.config.optimizer.lower() == "adam":
              generator_optimizer = torch.optim.Adam(
                generator.parameters(),
                lr=self.config.gen_lr,
                betas=(self.config.gen_beta_1, self.config.gen_beta_2),
             )
            if self.config.optimizer.lower() == "rmsprop":
                generator_optimizer = torch.optim.RMSprop(
                    generator.parameters(),
                    lr=self.config.gen_lr
                )
            # Discriminator:
            discriminator = env.objective.model
            discriminator_optimizer = None
            if self.config.optimizer.lower() == "adam":
              discriminator_optimizer = torch.optim.Adam(
                discriminator.parameters(),
                lr=self.config.disc_lr,
                betas=(self.config.disc_beta_1, self.config.disc_beta_2),
             )
            if self.config.optimizer.lower() == "rmsprop":
                discriminator_optimizer = torch.optim.RMSprop(
                    discriminator.parameters(),
                    lr=self.config.disc_lr
                )
            # Overwrite the loss function within the objective, so that we use the wasserstein objective:
            env.objective.config.loss_fn = "wasserstein"

            # EMA generator, if desired:
            ema_generator = self.get_ema_model(self.config.gen_ema, generator)

            # Synchronize generator models accross ranks:
            if self.pred_neighbours is None:
               self.grad_transport.sync_model(generator, generator_optimizer)
            if self.config.distribute_discriminator:
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
                # We do not run an update during the first epoch, just so that we can see the initial prediction(s):
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

                # Run online analysis, i.e. monitor all relevant metrics:
                analysis.forward(
                    generator, epoch, self.config.n_epochs, current_results, True, self.pred_rank
                )
            # +++++++++++++++++++++++++++++

            # Clean up:
            self.grad_transport.clear()

            # And now write information to file:
            analysis.forward()

        # Run with the profiler:
        run_with_profiler()

    # *********************************************
