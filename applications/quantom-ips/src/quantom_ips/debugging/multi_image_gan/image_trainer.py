import mpi4py
mpi4py.rc.thread_level = "serialized"
from mpi4py import MPI
import logging
import torch
import numpy as np
from typing import List, Any, Optional
from dataclasses import dataclass, field
from quantom_ips.utils.registration import register_with_hydra, make
from torch_model_components import get_dropout_probs

logger = logging.getLogger(__name__)

defaults = [
    {"gradient_transport": "torch_arar_pcgrad"},
    "_self_",
]


@dataclass
class ImageTrainerDefaults:
    id: str = "ImageTrainer"
    defaults: List[Any] = field(default_factory=lambda: defaults)
    n_epochs: int = 30
    outer_update_epochs: int = 2
    gradient_scale: float = 1.0
    gen_lr: float = 1e-5
    gen_beta_1: float = 0.5
    gen_beta_2: float = 0.999
    gen_ema: float = -1.0
    disc_lr: float = 1e-4
    disc_beta_1: float = 0.5
    disc_beta_2: float = 0.999
    batch_size: int = 10
    alpha_max: List[float] = field(default_factory=lambda: [0.9,0.6,0.3])
    E_warm_up: List[int] = field(default_factory=lambda: [1,1,1])
    E_ramp_up: List[int] = field(default_factory=lambda: [3,3,3])
    E_hold: List[int] = field(default_factory=lambda: [2,2,2])
    E_ramp_down: List[int] = field(default_factory=lambda: [1,1,1])
    skip_alignment: bool = False
    
@register_with_hydra(
    group="trainer",
    defaults=ImageTrainerDefaults,
    name="image_trainer",
)
class ImageTrainer:

    def __init__(self,config):
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

    def generator_update(
        self,
        loss,
        generator,
        generator_optim,
        update_outer_group,
        do_update,
        alpha_scale,
    ):
        if do_update:
            # Backpropagation:
            loss = generator.add_regularization_loss(loss)
            loss.backward(retain_graph=True)
            # Transprot gradients: 
            _, uncorr_cos_sim, corr_cos_sim = self.grad_transport.forward(
                generator, update_outer_group, gradient_scale=self.config.gradient_scale, pcgrad_scale=alpha_scale, skip_alignment=self.config.skip_alignment
            )
            # Update generator weights:
            generator_optim.step()

            return uncorr_cos_sim, corr_cos_sim
        return np.zeros(2), np.zeros(2)

    def discriminator_update(
        self, 
        discriminator_optim, 
        real_loss, 
        fake_loss, 
        do_update
    ):
        if do_update:
            # Compute full loss:
            full_disc_loss = real_loss + fake_loss
            # Backpropagation:
            full_disc_loss.backward()
            # Update discriminator weights:
            discriminator_optim.step()

    def train_step(
        self,
        env,
        opt,
        generator,
        generator_optim,
        discriminator_optim,
        update_outer_group,
        loss_norm,
        do_update,
        alpha_scale
    ):
        # Train the generator:

        # Reset optimizer first:
        generator_optim.zero_grad(set_to_none=True)
        # Determine parameters and determine loss:
        params = opt.forward(self.config.batch_size)
        gen_loss, fake_data = env.step(params)
        
        pre_cos_sim, post_cos_sim = self.generator_update(
            gen_loss['fake'],
            generator,
            generator_optim,
            update_outer_group,
            do_update,
            alpha_scale
        )

        # Record trainable dropout rate, if existent
        dropout_rates = get_dropout_probs(generator)

        # Train the discriminator:

        # Reset optimizer:
        discriminator_optim.zero_grad(set_to_none=True)
        # Get losses:
        objective_losses = env.get_objective_losses(fake_data)
        
        # Use losses to update the discriminator network:
        self.discriminator_update(
            discriminator_optim,
            objective_losses['real'],
            objective_losses['fake'],
            do_update
        )

        history = {
            "real_loss": objective_losses['real'].detach().cpu().numpy() / loss_norm,
            "fake_loss": objective_losses['fake'].detach().cpu().numpy() / loss_norm,
            "gen_loss": gen_loss['fake'].detach().cpu().numpy() / loss_norm,
            "pre_cos_sim_pos": pre_cos_sim[0],
            "pre_cos_sim_neg": pre_cos_sim[1],
            "post_cos_sim_pos": post_cos_sim[0],
            "post_cos_sim_neg": post_cos_sim[1],
        }
        
        # Register rho parameter if existent:
        if dropout_rates is not None:
            for i in range(dropout_rates.shape[0]):
                history[f'dropout_layer_{i}'] = dropout_rates[i]
       
        return history
    
    def single_alpha_schedule(self, epoch, alpha_max, E_warm_up, E_ramp_up, E_hold, E_ramp_down):
        if epoch < E_warm_up:
           return 0.0
        e = epoch - E_warm_up
        if e < E_ramp_up:
           return alpha_max * (e / max(1, E_ramp_up))
        e -= E_ramp_up
        if e < E_hold:
           return alpha_max
        e -= E_hold
        if e < E_ramp_down:
           return alpha_max * (1.0 - (e / max(1, E_ramp_down)))
        return 0.0
    
    def alpha_schedule(self, epoch, alpha_max, E_warm_up, E_ramp_up, E_hold, E_ramp_down):
        n_cycles = len(E_warm_up)
        min_epoch = 0
        idx = -1
        epoch_in = -1
        for i in range(n_cycles):
           max_epoch = E_warm_up[i] + E_ramp_up[i] + E_hold[i] + E_ramp_down[i] + min_epoch
           if epoch >= min_epoch and epoch < max_epoch:
            idx = i
            epoch_in = epoch - min_epoch
           min_epoch += E_warm_up[i] + E_ramp_up[i] + E_hold[i] + E_ramp_down[i]

        return self.single_alpha_schedule(
          epoch_in,
          alpha_max[idx],
          E_warm_up[idx],
          E_ramp_up[idx],
          E_hold[idx],
          E_ramp_down[idx]
        )

    def run(self,opt,env,analysis):
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
        # Synchronize generator models accross ranks:
        self.grad_transport.sync_model(generator, generator_optimizer)
        
        # Get the normalization for the loss:
        loss_norm = env.objective.loss_norm

        # Clock is ticking:
        self.grad_transport.comm.Barrier()
        t_start = MPI.Wtime()
        # +++++++++++++++++++++++++++++
        for epoch in range(1, self.config.n_epochs + 1):
            # We do not run an update during the first epoch, just so that we can see the initial prediction(s):
            update_gan = False
               
            if epoch > 1:
                update_gan = True
                
            outer_group_update = False
            if epoch % self.config.outer_update_epochs == 0:
                outer_group_update = True

            # Determine alpha scale for PCGrad alignment:
            alpha_scale = self.alpha_schedule(epoch,self.config.alpha_max,self.config.E_warm_up,self.config.E_ramp_up,self.config.E_hold,self.config.E_ramp_down)
           
            # Run a training step:
            current_results = self.train_step(
                env,
                opt,
                generator,
                generator_optimizer,
                discriminator_optimizer,
                update_outer_group=outer_group_update,
                loss_norm=loss_norm,
                do_update=update_gan,
                alpha_scale=alpha_scale
            )

            # Record time and register it with the loss history:
            t_end = MPI.Wtime()
            current_results["train_step_time"] = t_end - t_start

            # Register alpha scale:
            current_results["alpha_scale"] = alpha_scale

            # Run online analysis, i.e. monitor all relevant metrics:
            analysis.forward(
                epoch, current_results, None, None, True
            )
        # +++++++++++++++++++++++++++++

        # Clean up:
        self.grad_transport.clear()

        # And now write information to file:
        analysis.forward(generator=generator,data_idx=env.data_idx)