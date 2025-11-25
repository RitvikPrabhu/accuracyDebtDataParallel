import mpi4py
mpi4py.rc.thread_level = "serialized"
from mpi4py import MPI
import logging
import torch
import numpy as np
from typing import List, Any, Optional
from dataclasses import dataclass, field
from quantom_ips.utils.registration import register_with_hydra, make

logger = logging.getLogger(__name__)

defaults = [
    {"gradient_transport": "torch_arar"},
    "_self_",
]

@dataclass
class MGTrainerDefaults:
    id: str = "MGTrainer"
    defaults: List[Any] = field(default_factory=lambda: defaults)
    n_epochs: int = 30
    outer_update_epochs: int = 2
    gradient_scale: float = 1.0
    gen_lr: float = 1e-5
    gen_beta_1: float = 0.9
    gen_beta_2: float = 0.999
    gen_ema: float = -1.0
    disc_lr: float = 1e-4
    disc_beta_1: float = 0.9
    disc_beta_2: float = 0.999
    batch_size: int = 10
    
   
@register_with_hydra(
    group="trainer",
    defaults=MGTrainerDefaults,
    name="mg_trainer",
)
class MGTrainer:

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

    # Generator update:
    def generator_update(
        self,
        loss,
        generator,
        generator_optim,
        update_outer_group,
        do_update,
    ):
        if do_update:
            # Backpropagation:
            loss.backward(retain_graph=True)
            # Transprot gradients: 
            self.grad_transport.forward(
                generator, update_outer_group, gradient_scale=self.config.gradient_scale
            )
            
            # Update generator weights:
            generator_optim.step()
    
    # Discriminator update:
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

    # Full train step:
    def train_step(
        self,
        env,
        opt,
        generator,
        generator_optim,
        discriminator_optim,
        update_outer_group,
        loss_norm,
        do_update
    ):
        # Train the generator:

        # Reset optimizer first:
        generator_optim.zero_grad(set_to_none=True)
        # Determine parameters and determine loss:
        params = opt.forward(self.config.batch_size)
        gen_loss, fake_data = env.step(params)
        
        self.generator_update(
            gen_loss['fake'],
            generator,
            generator_optim,
            update_outer_group,
            do_update,
        )

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
        }
           
        return history
    
    # Now run the full show:
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
            )

            # Record time and register it with the loss history:
            t_end = MPI.Wtime()
            current_results["train_step_time"] = t_end - t_start

            # Run online analysis, i.e. monitor all relevant metrics:
            analysis.forward(
                epoch, self.config.n_epochs, current_results, generator, True
            )
        # +++++++++++++++++++++++++++++

        # Clean up:
        self.grad_transport.clear()

        # And now write information to file:
        analysis.forward()
