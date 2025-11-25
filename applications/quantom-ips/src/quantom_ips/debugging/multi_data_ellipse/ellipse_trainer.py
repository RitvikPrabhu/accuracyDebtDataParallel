from quantom_ips.debugging.multi_data_ellipse.ellipse_sampler import EllipseSampler
from quantom_ips.debugging.multi_data_ellipse.torch_mlp import TorchMLP
from quantom_ips.gradient_transport.torch_arar import TorchARAR
import torch
import numpy as np

class EllipseTrainer:

    def __init__(self,config,dtype=torch.float32):
        self.config = config
        self.grad_transport = TorchARAR(config=self.config,dtype=dtype)
        self.torch_device = self.grad_transport.devices
        self.torch_dtype = dtype
        self.current_rank = self.grad_transport.rank
        self.comm = self.grad_transport.comm
        ema_alpha = config.ema_alpha
        
        self.data_idx = 1
        if self.current_rank % 2 == 0:
            self.data_idx = 0

        print(f"I am rank: {self.current_rank} and I am using: {self.data_idx}")
        self.sampler = EllipseSampler(torch_dtype=dtype,torch_device=self.torch_device)

        self.generator = TorchMLP(
            n_inputs=self.config.latent_dim,
            n_outputs=4,
            n_hidden=config.n_hidden_generator,
            n_neurons=config.n_neurons_generator,
            dropouts=config.dropouts_generator
        ).to(self.torch_device)
        self.gen_optim = torch.optim.Adam(self.generator.model.parameters(),lr=config.lr_generator)

        # Set up ema model (if requested):
        self.ema_generator = None
        if ema_alpha > 0.0:
            ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: ema_alpha * averaged_model_parameter + (1.0-ema_alpha) * model_parameter
            self.ema_generator = torch.optim.swa_utils.AveragedModel(self.generator, avg_fn=ema_avg)

        self.discriminator = TorchMLP(
            n_inputs=2,
            n_outputs=1,
            n_hidden=config.n_hidden_discriminator,
            n_neurons=config.n_neurons_discriminator,
            dropouts=config.dropouts_discriminator
        ).to(self.torch_device)
        self.disc_optim = torch.optim.Adam(self.discriminator.model.parameters(),lr=config.lr_discriminator)

        self.loss_fn = torch.nn.MSELoss()
        self.loss_norm = 0.25

    def normalize_data(self,data,x_min,x_max):
        if x_min is None:
          x_min = torch.min(data,0).values
        if x_max is None:
          x_max = torch.max(data,0).values

        return (data - x_min) / (x_max - x_min)

    def train_step_generator(self,update,update_outer_group,x_min,x_max):
        noise = torch.normal(mean=0.0,std=1.0,size=(self.config.batch_size,self.config.latent_dim),dtype=self.torch_dtype,device=self.torch_device)
        params = self.generator(noise)

        params_std = torch.std(params,0)
        # std_arg = torch.sum(params_std)
        # std_loss = (1.0-torch.exp(-std_arg))*self.config.std_loss_scale
        std_arg = 1.0-torch.exp(-params_std)
        std_loss = torch.sum(std_arg)*self.config.std_loss_scale

        x_fake = self.normalize_data(self.sampler.forward(params,self.config.n_samples,self.data_idx),x_min,x_max)
        y_fake = self.discriminator(x_fake)
        
        self.gen_optim.zero_grad(set_to_none=True)
        gen_loss = self.loss_fn(y_fake,torch.ones_like(y_fake))
        
        if update:
            gen_loss.backward(retain_graph=True)
            if self.config.std_loss_scale > 0.0:
              std_loss.backward()

            self.grad_transport.forward(
                self.generator, update_outer_group, gradient_scale=1.0
            )

            self.gen_optim.step()
            if self.ema_generator is not None:
                self.ema_generator.update_parameters(self.generator)

        return gen_loss, x_fake, std_loss
    
    
    def train_step_discriminator(self,x_real,x_fake,update):
        y_real = self.discriminator(x_real.detach())
        y_fake = self.discriminator(x_fake.detach())
        
        self.disc_optim.zero_grad(set_to_none=True)
        real_loss = self.loss_fn(y_real,torch.ones_like(y_real))
        fake_loss = self.loss_fn(y_fake,torch.zeros_like(y_fake))
        
        if update:
            loss = real_loss + fake_loss
            loss.backward()

            self.disc_optim.step()

        return real_loss, fake_loss
    
    def fit(self,data):

        gen_losses = []
        std_losses = []
        real_losses = []
        fake_losses = []

        global_x_min = None
        global_x_max = None

        if self.config.use_global_scale:
            global_x_min = torch.as_tensor(np.min(data,0),dtype=self.torch_dtype,device=self.torch_device)
            global_x_max = torch.as_tensor(np.max(data,0),dtype=self.torch_dtype,device=self.torch_device)

        for epoch in range(1,1+self.config.n_epochs):
            update_gan = False
            if epoch > 1:
                update_gan = True

            update_outer_group = False
            if epoch % self.config.outer_group_frequency == 0:
                update_outer_group = True

            idx = np.random.choice(data.shape[0],self.config.n_samples)
            x_real = self.normalize_data(torch.as_tensor(data[idx],dtype=self.torch_dtype,device=self.torch_device),global_x_min,global_x_max)

            gen_loss, x_fake, std_loss = self.train_step_generator(update_gan,update_outer_group,global_x_min,global_x_max)
            real_loss, fake_loss = self.train_step_discriminator(x_real,x_fake,update_gan)

            if epoch % self.config.read_frequency == 0:
                gen_losses.append(gen_loss.detach().cpu().numpy()/self.loss_norm)
                std_losses.append(std_loss.detach().cpu().numpy())
                real_losses.append(real_loss.detach().cpu().numpy()/self.loss_norm)
                fake_losses.append(fake_loss.detach().cpu().numpy()/self.loss_norm)

            if epoch % self.config.print_frequency == 0 and self.current_rank == 0:
                print(" ")
                print(f"Epoch: {epoch} / {self.config.n_epochs}")
                print(f"Real Loss: {real_losses[-1]}")
                print(f"Fake Loss: {fake_losses[-1]}")
                print(f"Gen. Loss: {gen_losses[-1]}")
                print(f"Std. Loss: {std_losses[-1]}")

            
        return {
            'gen_loss': np.array(gen_losses),
            'real_loss':np.array(real_losses),
            'fake_loss':np.array(fake_losses),
            'std_loss':np.array(std_losses)
        }






    



