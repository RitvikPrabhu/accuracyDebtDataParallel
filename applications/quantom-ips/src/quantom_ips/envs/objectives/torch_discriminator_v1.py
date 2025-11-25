from quantom_ips.utils.torch_sequential_model import TorchSequentialModel
from quantom_ips.utils.torch_nn_registry import torch_losses
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from omegaconf import MISSING
from dataclasses import dataclass, field


@dataclass
class TorchDiscriminatorDefaults:
    id: str = "TorchDiscriminator"
    layers: dict = MISSING
    label_noise: float = 0.0
    loss_fn: str = "MSE"
    normalize_inputs: bool = False
    use_log: bool = False
    observed_feature_range_min: list = field(default_factory=lambda: [])
    observed_feature_range_max: list = field(default_factory=lambda: [])
    scaled_feature_range_min: float = 0.0
    scaled_feature_range_max: float = 1.0


class TorchDiscriminator:
    def __init__(self, config, devices, dtype=torch.float32):
        self.id = config.id
        self.config = config
        self.dtype = dtype
        self.devices = devices

        self.real_label = 1.0
        self.fake_label = 0.0

        self.model = TorchSequentialModel(config.layers, self.devices, self.dtype)
        self.loss_fn = torch_losses[self.config.loss_fn]()
        self.use_log = config.use_log

        # Handle potential normalization of the input data:
        self.normalize_inputs = config.normalize_inputs
        self.min_scale_base = config.scaled_feature_range_min * torch.ones(
            size=(1, 1), dtype=self.dtype, device=self.devices
        )
        self.max_scale_base = config.scaled_feature_range_max * torch.ones(
            size=(1, 1), dtype=self.dtype, device=self.devices
        )

        self.min_observed_base = None
        self.max_observed_base = None

        if len(self.config.observed_feature_range_min) > 0:
            self.min_observed_base = torch.as_tensor(
                self.config.observed_feature_range_min,
                dtype=self.dtype,
                device=self.devices,
            )
        if len(self.config.observed_feature_range_max) > 0:
            self.max_observed_base = torch.as_tensor(
                self.config.observed_feature_range_max,
                dtype=self.dtype,
                device=self.devices,
            )

    def forward(self, x_gen, x_real, training=False):
        x_gen = x_gen.flatten(end_dim=-2)
        x_gen = self.preprocessing(x_gen)
        losses = {}
        if training:
            x_real = x_real.flatten(end_dim=-2)
            x_real = self.preprocessing(x_real)
            real_preds = self.predict(x_real)
            gen_preds = self.predict(x_gen)
            
            if self.config.loss_fn.lower() == "wasserstein":
               losses["real"] = torch.mean(real_preds)
               # Add gradient penalty, if a wasserstein GAN is used:
               losses["gradient_penalty"] = self.compute_gradient_penalty(x_real,x_gen)
            else:
               labels = torch.full(
                   size=(real_preds.shape[0], 1),
                   fill_value=self.real_label,
                   dtype=self.dtype,
                   device=self.devices,
               )
               labels -= self.config.label_noise * torch.rand(
                size=labels.shape, dtype=self.dtype, device=self.devices
               )
               losses["real"] = self.loss_fn(real_preds, labels)

            if self.config.loss_fn.lower() == "wasserstein":
                losses["generator"] = torch.mean(gen_preds)
            else:
                labels = torch.full(
                   size=(gen_preds.shape[0], 1),
                   fill_value=self.fake_label,
                   dtype=self.dtype,
                   device=self.devices,
                )
                labels += self.config.label_noise * torch.rand(
                   size=labels.shape, dtype=self.dtype, device=self.devices
                )
                losses["generator"] = self.loss_fn(gen_preds, labels)
        else:
            gen_preds = self.predict(x_gen)
            if self.config.loss_fn.lower() == "wasserstein":
                losses["generator"] = -torch.mean(gen_preds)
            else:
                labels = torch.full(
                   size=(gen_preds.shape[0], 1),
                   fill_value=self.real_label,
                   dtype=self.dtype,
                   device=self.devices,
                )
                labels -= self.config.label_noise * torch.rand(
                  size=labels.shape, dtype=self.dtype, device=self.devices
                )
                losses["generator"] = self.loss_fn(gen_preds, labels)

        return losses
    
    # Gradient penalty for the discriminator:
    def compute_gradient_penalty(self,x_real,x_fake):
        alpha = torch.rand(x_real.size(),device=self.devices)

        x_interpolated = alpha*x_real + (1-alpha)*x_fake
        x_interpolated = Variable(x_interpolated, requires_grad=True)
        y_interpolated = self.predict(x_interpolated)

        gradients = torch_grad(outputs=y_interpolated, inputs=x_interpolated,
                               grad_outputs=torch.ones(y_interpolated.size(),device=self.devices),
                               create_graph=True, retain_graph=True)[0]
        
        gradients = gradients.view(x_real.size()[0], -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        return torch.mean(gradients_norm)

    def normalize(self, x, x_min=None, x_max=None):
        if x_min is None:
            x_min = torch.min(x, dim=0).values
        if x_max is None:
            x_max = torch.max(x, dim=0).values

        min_scale = torch.repeat_interleave(self.min_scale_base, x.size()[1], dim=1)
        max_scale = torch.repeat_interleave(self.max_scale_base, x.size()[1], dim=1)

        x_std = (x - x_min) / (x_max - x_min)
        return (max_scale - min_scale) * x_std + min_scale
    
    def preprocessing(self,x):
        if self.use_log:
            x = torch.log(x+1e-7)
        if self.normalize_inputs:
            x = self.normalize(x, self.min_observed_base, self.max_observed_base)
        return x

    def predict(self, x):
        return self.model.forward(x)
