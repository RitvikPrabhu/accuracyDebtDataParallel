from quantom_ips.utils.torch_sequential_model import TorchSequentialModel

import torch
from omegaconf import OmegaConf, MISSING
from dataclasses import dataclass, field
from quantom_ips.utils.registration import register_with_hydra


@dataclass
class TorchSequentialGeneratorDefaults:
    id: str = "TorchSequentialGenerator"
    input_shape: list = field(default_factory=lambda: [None, 100])
    layers: dict = MISSING


class TorchSequentialGenerator:
    def __init__(self, config, devices, dtype):
        self.id = config.id
        self.devices = devices
        self.dtype = dtype
        self.input_shape = OmegaConf.to_container(config.input_shape)

        self.model = TorchSequentialModel(config.layers, self.devices, self.dtype)
        self.parameters = self.model.parameters
        super().__init__()

    def forward(self, batch_size):
        self.input_shape[0] = batch_size
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=self.input_shape,
            device=self.devices,
            dtype=self.dtype,
        )
        return self.model.forward(noise)

    def predict(self, x, batch_size=None):
        return self.model.forward(x)
