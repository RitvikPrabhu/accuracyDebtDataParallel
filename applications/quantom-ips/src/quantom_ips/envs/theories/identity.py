import torch
from quantom_ips.utils.registration import register_with_hydra
from dataclasses import dataclass, field


@dataclass
class IdentityTheoryDefaults:
    id: str = "IdentityTheory"
    grid_minimums: list = field(default_factory=lambda: [0.0])
    grid_maximums: list = field(default_factory=lambda: [1.0])
    average: bool = True


@register_with_hydra(
    group="environment/theory", defaults=IdentityTheoryDefaults, name="identity"
)
class IdentityTheory:
    def __init__(self, config, devices="cpu", dtype=torch.float32):
        self.devices = devices
        self.dtype = dtype

        self.grid_minimums = config.grid_minimums
        self.grid_maximums = config.grid_maximums
        self.average = config.average

    def forward(self, data):
        grid = self.create_grid(data.shape[2:])
        loss = self.get_loss(data, grid)
        return self.transform(data, grid), grid, loss

    def get_loss(self, data, grid):
        return {}

    def transform(self, data, grid):
        if self.average:
            return data.mean(dim=0, keepdim=True)
        return data

    def create_grid(self, dims):
        if len(self.grid_maximums) == 1:
            grid_maximums = [self.grid_maximums[0] for _ in dims]
        else:
            grid_maximums = self.grid_maximums
        if len(self.grid_minimums) == 1:
            grid_minimums = [self.grid_minimums[0] for _ in dims]
        else:
            grid_minimums = self.grid_minimums

        grid = []
        for d, min, max in zip(dims, grid_minimums, grid_maximums):
            grid.append(
                torch.linspace(min, max, d, device=self.devices, dtype=self.dtype)
            )

        return grid
