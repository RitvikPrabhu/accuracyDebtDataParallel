import torch

from dataclasses import dataclass, field
from quantom_ips.utils.registration import register_with_hydra


@dataclass
class ProxyOptimizerDefaults:
    id: str = "ProxyOptimizer"
    pdf_shape: list = field(default_factory=lambda: [10, 10])


@register_with_hydra(
    group="optimizer", name="parameter", defaults=ProxyOptimizerDefaults
)
class ProxyOptimizer:
    def __init__(self, config, devices, dtype):
        self.model = ProxyOptimizerModel(config, devices, dtype)

    def forward(self, n_solutions):
        solution = self.model.forward()
        return solution.expand(n_solutions, -1, -1).unsqueeze(1)

    def predict(self, x, batch_size=None):
        # This model doesn't take an input so we just call forward()
        return self.forward(x.shape[0])


class ProxyOptimizerModel(torch.nn.Module):
    def __init__(self, config, devices="cpu", dtype=torch.float32):
        super().__init__()

        self.dtype = dtype
        self.devices = devices
        self.pdf_shape = config.pdf_shape

        self.x = torch.linspace(
            0.001, 0.999, self.pdf_shape[0], dtype=self.dtype, device=self.devices
        )
        self.y = torch.linspace(
            0.001, 0.999, self.pdf_shape[1], dtype=self.dtype, device=self.devices
        )

        self.xx, self.yy = torch.meshgrid(self.x, self.y, indexing="ij")

        data = 0.5 * torch.ones((1, 5), dtype=self.dtype, device=self.devices)
        self.params = torch.nn.Parameter(data=data, requires_grad=True)

        self.p_min = torch.tensor(
            [-0.5, 2.75, 0.0, 3.0, 0.0], dtype=self.dtype, device=self.devices
        )
        self.p_max = torch.tensor(
            [1.0, 4.0, 1.3, 4.5, 1.5], dtype=self.dtype, device=self.devices
        )

    def A(self, p, x, y):
        return (
            torch.pow(x, p[0])
            * torch.pow((1.0 - x), p[1])
            * torch.pow(y, p[2])
            * torch.pow((1.0 - y), p[3])
            * (1.0 + p[4] * x * y)
        )

    def forward(self):
        p_scaled = (self.p_max - self.p_min) * self.params + self.p_min
        out = torch.vmap(lambda par: self.A(par, self.xx, self.yy), in_dims=0)(p_scaled)
        #        print(out.shape)
        return out
