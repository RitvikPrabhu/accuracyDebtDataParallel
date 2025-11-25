from .torch_sequential_generator import (
    TorchSequentialGenerator,
    TorchSequentialGeneratorDefaults,
)
from quantom_ips.utils.registration import register_with_hydra

from dataclasses import dataclass, field


@dataclass
class Conv2DTransposeOptimizerDefaults(TorchSequentialGeneratorDefaults):
    layers: dict = field(
        default_factory=lambda: {
            "Linear": {
                "index": 0,
                "class": "Linear",
                "config": {
                    "bias": True,
                    "in_features": 100,
                    "out_features": 200,
                },
                "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
            },
            "Unflatten": {
                "index": 1,
                "class": "Unflatten",
                "config": {"dim": 1, "unflattened_size": [200, 1, 1]},
            },
            "ConvTranspose2dStack": {
                "index": 2,
                "class": "ConvTranspose2dStack",
                "config": {
                    "in_channels": 200,
                    "out_channels": 1,
                    "kernel_size": 3,
                    "internal_channels": 100,
                    "n_layers": 4,
                    "activation": {
                        "class": "LeakyReLU",
                        "config": {"negative_slope": 0.2},
                    },
                    "out_activation": "Sigmoid",
                },
            },
            "Upsample": {
                "index": 3,
                "class": "Upsample",
                "config": {"mode": "bilinear", "size": [10, 10]},
            },
        }
    )
    id: str = "Conv2DTransposeOptimizer"


@register_with_hydra(
    group="optimizer", defaults=Conv2DTransposeOptimizerDefaults, name="conv2D"
)
class Conv2DTransposeOptimizer(TorchSequentialGenerator):
    pass
