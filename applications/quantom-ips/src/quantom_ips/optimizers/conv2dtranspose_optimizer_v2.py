from .torch_sequential_generator import (
    TorchSequentialGenerator,
    TorchSequentialGeneratorDefaults,
)
from quantom_ips.utils.registration import register_with_hydra

from dataclasses import dataclass, field


@dataclass
class Conv2DTransposeOptimizerV2Defaults(TorchSequentialGeneratorDefaults):
    layers: dict = field(
        default_factory=lambda: {
            "Linear_1": {
                "index": 0,
                "class": "Linear",
                "config": {
                    "bias": True,
                    "in_features": 100,
                    "out_features": 100,
                },
                "weight_init": "kaiming_normal",
                "bias_init": "normal",
                "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
            },
            "Linear_2": {
                "index": 1,
                "class": "Linear",
                "config": {
                    "bias": True,
                    "in_features": 100,
                    "out_features": 100,
                },
                "weight_init": "kaiming_normal",
                "bias_init": "normal",
                "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
            },
            "Linear_3": {
                "index": 2,
                "class": "Linear",
                "config": {
                    "bias": True,
                    "in_features": 100,
                    "out_features": 100,
                },
                "weight_init": "kaiming_normal",
                "bias_init": "normal",
                "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
            },
            "Unflatten": {
                "index": 3,
                "class": "Unflatten",
                "config": {"dim": 1, "unflattened_size": [100, 1, 1]},
            },
            "ConvTranspose2dStack": {
                "index": 4,
                "class": "ConvTranspose2dStack",
                "config": {
                    "in_channels": 100,
                    "out_channels": 1,
                    "kernel_size": 3,
                    "internal_channels": 100,
                    "n_layers": 4,
                    "activation": {
                        "class": "LeakyReLU",
                        "config": {"negative_slope": 0.2},
                    },
                    "weight_init": "kaiming_normal",
                    "bias_init": "normal",
                    "out_activation": "Sigmoid",
                    "weight_out_init": "xavier_normal",
                    "bias_out_init": "normal",
                },
            },
            "Upsample": {
                "index": 5,
                "class": "Upsample",
                "config": {"mode": "bilinear", "size": [10, 10]},
            },
        }
    )
    id: str = "Conv2DTransposeOptimizerV2"


@register_with_hydra(
    group="optimizer", defaults=Conv2DTransposeOptimizerV2Defaults, name="conv2D_v2"
)
class Conv2DTransposeOptimizerV2(TorchSequentialGenerator):
    pass
