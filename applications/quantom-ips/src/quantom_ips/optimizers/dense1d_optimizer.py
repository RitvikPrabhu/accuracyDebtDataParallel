from .torch_sequential_generator import (
    TorchSequentialGenerator,
    TorchSequentialGeneratorDefaults,
)
from quantom_ips.utils.registration import register_with_hydra

from dataclasses import dataclass, field


@dataclass
class Dense1DOptimizerDefaults(TorchSequentialGeneratorDefaults):
    layers: dict = field(
        default_factory=lambda: {
            "Linear_1": {
                "index": 0,
                "class": "Linear",
                "config": {
                    "bias": True,
                    "in_features": 100,
                    "out_features": 128,
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
                    "in_features": 128,
                    "out_features": 128,
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
                    "in_features": 128,
                    "out_features": 128,
                },
                "weight_init": "kaiming_normal",
                "bias_init": "normal",
                "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
            },
            "Linear_4": {
                "index": 3,
                "class": "Linear",
                "config": {
                    "bias": True,
                    "in_features": 128,
                    "out_features": 128,
                },
                "weight_init": "kaiming_normal",
                "bias_init": "normal",
                "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
            },
            "Linear_5": {
                "index": 4,
                "class": "Linear",
                "config": {
                    "bias": True,
                    "in_features": 128,
                    "out_features": 10,
                },
                "weight_init": "xavier_normal",
                "bias_init": "normal",
                "activation": {"class": "Sigmoid"},
            },
            "Unflatten": {
                "index": 5,
                "class": "Unflatten",
                "config": {"dim": 1, "unflattened_size": [1, 10]},
            },
        }
    )
    id: str = "Dense1DOptimizer"


@register_with_hydra(
    group="optimizer", defaults=Dense1DOptimizerDefaults, name="dense1D"
)
class Dense1DOptimizer(TorchSequentialGenerator):
    pass
