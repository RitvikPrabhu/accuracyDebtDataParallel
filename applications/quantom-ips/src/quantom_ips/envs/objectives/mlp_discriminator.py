from quantom_ips.envs.objectives.torch_discriminator_v1 import (
    TorchDiscriminator,
    TorchDiscriminatorDefaults,
)
from quantom_ips.utils.registration import register_with_hydra
from dataclasses import dataclass, field


@dataclass
class MLPDiscriminatorDefaults(TorchDiscriminatorDefaults):
    id: str = "MLPDiscriminator"
    layers: dict = field(
        default_factory=lambda: {
            "Linear_1": {
                "index": 0,
                "class": "Linear",
                "config": {"in_features": 2, "out_features": 128},
                "weight_init": "kaiming_normal",
                "bias_init": "normal",
                "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
            },
            "Linear_2": {
                "index": 1,
                "class": "Linear",
                "config": {"in_features": 128, "out_features": 128},
                "weight_init": "kaiming_normal",
                "bias_init": "normal",
                "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
            },
            "Linear_3": {
                "index": 2,
                "class": "Linear",
                "config": {"in_features": 128, "out_features": 128},
                "weight_init": "kaiming_normal",
                "bias_init": "normal",
                "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
            },
            "Linear_4": {
                "index": 3,
                "class": "Linear",
                "config": {"in_features": 128, "out_features": 128},
                "weight_init": "kaiming_normal",
                "bias_init": "normal",
                "activation": {"class": "LeakyReLU", "config": {"negative_slope": 0.2}},
            },
            "Linear_5": {
                "index": 4,
                "class": "Linear",
                "config": {"in_features": 128, "out_features": 1},
                "weight_init": "xavier_normal",
                "bias_init": "normal",
                "activation": {"class": "Sigmoid"},
            },
        }
    )


@register_with_hydra(
    group="environment/objective",
    defaults=MLPDiscriminatorDefaults,
    name="MLPDiscriminator",
)
class MLPDiscriminator(TorchDiscriminator):
    pass
