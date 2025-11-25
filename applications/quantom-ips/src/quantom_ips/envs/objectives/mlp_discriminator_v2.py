from quantom_ips.envs.objectives.torch_discriminator_v1 import (
    TorchDiscriminator,
    TorchDiscriminatorDefaults,
)
from quantom_ips.utils.registration import register_with_hydra
from dataclasses import dataclass, field


@dataclass
class MLPDiscriminatorV2Defaults(TorchDiscriminatorDefaults):
    id: str = "MLPDiscriminatorV2"
    layers: dict = field(
        default_factory=lambda: {
           "LinearStack": {
               "index": 0,
               "class": "LinearStack",
               "config": {
                   "n_inputs": 2,
                   "n_neurons": 128,
                   "n_layers": 4,
                   "dropout": 0.0,
                   "activation": {
                        "class": "LeakyReLU",
                        "config": {"negative_slope": 0.2},
                    },
               },           
           },
            "Linear_1": {
                "index": 1,
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
    defaults=MLPDiscriminatorV2Defaults,
    name="MLPDiscriminatorV2",
)
class MLPDiscriminatorV2(TorchDiscriminator):
    pass
