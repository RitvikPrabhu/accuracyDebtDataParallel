import torch
from quantom_ips.utils.registration import register_with_hydra
from dataclasses import dataclass


@dataclass
class IdentitySampleTransformerDefaults:
    id: str = "IdentitySampleTransformer"


@register_with_hydra(
    group=[
        "environment/experiment",
        "environment/event_filter",
        "environment/preprocessor",
    ],
    defaults=IdentitySampleTransformerDefaults,
    name="identity",
)
class IdentitySampleTransformer:
    def __init__(self, config, devices="cpu", dtype=torch.float32):
        # Since this module doesn't do anything
        # we don't save any config or device info
        pass

    def forward(self, data):
        return data
