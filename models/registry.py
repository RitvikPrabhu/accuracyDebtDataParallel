from dataclasses import dataclass
from typing import Callable, Dict, Iterable

import torch.nn as nn
import torch.optim as optim

from .convex_logreg import ConvexLogReg
from .bn_cnn_debug import BNCNNDebug
from .resnet18_cifar import ResNet18CIFAR


@dataclass
class ModelSpec:
    """Model recipe stored in the registry."""

    name: str
    build_model: Callable[[], nn.Module]
    build_loss: Callable[[], nn.Module]
    build_optimizer: Callable[[Iterable[nn.Parameter], float], optim.Optimizer]
    default_lr: float
    description: str


def _sgd_no_momentum(params, lr: float):
    return optim.SGD(params, lr=lr, momentum=0.0, weight_decay=0.0)


def _sgd_momentum(params, lr: float):
    return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "convex_logreg": ModelSpec(
        name="convex_logreg",
        build_model=lambda: ConvexLogReg(),
        build_loss=lambda: nn.CrossEntropyLoss(),
        build_optimizer=_sgd_no_momentum,
        default_lr=0.1,
        description="Multiclass logistic regression on CIFAR-10 (convex baseline).",
    ),
    "bn_cnn_debug": ModelSpec(
        name="bn_cnn_debug",
        build_model=lambda: BNCNNDebug(),
        build_loss=lambda: nn.CrossEntropyLoss(),
        build_optimizer=_sgd_momentum,
        default_lr=0.01,
        description="Small CNN with BatchNorm layers to expose BN/SyncBN issues.",
    ),
    "resnet18_cifar": ModelSpec(
        name="resnet18_cifar",
        build_model=lambda: ResNet18CIFAR(),
        build_loss=lambda: nn.CrossEntropyLoss(),
        build_optimizer=_sgd_momentum,
        default_lr=0.01,
        description="ResNet-18 backbone adapted to CIFAR-10.",
    ),
}


def list_models():
    """Return a sorted list of available model names."""
    return sorted(MODEL_REGISTRY.keys())


def get_model_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {', '.join(list_models())}"
        )
    return MODEL_REGISTRY[name]


def build_model_from_name(name: str) -> nn.Module:
    spec = get_model_spec(name)
    return spec.build_model()


def build_loss_from_name(name: str) -> nn.Module:
    spec = get_model_spec(name)
    return spec.build_loss()


def build_optimizer_from_name(name: str, params, lr: float = None) -> optim.Optimizer:
    spec = get_model_spec(name)
    effective_lr = lr if lr is not None else spec.default_lr
    return spec.build_optimizer(params, effective_lr)
