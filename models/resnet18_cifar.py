import torch.nn as nn
from torchvision import models


class ResNet18CIFAR(nn.Module):
    """ResNet-18 adapted to CIFAR-10 (32x32 RGB inputs, 10 classes)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        backbone = models.resnet18(pretrained=False)
        # Replace final fully connected layer
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.model = backbone

    def forward(self, x):
        return self.model(x)
