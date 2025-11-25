import torch
import torchvision as tv
from core.registry import MODELS

@MODELS.register("resnet50_torch")
class ResNet50Factory:
    def __init__(self, pretrained=False):
        self.pretrained = pretrained
    def build(self, num_classes: int):
        m = tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT) if self.pretrained else tv.models.resnet50()
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        return m
