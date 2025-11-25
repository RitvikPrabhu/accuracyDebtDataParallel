import torch
import torchvision as tv
from core.registry import DATASETS

@DATASETS.register("cifar10_torch")
class CIFAR10Factory:
    def __init__(self, root="~/.cache/cifar10", augment=None, pin_memory=True, shuffle=True):
        self.root = root
        self.augment = augment or {}
        self.pin_memory = pin_memory
        self.shuffle = shuffle

    def make(self, split: str, batch_size: int, num_workers: int, **kwargs):
        tfms = []
        if self.augment.get("random_crop", True):
            tfms += [tv.transforms.RandomCrop(32, padding=4)]
        if self.augment.get("random_flip", True):
            tfms += [tv.transforms.RandomHorizontalFlip()]
        tfms += [tv.transforms.ToTensor()]
        transform = tv.transforms.Compose(tfms)
        train = split == "train"
        ds = tv.datasets.CIFAR10(root=self.root, train=train, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=self.shuffle if train else False,
            num_workers=num_workers, pin_memory=self.pin_memory
        )
        return loader, 10
