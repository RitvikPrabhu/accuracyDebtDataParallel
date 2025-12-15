import torch.nn as nn
import torch.nn.functional as F


class BNCNNDebug(nn.Module):
    """Small CNN with BatchNorm to stress BN behavior in distributed training."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.head = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32 -> 16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16 -> 8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8 -> 4
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x
