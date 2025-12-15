import torch.nn as nn


class ConvexLogReg(nn.Module):
    """Multiclass logistic regression on flattened 32x32x3 inputs.

    This is a convex model in the parameters, useful as a theory-friendly
    baseline for convergence experiments.
    """

    def __init__(self, in_dim: int = 32 * 32 * 3, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
