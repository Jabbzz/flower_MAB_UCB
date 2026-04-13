"""Model definitions for Phase 1 tasks."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class LeNetGTSRB(nn.Module):
    """LeNet-style model for 32x32 traffic-sign images."""

    def __init__(self, num_classes: int = 43) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_model(task_name: str) -> nn.Module:
    """Return a model compatible with the chosen benchmark task."""
    if task_name == "cifar10_resnet18":
        return resnet18(weights=None, num_classes=10)
    if task_name == "gtsrb_lenet":
        return LeNetGTSRB(num_classes=43)
    raise ValueError(f"Unsupported task '{task_name}'")
