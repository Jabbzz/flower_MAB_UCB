"""Training and evaluation loops."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def set_global_seed(seed: int) -> None:
    """Seed all RNG sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    local_epochs: int,
    lr: float,
    device: torch.device,
) -> Tuple[float, float]:
    """Run local training and return (loss, accuracy)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for _ in range(local_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        return 0.0, 0.0
    return total_loss / total_examples, total_correct / total_examples


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model and return (loss, accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        return 0.0, 0.0
    return total_loss / total_examples, total_correct / total_examples
