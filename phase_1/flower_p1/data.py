"""Dataset utilities and deterministic client partitioning."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def _partition_indices(size: int, cid: int, num_clients: int) -> np.ndarray:
    all_indices = np.arange(size)
    return all_indices[cid::num_clients]


def _safe_dataset(
    task_name: str,
    train: bool,
    use_fake_data: bool,
    train_samples_per_client: int,
    eval_samples_per_client: int,
) -> Dataset:
    if task_name == "cifar10_resnet18":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        if use_fake_data:
            size = train_samples_per_client * 10 if train else eval_samples_per_client * 10
            return datasets.FakeData(
                size=size,
                image_size=(3, 32, 32),
                num_classes=10,
                transform=transform,
            )
        return datasets.CIFAR10(
            root="./datasets",
            train=train,
            download=True,
            transform=transform,
        )

    if task_name == "gtsrb_lenet":
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
            ]
        )
        if use_fake_data:
            size = train_samples_per_client * 10 if train else eval_samples_per_client * 10
            return datasets.FakeData(
                size=size,
                image_size=(3, 32, 32),
                num_classes=43,
                transform=transform,
            )
        split = "train" if train else "test"
        return datasets.GTSRB(
            root="./datasets",
            split=split,
            download=True,
            transform=transform,
        )

    raise ValueError(f"Unsupported task '{task_name}'")


def load_partition_data(
    task_name: str,
    cid: int,
    num_clients: int,
    batch_size: int,
    use_fake_data: bool,
    train_samples_per_client: int,
    eval_samples_per_client: int,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Load partitioned train/eval data for one client."""
    train_ds = _safe_dataset(
        task_name,
        train=True,
        use_fake_data=use_fake_data,
        train_samples_per_client=train_samples_per_client,
        eval_samples_per_client=eval_samples_per_client,
    )
    eval_ds = _safe_dataset(
        task_name,
        train=False,
        use_fake_data=use_fake_data,
        train_samples_per_client=train_samples_per_client,
        eval_samples_per_client=eval_samples_per_client,
    )

    train_idx = _partition_indices(len(train_ds), cid, num_clients)
    eval_idx = _partition_indices(len(eval_ds), cid, num_clients)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        Subset(train_ds, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=g,
    )
    eval_loader = DataLoader(
        Subset(eval_ds, eval_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, eval_loader


def load_global_eval_data(
    task_name: str,
    batch_size: int,
    use_fake_data: bool,
    train_samples_per_client: int,
    eval_samples_per_client: int,
) -> DataLoader:
    """Load server-side evaluation data."""
    eval_ds = _safe_dataset(
        task_name,
        train=False,
        use_fake_data=use_fake_data,
        train_samples_per_client=train_samples_per_client,
        eval_samples_per_client=eval_samples_per_client,
    )
    return DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)
