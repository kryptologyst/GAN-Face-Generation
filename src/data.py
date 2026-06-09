"""Data utilities for GAN training."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


def load_mnist_data(batch_size: int = 64, img_size: int = 64) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    logger.info("Loaded MNIST: %d samples", len(dataset))
    return loader


def generate_synthetic_data(
    n_samples: int = 1000,
    img_size: int = 64,
) -> DataLoader:
    rng = np.random.RandomState(42)
    data = rng.rand(n_samples, 1, img_size, img_size).astype(np.float32)
    data = data * 2 - 1
    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    logger.info("Generated %d synthetic images", n_samples)
    return loader
