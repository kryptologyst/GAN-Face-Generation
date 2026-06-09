"""Visualization utilities for GAN outputs."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_generated_grid(
    images: np.ndarray,
    n_rows: int = 4,
    n_cols: int = 4,
    save_path: str | None = None,
) -> None:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            img = images[i]
            if img.shape[0] == 1:
                ax.imshow(img[0], cmap="gray")
            else:
                ax.imshow(np.transpose(img, (1, 2, 0)))
        ax.axis("off")
    fig.suptitle("Generated Images", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Generated grid saved to %s", save_path)
    plt.close(fig)


def plot_interpolation(
    images: np.ndarray,
    save_path: str | None = None,
) -> None:
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(f"α={i / max(n - 1, 1):.2f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Latent Space Interpolation", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Interpolation plot saved to %s", save_path)
    plt.close(fig)


def plot_losses(
    g_losses: list[float],
    d_losses: list[float],
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(g_losses, label="Generator Loss", color="steelblue")
    ax.plot(d_losses, label="Discriminator Loss", color="coral")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GAN Training Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Loss plot saved to %s", save_path)
    plt.close(fig)
