"""DCGAN for face generation with latent space interpolation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class Generator(nn.Module):
    """DCGAN generator mapping latent vector to image."""

    def __init__(self, latent_dim: int = 100, img_channels: int = 1, feature_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            self._block(latent_dim, feature_dim * 8, 4, 1, 0),
            self._block(feature_dim * 8, feature_dim * 4, 4, 2, 1),
            self._block(feature_dim * 4, feature_dim * 2, 4, 2, 1),
            self._block(feature_dim * 2, feature_dim, 4, 2, 1),
            nn.ConvTranspose2d(feature_dim, img_channels, 4, 2, 1),
            nn.Tanh(),
        )

    @staticmethod
    def _block(in_c: int, out_c: int, kernel: int, stride: int, padding: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """DCGAN discriminator classifying real vs fake images."""

    def __init__(self, img_channels: int = 1, feature_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(feature_dim, feature_dim * 2, 4, 2, 1),
            self._block(feature_dim * 2, feature_dim * 4, 4, 2, 1),
            self._block(feature_dim * 4, feature_dim * 8, 4, 2, 1),
            nn.Conv2d(feature_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
        )

    @staticmethod
    def _block(in_c: int, out_c: int, kernel: int, stride: int, padding: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GANTrainer:
    """DCGAN training harness with loss tracking."""

    def __init__(
        self,
        latent_dim: int = 100,
        img_channels: int = 1,
        img_size: int = 64,
        feature_dim: int = 64,
        lr: float = 2e-4,
        device: str = "cpu",
    ) -> None:
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.device = device

        self.generator = Generator(latent_dim, img_channels, feature_dim).to(device)
        self.discriminator = Discriminator(img_channels, feature_dim).to(device)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCEWithLogitsLoss()

        self.g_losses: list[float] = []
        self.d_losses: list[float] = []

    def train_step(self, real_images: torch.Tensor) -> tuple[float, float]:
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        real_labels = torch.ones(batch_size, device=self.device)
        fake_labels = torch.zeros(batch_size, device=self.device)

        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        fake_images = self.generator(z)

        d_real = self.discriminator(real_images)
        d_fake = self.discriminator(fake_images.detach())
        d_loss = self.criterion(d_real, real_labels) + self.criterion(d_fake, fake_labels)

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        fake_images = self.generator(z)
        g_loss = self.criterion(self.discriminator(fake_images), real_labels)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return float(g_loss.item()), float(d_loss.item())

    def fit(
        self,
        dataloader: DataLoader,
        epochs: int = 50,
        log_interval: int = 10,
    ) -> None:
        for epoch in range(1, epochs + 1):
            g_epoch_loss = 0.0
            d_epoch_loss = 0.0
            n_batches = 0
            for real_images, in dataloader:
                g_loss, d_loss = self.train_step(real_images[0] if isinstance(real_images, list) else real_images)
                g_epoch_loss += g_loss
                d_epoch_loss += d_loss
                n_batches += 1
            avg_g = g_epoch_loss / max(n_batches, 1)
            avg_d = d_epoch_loss / max(n_batches, 1)
            self.g_losses.append(avg_g)
            self.d_losses.append(avg_d)
            if epoch % log_interval == 0:
                logger.info("Epoch %d/%d | G Loss: %.4f | D Loss: %.4f", epoch, epochs, avg_g, avg_d)

    def generate(self, n: int = 16) -> np.ndarray:
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, 1, 1, device=self.device)
            images = self.generator(z).cpu().numpy()
        return images

    def interpolate(self, steps: int = 8) -> np.ndarray:
        self.generator.eval()
        with torch.no_grad():
            z1 = torch.randn(1, self.latent_dim, 1, 1, device=self.device)
            z2 = torch.randn(1, self.latent_dim, 1, 1, device=self.device)
            alphas = np.linspace(0, 1, steps)
            images = []
            for alpha in alphas:
                z = (1 - alpha) * z1 + alpha * z2
                img = self.generator(z).cpu().numpy()[0, 0]
                images.append(img)
        return np.array(images)

    def save(self, path: str) -> None:
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "g_losses": self.g_losses,
                "d_losses": self.d_losses,
            },
            path,
        )
        logger.info("GAN saved to %s", path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.g_losses = checkpoint.get("g_losses", [])
        self.d_losses = checkpoint.get("d_losses", [])
        logger.info("GAN loaded from %s", path)
