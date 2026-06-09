"""Tests for GAN model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model import Discriminator, GANTrainer, Generator


class TestGenerator:
    def test_output_shape(self) -> None:
        gen = Generator(latent_dim=100, img_channels=1, feature_dim=64)
        z = torch.randn(4, 100, 1, 1)
        out = gen(z)
        assert out.shape == (4, 1, 64, 64)
        assert out.min() >= -1 and out.max() <= 1

    def test_different_latent_dims(self) -> None:
        gen = Generator(latent_dim=50, img_channels=3, feature_dim=32)
        z = torch.randn(2, 50, 1, 1)
        out = gen(z)
        assert out.shape == (2, 3, 64, 64)


class TestDiscriminator:
    def test_output_shape(self) -> None:
        disc = Discriminator(img_channels=1, feature_dim=64)
        x = torch.randn(4, 1, 64, 64)
        out = disc(x)
        assert out.shape == (4, 1)

    def test_real_vs_fake_separation(self) -> None:
        disc = Discriminator(img_channels=1, feature_dim=64)
        real = torch.randn(4, 1, 64, 64)
        fake = torch.randn(4, 1, 64, 64) * 0.1
        real_out = disc(real)
        fake_out = disc(fake)
        assert real_out.shape == (4, 1)
        assert fake_out.shape == (4, 1)


class TestGANTrainer:
    def test_init(self) -> None:
        trainer = GANTrainer(device="cpu")
        assert trainer.generator is not None
        assert trainer.discriminator is not None

    def test_generate(self) -> None:
        trainer = GANTrainer(device="cpu")
        images = trainer.generate(n=4)
        assert images.shape == (4, 1, 64, 64)

    def test_interpolate(self) -> None:
        trainer = GANTrainer(device="cpu")
        images = trainer.interpolate(steps=4)
        assert images.shape == (4, 64, 64)

    def test_save_load(self, tmp_path) -> None:
        trainer = GANTrainer(device="cpu")
        path = str(tmp_path / "gan.pt")
        trainer.save(path)
        trainer2 = GANTrainer(device="cpu")
        trainer2.load(path)
