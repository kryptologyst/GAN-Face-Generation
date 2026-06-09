"""CLI for GAN face generation."""

from __future__ import annotations

import logging

import typer

from src.data import generate_synthetic_data, load_mnist_data
from src.model import GANTrainer
from src.visualizer import plot_generated_grid, plot_interpolation, plot_losses

app = typer.Typer(help="GAN Face Generation CLI")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@app.command()
def train(
    dataset: str = typer.Option("mnist", help="Dataset: mnist, synthetic"),
    epochs: int = typer.Option(50, help="Training epochs"),
    latent_dim: int = typer.Option(100, help="Latent vector dimension"),
    lr: float = typer.Option(2e-4, help="Learning rate"),
    device: str = typer.Option("cpu", help="Device: cpu, cuda, mps"),
    save_model: str | None = typer.Option(None, help="Save trained GAN"),
    save_plot: str | None = typer.Option(None, help="Save loss plot"),
) -> None:
    if dataset == "mnist":
        loader = load_mnist_data()
    else:
        loader = generate_synthetic_data()

    trainer = GANTrainer(latent_dim=latent_dim, device=device, lr=lr)
    trainer.fit(loader, epochs=epochs)

    logger.info("Training complete")

    if save_model:
        trainer.save(save_model)
    if save_plot:
        plot_losses(trainer.g_losses, trainer.d_losses, save_path=save_plot)


@app.command()
def generate(
    model_path: str = typer.Option(..., help="Path to saved GAN model"),
    n: int = typer.Option(16, help="Number of images to generate"),
    device: str = typer.Option("cpu", help="Device"),
    save_plot: str | None = typer.Option(None, help="Save generated grid"),
) -> None:
    trainer = GANTrainer(device=device)
    trainer.load(model_path)
    images = trainer.generate(n=n)
    if save_plot:
        plot_generated_grid(images, save_path=save_plot)
    logger.info("Generated %d images", n)


@app.command()
def interpolate(
    model_path: str = typer.Option(..., help="Path to saved GAN model"),
    steps: int = typer.Option(8, help="Interpolation steps"),
    device: str = typer.Option("cpu", help="Device"),
    save_plot: str | None = typer.Option(None, help="Save interpolation plot"),
) -> None:
    trainer = GANTrainer(device=device)
    trainer.load(model_path)
    images = trainer.interpolate(steps=steps)
    if save_plot:
        plot_interpolation(images, save_path=save_plot)
    logger.info("Generated %d interpolation steps", steps)


if __name__ == "__main__":
    app()
