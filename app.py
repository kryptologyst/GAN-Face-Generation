"""Streamlit dashboard for GAN face generation."""

from __future__ import annotations

import streamlit as st

from src.data import generate_synthetic_data, load_mnist_data
from src.model import GANTrainer
from src.visualizer import plot_generated_grid, plot_interpolation, plot_losses

st.set_page_config(page_title="GAN Face Generator", layout="wide")
st.title("DCGAN Face Generator")
st.caption("Generative Adversarial Network for image synthesis")

tab1, tab2, tab3 = st.tabs(["Train", "Generate", "Interpolate"])

with tab1:
    st.header("Train GAN")
    col1, col2, col3 = st.columns(3)
    with col1:
        dataset = st.selectbox("Dataset", ["mnist", "synthetic"])
        epochs = st.slider("Epochs", 10, 200, 30, 10)
    with col2:
        latent_dim = st.slider("Latent Dim", 32, 256, 100, 32)
        lr = st.select_slider("Learning Rate", options=[1e-4, 2e-4, 5e-4, 1e-3], value=2e-4)
    with col3:
        device = st.selectbox("Device", ["cpu", "mps"], index=0)

    if st.button("Train GAN", type="primary"):
        loader = load_mnist_data() if dataset == "mnist" else generate_synthetic_data()
        trainer = GANTrainer(latent_dim=latent_dim, lr=lr, device=device)

        progress = st.progress(0)
        status = st.empty()

        for epoch in range(1, epochs + 1):
            g_epoch, d_epoch = 0.0, 0.0
            n_batches = 0
            for real_images, in loader:
                g_loss, d_loss = trainer.train_step(real_images[0] if isinstance(real_images, list) else real_images)
                g_epoch += g_loss
                d_epoch += d_loss
                n_batches += 1
            trainer.g_losses.append(g_epoch / max(n_batches, 1))
            trainer.d_losses.append(d_epoch / max(n_batches, 1))
            progress.progress(epoch / epochs)
            if epoch % 5 == 0:
                status.text(f"Epoch {epoch}: G={trainer.g_losses[-1]:.4f}, D={trainer.d_losses[-1]:.4f}")

        st.success("Training complete!")
        plot_losses(trainer.g_losses, trainer.d_losses, save_path="/tmp/gan_losses.png")
        st.image("/tmp/gan_losses.png")

        images = trainer.generate(16)
        plot_generated_grid(images, save_path="/tmp/gan_grid.png")
        st.image("/tmp/gan_grid.png")

with tab2:
    st.header("Generate Images")
    st.info("Use CLI to generate from a saved model: `python -m src.main generate --model-path gan.pt`")

with tab3:
    st.header("Latent Space Interpolation")
    st.info("Use CLI: `python -m src.main interpolate --model-path gan.pt --steps 8`")
    st.markdown("""
    **How it works**: Two random latent vectors are sampled and linearly interpolated.
    The generator produces images at each interpolation step, showing smooth transitions
    through the learned manifold.
    """)
