# GAN Face Generation

**DCGAN** for generating synthetic images with latent space interpolation.

## Overview

- Deep Convolutional GAN (DCGAN) architecture
- Generator and Discriminator with batch normalization
- Latent space interpolation for smooth transitions
- Training loss visualization
- **Streamlit dashboard** with live training

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
# CLI: python -m src.main train --dataset mnist --epochs 50
# CLI: python -m src.main generate --model-path gan.pt --n 16
# CLI: python -m src.main interpolate --model-path gan.pt
pytest tests/ -v
```

## Docker

```bash
docker compose up --build
```

## License

MIT
# GAN-Face-Generation
