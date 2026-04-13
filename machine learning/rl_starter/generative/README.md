# Generative Models

From-scratch implementations of generative adversarial networks and variational autoencoders in PyTorch.

## Projects

### `dcgan.py` — Deep Convolutional GAN
- Generator and Discriminator with strided convolutions (Radford et al., 2016)
- Generates 64×64 images from random noise
- Tracks discriminator/generator loss balance
- Saves generated image grids every few epochs
- Interpolation in latent space (morphing between images)

### `vae.py` — Variational Autoencoder
- Convolutional VAE with reparameterisation trick
- KL divergence + reconstruction loss (ELBO)
- Latent space interpolation & sampling
- t-SNE visualisation of latent space by digit class

## Quick Start

```powershell
pip install -r requirements.txt

# Train DCGAN on MNIST (generates handwritten digits)
python dcgan.py --epochs 50 --device auto

# Train DCGAN on CelebA faces (requires manual download)
python dcgan.py --dataset celeba --epochs 25 --device auto

# Train VAE on MNIST
python vae.py --epochs 30 --device auto
```

## Sample Results

After training, check the `outputs/` folder for:
- `dcgan_grid_epoch_*.png` — Generated image grids at each checkpoint
- `dcgan_interpolation.png` — Smooth transitions between generated images
- `vae_samples.png` — Random samples from the VAE latent space
- `vae_reconstructions.png` — Original vs reconstructed images
- `vae_latent_tsne.png` — t-SNE of the latent space
