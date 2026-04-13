"""
vae.py — Convolutional Variational Autoencoder on MNIST.

Features:
  - Conv encoder/decoder with reparameterisation trick
  - ELBO loss = reconstruction + KL divergence
  - Random sampling from latent space
  - Latent space interpolation grid
  - t-SNE visualisation coloured by digit class

Usage:
    python vae.py --epochs 30 --device auto
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm

OUTPUT_DIR = Path("outputs")
LATENT_DIM = 16


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)


def elbo_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, _ in loader:
        images = images.to(device)
        recon, mu, logvar = model(images)
        loss = elbo_loss(recon, images, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def save_samples(model, device) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    model.eval()
    samples = model.sample(64, device)
    grid = make_grid(samples, nrow=8)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    ax.axis("off")
    ax.set_title("VAE Random Samples")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "vae_samples.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / 'vae_samples.png'}")


@torch.no_grad()
def save_reconstructions(model, loader, device) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    model.eval()
    images, _ = next(iter(loader))
    images = images[:10].to(device)
    recon, _, _ = model(images)
    fig, axes = plt.subplots(2, 10, figsize=(14, 3))
    for i in range(10):
        axes[0, i].imshow(images[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].cpu().squeeze(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
    fig.suptitle("VAE Reconstructions")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "vae_reconstructions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / 'vae_reconstructions.png'}")


@torch.no_grad()
def save_latent_grid(model, device) -> None:
    """Sample a 2D grid from latent space (works best with latent_dim=2)."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    model.eval()
    n = 15
    grid_x = torch.linspace(-3, 3, n)
    grid_y = torch.linspace(-3, 3, n)
    canvas = []
    for yi in grid_y:
        row = []
        for xi in grid_x:
            z = torch.zeros(1, model.latent_dim, device=device)
            z[0, 0] = xi
            z[0, 1] = yi
            img = model.decoder(z).cpu().squeeze()
            row.append(img)
        canvas.append(torch.cat(row, dim=1))
    canvas = torch.cat(canvas, dim=0)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas.numpy(), cmap="gray")
    ax.axis("off")
    ax.set_title("VAE Latent Space Grid (dims 0 & 1)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "vae_latent_grid.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / 'vae_latent_grid.png'}")


@torch.no_grad()
def save_latent_tsne(model, loader, device) -> None:
    from sklearn.manifold import TSNE

    OUTPUT_DIR.mkdir(exist_ok=True)
    model.eval()
    all_mu, all_y = [], []
    for images, labels in loader:
        mu, _ = model.encoder(images.to(device))
        all_mu.append(mu.cpu().numpy())
        all_y.append(labels.numpy())
        if sum(len(y) for y in all_y) >= 5000:
            break
    Z = np.concatenate(all_mu)[:5000]
    Y = np.concatenate(all_y)[:5000]

    print("  Computing t-SNE ...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = tsne.fit_transform(Z)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=Y, cmap="tab10", s=5, alpha=0.7)
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.ax.set_yticklabels([str(i) for i in range(10)])
    ax.set_title("VAE Latent Space (t-SNE)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "vae_latent_tsne.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / 'vae_latent_tsne.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VAE on MNIST")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = VAE(args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"VAE: {total_params:,} parameters (latent_dim={args.latent_dim})\n")

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        avg_loss = train_epoch(model, train_loader, optimizer, device)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  ELBO loss={avg_loss:.2f}")

    save_samples(model, device)
    save_reconstructions(model, test_loader, device)
    save_latent_grid(model, device)
    save_latent_tsne(model, test_loader, device)

    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "vae_mnist.pt")
    print("\nDone ✓")


if __name__ == "__main__":
    main()
