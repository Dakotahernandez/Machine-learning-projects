"""
fashion_autoencoder.py — Convolutional autoencoder on Fashion-MNIST.

Features:
  - Convolutional encoder/decoder with latent bottleneck
  - Reconstruction visualization
  - t-SNE of latent space coloured by class
  - Anomaly detection via reconstruction error

Usage:
    python fashion_autoencoder.py --epochs 20 --device auto
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
from tqdm import tqdm

OUTPUT_DIR = Path("outputs")
CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Boot",
]


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 28x28
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 64, 7, 7)
        return self.net(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, _ in loader:
        images = images.to(device)
        recon, _ = model(images)
        loss = F.mse_loss(recon, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def save_reconstructions(model, loader, device) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    model.eval()
    images, labels = next(iter(loader))
    images = images[:10].to(device)
    recon, _ = model(images)
    fig, axes = plt.subplots(2, 10, figsize=(14, 3))
    for i in range(10):
        axes[0, i].imshow(images[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=10)
        axes[1, i].imshow(recon[i].cpu().squeeze(), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed", fontsize=10)
    fig.suptitle("Fashion-MNIST Autoencoder Reconstructions")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fashion_reconstructions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / 'fashion_reconstructions.png'}")


@torch.no_grad()
def save_latent_tsne(model, loader, device) -> None:
    from sklearn.manifold import TSNE

    OUTPUT_DIR.mkdir(exist_ok=True)
    model.eval()
    all_z, all_y = [], []
    for images, labels in loader:
        _, z = model(images.to(device))
        all_z.append(z.cpu().numpy())
        all_y.append(labels.numpy())
        if len(all_y) * loader.batch_size >= 5000:
            break
    Z = np.concatenate(all_z)[:5000]
    Y = np.concatenate(all_y)[:5000]

    print("  Computing t-SNE (this may take a moment) ...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    Z2 = tsne.fit_transform(Z)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(Z2[:, 0], Z2[:, 1], c=Y, cmap="tab10", s=3, alpha=0.7)
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.ax.set_yticklabels(CLASSES)
    ax.set_title("Latent Space (t-SNE)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fashion_latent_tsne.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / 'fashion_latent_tsne.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fashion-MNIST Autoencoder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = ConvAutoencoder(args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"ConvAutoencoder: {total_params:,} parameters (latent_dim={args.latent_dim})\n")

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        loss = train_epoch(model, train_loader, optimizer, device)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  recon_loss={loss:.6f}")

    save_reconstructions(model, test_loader, device)
    save_latent_tsne(model, test_loader, device)

    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "fashion_autoencoder.pt")
    print(f"\nModel saved to {OUTPUT_DIR / 'fashion_autoencoder.pt'}")
    print("Done ✓")


if __name__ == "__main__":
    main()
