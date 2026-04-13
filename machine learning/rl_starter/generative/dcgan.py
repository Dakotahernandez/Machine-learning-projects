"""
dcgan.py — Deep Convolutional GAN (Radford et al., 2016).

Generates 64x64 images from random noise vectors.
Supports MNIST and CelebA datasets.

Features:
  - Strided conv generator + discriminator
  - Spectral normalisation on discriminator
  - Image grid snapshots every N epochs
  - Latent space interpolation

Usage:
    python dcgan.py --epochs 50 --device auto
    python dcgan.py --dataset celeba --epochs 25 --device auto
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm

OUTPUT_DIR = Path("outputs")
LATENT_DIM = 100


class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # latent_dim → 512 x 4 x 4
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 4 x 4 → 256 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 8 x 8 → 128 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 16 x 16 → 64 x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 32 x 32 → channels x 64 x 64
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # channels x 64 x 64 → 64 x 32 x 32
            nn.utils.spectral_norm(nn.Conv2d(channels, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # → 128 x 16 x 16
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # → 256 x 8 x 8
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # → 512 x 4 x 4
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # → 1 x 1 x 1
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def get_dataloader(dataset_name: str, batch_size: int):
    if dataset_name == "mnist":
        tf = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        ds = datasets.MNIST("data", train=True, download=True, transform=tf)
        channels = 1
    elif dataset_name == "celeba":
        tf = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        ds = datasets.CelebA("data", split="train", download=True, transform=tf)
        channels = 3
    elif dataset_name == "fashion":
        tf = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        ds = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
        channels = 1
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    return loader, channels


def save_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> None:
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_interpolation(gen, device, channels: int) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    gen.eval()
    z1 = torch.randn(1, LATENT_DIM, device=device)
    z2 = torch.randn(1, LATENT_DIM, device=device)
    alphas = torch.linspace(0, 1, 10, device=device).unsqueeze(1)
    z_interp = z1 * (1 - alphas) + z2 * alphas
    with torch.no_grad():
        imgs = gen(z_interp)
    save_grid(imgs, OUTPUT_DIR / "dcgan_interpolation.png", nrow=10)
    print(f"  Saved {OUTPUT_DIR / 'dcgan_interpolation.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DCGAN")
    parser.add_argument("--dataset", choices=["mnist", "fashion", "celeba"], default="mnist")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-every", type=int, default=5, help="Save image grid every N epochs")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    loader, channels = get_dataloader(args.dataset, args.batch_size)
    gen = Generator(LATENT_DIM, channels).to(device)
    disc = Discriminator(channels).to(device)

    # weight init
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    gen.apply(init_weights)
    disc.apply(init_weights)

    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(64, LATENT_DIM, device=device)

    g_params = sum(p.numel() for p in gen.parameters())
    d_params = sum(p.numel() for p in disc.parameters())
    print(f"Generator: {g_params:,}  Discriminator: {d_params:,}")
    print(f"Dataset: {args.dataset} ({channels}ch)  Epochs: {args.epochs}\n")

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        gen.train()
        disc.train()
        g_loss_sum, d_loss_sum, n_batches = 0.0, 0.0, 0

        for real, _ in loader:
            real = real.to(device)
            bs = real.size(0)

            # ── discriminator ──
            z = torch.randn(bs, LATENT_DIM, device=device)
            fake = gen(z).detach()
            d_real = disc(real)
            d_fake = disc(fake)
            d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # ── generator ──
            z = torch.randn(bs, LATENT_DIM, device=device)
            fake = gen(z)
            g_out = disc(fake)
            g_loss = criterion(g_out, torch.ones_like(g_out))
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()
            n_batches += 1

        avg_g = g_loss_sum / n_batches
        avg_d = d_loss_sum / n_batches

        if epoch % args.save_every == 0 or epoch == 1:
            gen.eval()
            with torch.no_grad():
                samples = gen(fixed_noise)
            OUTPUT_DIR.mkdir(exist_ok=True)
            save_grid(samples, OUTPUT_DIR / f"dcgan_grid_epoch_{epoch:03d}.png")
            print(f"  Epoch {epoch:3d}  G_loss={avg_g:.4f}  D_loss={avg_d:.4f}")

    save_interpolation(gen, device, channels)

    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.save(gen.state_dict(), OUTPUT_DIR / "dcgan_generator.pt")
    torch.save(disc.state_dict(), OUTPUT_DIR / "dcgan_discriminator.pt")
    print("Done ✓")


if __name__ == "__main__":
    main()
