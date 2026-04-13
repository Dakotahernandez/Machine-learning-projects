"""
mnist_cnn.py — Train a custom CNN on MNIST from scratch.

Features:
  - Custom Conv → BatchNorm → ReLU → MaxPool architecture
  - OneCycleLR for fast convergence
  - Training curves and sample prediction plots
  - Achieves 99%+ accuracy in ~5 epochs

Usage:
    python mnist_cnn.py --epochs 10 --device auto
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

OUTPUT_DIR = Path("outputs")


class MNISTNet(nn.Module):
    """Custom CNN for MNIST with batch normalisation."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * images.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        total_loss += F.cross_entropy(out, labels).item() * images.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def save_training_curves(history: dict, tag: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    fig.suptitle(f"MNIST CNN Training ({tag})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_curves.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def save_predictions(model, loader, device, tag: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    preds = model(images).argmax(1)
    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].cpu().squeeze(), cmap="gray")
        color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(f"{preds[i].item()}", color=color, fontsize=14)
        ax.axis("off")
    fig.suptitle("Sample Predictions (green=correct, red=wrong)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_predictions.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST CNN")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = MNISTNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr * 10, steps_per_epoch=len(train_loader), epochs=args.epochs
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    print(f"Training for {args.epochs} epochs on {len(train_ds):,} samples\n")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        tl, ta = train_epoch(model, train_loader, optimizer, scheduler, device)
        vl, va = evaluate(model, test_loader, device)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        print(f"  Epoch {epoch:2d}  loss={tl:.4f}  acc={ta:.4f}  val_loss={vl:.4f}  val_acc={va:.4f}")

    save_training_curves(history, "mnist")
    save_predictions(model, test_loader, device, "mnist")

    # save model
    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "mnist_cnn.pt")
    print(f"\nFinal test accuracy: {history['val_acc'][-1]:.4f}")
    print(f"Model saved to {OUTPUT_DIR / 'mnist_cnn.pt'}")
    print("Done ✓")


if __name__ == "__main__":
    main()
