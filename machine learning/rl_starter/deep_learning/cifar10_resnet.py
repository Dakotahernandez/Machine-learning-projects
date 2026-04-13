"""
cifar10_resnet.py — Custom ResNet for CIFAR-10 with mixed precision.

Features:
  - ResNet-style residual blocks built from scratch
  - Data augmentation (RandomCrop, HorizontalFlip, ColorJitter)
  - Cosine annealing LR schedule
  - Mixed precision training (AMP) for faster GPU training
  - Targets 92%+ accuracy

Usage:
    python cifar10_resnet.py --epochs 50 --device auto
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
CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class CIFARResNet(nn.Module):
    """ResNet-20-style network for CIFAR-10."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 3, stride=2)
        self.layer3 = self._make_layer(128, 256, 3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def get_dataloaders(batch_size: int):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10("data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(images)
            loss = F.cross_entropy(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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


def save_curves(history: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend()
    fig.suptitle("CIFAR-10 ResNet Training")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cifar10_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / 'cifar10_curves.png'}")


@torch.no_grad()
def save_predictions(model, loader, device) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images[:16].to(device), labels[:16].to(device)
    preds = model(images).argmax(1)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu() * std + mean
        ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
        color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(CLASSES[preds[i]], color=color, fontsize=10)
        ax.axis("off")
    fig.suptitle("CIFAR-10 Predictions (green=correct)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cifar10_predictions.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 ResNet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(args.batch_size)
    model = CIFARResNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"CIFARResNet: {total_params:,} parameters")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    scaler = torch.GradScaler()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        tl, ta = train_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        vl, va = evaluate(model, test_loader, device)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        if va > best_acc:
            best_acc = va
            OUTPUT_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_DIR / "cifar10_resnet_best.pt")
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  loss={tl:.4f}  acc={ta:.4f}  val_acc={va:.4f}  best={best_acc:.4f}")

    save_curves(history)
    save_predictions(model, test_loader, device)
    print(f"\nBest test accuracy: {best_acc:.4f}")
    print("Done ✓")


if __name__ == "__main__":
    main()
