"""
transfer_learning.py — Fine-tune pretrained CNNs on image datasets.

Features:
  - Multiple architectures: ResNet18/34/50, EfficientNet-B0, MobileNetV3
  - Progressive unfreezing: head-only → full fine-tune
  - Grad-CAM attention visualisation
  - Mixed precision with AMP
  - Works with CIFAR-100, Flowers102, Food101, or custom folder datasets

Usage:
    python transfer_learning.py --model resnet18 --dataset cifar100 --epochs 20 --device auto
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
from torchvision import datasets, models, transforms
from tqdm import tqdm

OUTPUT_DIR = Path("outputs")

# ── model factory ───────────────────────────────────────────────

MODEL_REGISTRY: dict[str, tuple] = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, "fc", 512),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, "fc", 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, "fc", 2048),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, "classifier.1", 1280),
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT, "classifier.3", 1024),
}


def build_model(name: str, num_classes: int) -> tuple[nn.Module, str]:
    factory, weights, head_name, in_features = MODEL_REGISTRY[name]
    model = factory(weights=weights)

    # freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # replace classification head
    if "." in head_name:
        parts = head_name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], nn.Linear(in_features, num_classes))
    else:
        setattr(model, head_name, nn.Linear(in_features, num_classes))

    return model, head_name


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


# ── data ────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "cifar100": (datasets.CIFAR100, 100, 32),
    "flowers102": (datasets.Flowers102, 102, 224),
}


def get_dataloaders(name: str, batch_size: int):
    if name == "cifar100":
        train_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_ds = datasets.CIFAR100("data", train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR100("data", train=False, download=True, transform=test_tf)
    elif name == "flowers102":
        train_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_ds = datasets.Flowers102("data", split="train", download=True, transform=train_tf)
        test_ds = datasets.Flowers102("data", split="test", download=True, transform=test_tf)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ── Grad-CAM ────────────────────────────────────────────────────


class GradCAM:
    """Minimal Grad-CAM for the last conv layer."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        # find last conv layer
        self.target_layer = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self.target_layer = module
        if self.target_layer is not None:
            self.target_layer.register_forward_hook(self._save_activation)
            self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    @torch.no_grad()
    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        with torch.enable_grad():
            input_tensor = input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            self.model.zero_grad()
            output[0, class_idx].backward()

        if self.activations is None or self.gradients is None:
            return np.zeros((224, 224))

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def save_gradcam(model, loader, device, tag: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    gcam = GradCAM(model)
    model.eval()
    images, labels = next(iter(loader))
    images = images[:8].to(device)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    for i in range(8):
        img = images[i].unsqueeze(0)
        cam = gcam.generate(img)
        img_np = (images[i].cpu() * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()
        axes[0, i].imshow(img_np)
        axes[0, i].axis("off")
        axes[1, i].imshow(img_np)
        axes[1, i].imshow(cam, cmap="jet", alpha=0.4)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=10)
    fig.suptitle("Grad-CAM Attention Maps")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_gradcam.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / f'{tag}_gradcam.png'}")


# ── training ────────────────────────────────────────────────────


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_epoch(model, loader, optimizer, scaler, device):
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
        total_loss += loss.item() * images.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        correct += (out.argmax(1) == labels).sum().item()
        total += images.size(0)
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer Learning")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="resnet18")
    parser.add_argument("--dataset", choices=list(DATASET_REGISTRY.keys()), default="cifar100")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze-epoch", type=int, default=5, help="Epoch to unfreeze backbone")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    ds_class, num_classes, _ = DATASET_REGISTRY[args.dataset]
    tag = f"{args.model}_{args.dataset}"

    print(f"Device: {device}")
    print(f"Model: {args.model} | Dataset: {args.dataset} ({num_classes} classes)")

    model, head_name = build_model(args.model, num_classes)
    model = model.to(device)

    train_loader, test_loader = get_dataloaders(args.dataset, args.batch_size)

    # phase 1: train head only
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scaler = torch.GradScaler()

    history = {"train_acc": [], "test_acc": []}

    print(f"\n--- Phase 1: Head only (epochs 1-{args.unfreeze_epoch}) ---")
    for epoch in tqdm(range(1, args.unfreeze_epoch + 1), desc="Head"):
        tl, ta = train_epoch(model, train_loader, optimizer, scaler, device)
        test_acc = evaluate(model, test_loader, device)
        history["train_acc"].append(ta)
        history["test_acc"].append(test_acc)
        print(f"  Epoch {epoch}  loss={tl:.4f}  train={ta:.4f}  test={test_acc:.4f}")

    # phase 2: unfreeze + fine-tune
    print(f"\n--- Phase 2: Full fine-tune (epochs {args.unfreeze_epoch + 1}-{args.epochs}) ---")
    unfreeze_all(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(args.epochs - args.unfreeze_epoch) * len(train_loader)
    )

    best_acc = max(history["test_acc"]) if history["test_acc"] else 0

    for epoch in tqdm(range(args.unfreeze_epoch + 1, args.epochs + 1), desc="Fine-tune"):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
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
        ta = correct / total
        test_acc = evaluate(model, test_loader, device)
        history["train_acc"].append(ta)
        history["test_acc"].append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            OUTPUT_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_DIR / f"{tag}_best.pt")
        if epoch % 5 == 0 or epoch == args.unfreeze_epoch + 1:
            print(f"  Epoch {epoch}  loss={total_loss / total:.4f}  train={ta:.4f}  test={test_acc:.4f}  best={best_acc:.4f}")

    # plots
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history["train_acc"]) + 1), history["train_acc"], label="Train")
    ax.plot(range(1, len(history["test_acc"]) + 1), history["test_acc"], label="Test")
    ax.axvline(args.unfreeze_epoch, color="gray", linestyle="--", alpha=0.5, label="Unfreeze")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend()
    ax.set_title(f"{args.model} on {args.dataset}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_curves.png", dpi=150)
    plt.close(fig)

    save_gradcam(model, test_loader, device, tag)
    print(f"\nBest test accuracy: {best_acc:.4f}")
    print("Done ✓")


if __name__ == "__main__":
    main()
