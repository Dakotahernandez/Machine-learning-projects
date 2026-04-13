"""
transformer_forecaster.py — Transformer encoder for time series forecasting.

Uses positional encoding + multi-head self-attention for
sequence-to-one prediction on time series data.

Usage:
    python transformer_forecaster.py --epochs 30 --device auto
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

OUTPUT_DIR = Path("outputs")


def generate_synthetic(n: int = 2000) -> np.ndarray:
    t = np.arange(n, dtype=np.float32)
    return 0.01 * t + np.sin(2 * np.pi * t / 50) + 0.5 * np.sin(2 * np.pi * t / 120) + 0.3 * np.random.randn(n).astype(np.float32)


class MinMaxScaler:
    def __init__(self):
        self.min_, self.max_ = 0.0, 1.0

    def fit_transform(self, data):
        self.min_, self.max_ = float(data.min()), float(data.max())
        return (data - self.min_) / (self.max_ - self.min_ + 1e-8)

    def inverse_transform(self, data):
        return data * (self.max_ - self.min_) + self.min_


class TSDataset(Dataset):
    def __init__(self, data, window):
        self.data = torch.from_numpy(data).float()
        self.window = window

    def __len__(self):
        return len(self.data) - self.window

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window].unsqueeze(1)
        y = self.data[idx + self.window].unsqueeze(0)
        return x, y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def predict_all(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    preds = []
    for x, _ in loader:
        preds.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer Time Series Forecaster")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    raw = generate_synthetic(2000)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(raw)

    train_size = int(len(data) * 0.8)
    train_ds = TSDataset(data[:train_size], args.window)
    full_ds = TSDataset(data, args.window)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = TransformerForecaster().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"TransformerForecaster: {total_params:,} parameters\n")

    losses = []
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        losses.append(loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  MSE={loss:.6f}")

    predictions = predict_all(model, full_ds, device)
    actual_inv = scaler.inverse_transform(raw)
    pred_inv = scaler.inverse_transform(predictions.flatten())

    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual_inv, label="Actual", alpha=0.7)
    pred_x = np.arange(args.window, args.window + len(pred_inv))
    ax.plot(pred_x, pred_inv, label="Transformer", alpha=0.8)
    ax.axvline(train_size, color="gray", linestyle="--", alpha=0.5, label="Train/Test split")
    ax.set_title("Transformer Time Series Forecast")
    ax.set_xlabel("Time step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "transformer_forecast.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.set_title("Training Loss")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "transformer_loss.png", dpi=150)
    plt.close(fig)

    print(f"\n  Saved {OUTPUT_DIR / 'transformer_forecast.png'}")
    print("Done ✓")


if __name__ == "__main__":
    main()
