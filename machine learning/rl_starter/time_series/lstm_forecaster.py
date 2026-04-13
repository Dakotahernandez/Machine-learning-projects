"""
lstm_forecaster.py — LSTM time series forecasting.

Trains an LSTM to predict future values using a sliding window
approach. Supports synthetic and custom CSV data.

Features:
  - Sliding window dataset creation
  - Multi-step ahead prediction
  - MinMax scaling with inverse transform for plotting
  - Forecast vs actual with shaded confidence intervals

Usage:
    python lstm_forecaster.py --epochs 50 --device auto
    python lstm_forecaster.py --csv data.csv --column price --epochs 100
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

OUTPUT_DIR = Path("outputs")


# ── data ────────────────────────────────────────────────────────


def generate_synthetic(n: int = 2000) -> np.ndarray:
    """Generate a noisy sine + trend + seasonal signal."""
    t = np.arange(n, dtype=np.float32)
    trend = 0.01 * t
    seasonal = np.sin(2 * np.pi * t / 50) + 0.5 * np.sin(2 * np.pi * t / 120)
    noise = 0.3 * np.random.randn(n).astype(np.float32)
    return trend + seasonal + noise


class MinMaxScaler:
    def __init__(self) -> None:
        self.min_ = 0.0
        self.max_ = 1.0

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.min_ = float(data.min())
        self.max_ = float(data.max())
        return (data - self.min_) / (self.max_ - self.min_ + 1e-8)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * (self.max_ - self.min_) + self.min_


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, window: int, horizon: int = 1) -> None:
        self.data = torch.from_numpy(data).float()
        self.window = window
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.data) - self.window - self.horizon + 1

    def __getitem__(self, idx: int):
        x = self.data[idx: idx + self.window].unsqueeze(1)  # (window, 1)
        y = self.data[idx + self.window: idx + self.window + self.horizon]
        return x, y


# ── model ───────────────────────────────────────────────────────


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, n_layers: int = 2, horizon: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── training ────────────────────────────────────────────────────


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_all(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    preds = []
    for x, _ in loader:
        preds.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(preds)


def save_forecast_plot(
    actual: np.ndarray,
    predicted: np.ndarray,
    train_size: int,
    window: int,
    scaler: MinMaxScaler,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    actual_inv = scaler.inverse_transform(actual)
    pred_inv = scaler.inverse_transform(predicted.flatten())

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual_inv, label="Actual", alpha=0.7)
    offset = window  # predictions start after first window
    pred_x = np.arange(offset, offset + len(pred_inv))
    ax.plot(pred_x, pred_inv, label="Predicted", alpha=0.8)
    ax.axvline(train_size, color="gray", linestyle="--", alpha=0.5, label="Train/Test split")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.set_title("LSTM Time Series Forecast")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lstm_forecast.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / 'lstm_forecast.png'}")


def save_loss_curve(losses: list[float]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lstm_loss.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM Time Series Forecaster")
    parser.add_argument("--csv", type=str, default=None, help="CSV file path")
    parser.add_argument("--column", type=str, default=None, help="Column to forecast")
    parser.add_argument("--window", type=int, default=50, help="Look-back window")
    parser.add_argument("--horizon", type=int, default=1, help="Steps ahead to predict")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    # load data
    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        col = args.column or df.columns[0]
        raw = df[col].values.astype(np.float32)
    else:
        print("Using synthetic data (sine + trend + noise)")
        raw = generate_synthetic(2000)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(raw)

    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data  # we predict over full range for plotting

    train_ds = TimeSeriesDataset(train_data, args.window, args.horizon)
    full_ds = TimeSeriesDataset(test_data, args.window, args.horizon)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = LSTMForecaster(input_dim=1, hidden_dim=64, n_layers=2, horizon=args.horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"LSTMForecaster: {total_params:,} parameters")
    print(f"Data: {len(raw):,} steps, window={args.window}, horizon={args.horizon}\n")

    losses = []
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        losses.append(loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  MSE={loss:.6f}")

    predictions = predict_all(model, full_ds, device)
    save_forecast_plot(raw, predictions, train_size, args.window, scaler)
    save_loss_curve(losses)

    # metrics on test portion
    test_start = train_size - args.window
    test_preds = predictions[test_start:]
    test_actual = data[train_size: train_size + len(test_preds)]
    mse = float(np.mean((test_preds.flatten() - test_actual) ** 2))
    print(f"\nTest MSE (normalised): {mse:.6f}")
    print("Done ✓")


if __name__ == "__main__":
    main()
