"""
classical_forecast.py — Classical time series methods: ARIMA & Holt-Winters.

Features:
  - ADF stationarity test
  - Time series decomposition (trend + seasonal + residual)
  - ACF/PACF plots for parameter selection
  - Auto-ARIMA with grid search
  - Holt-Winters exponential smoothing
  - Side-by-side forecast comparison

Usage:
    python classical_forecast.py
    python classical_forecast.py --csv data.csv --column sales
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("outputs")


def generate_synthetic(n: int = 500) -> pd.Series:
    """Monthly-like seasonal data with trend."""
    np.random.seed(42)
    t = np.arange(n, dtype=np.float32)
    trend = 50 + 0.1 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise = 3 * np.random.randn(n).astype(np.float32)
    dates = pd.date_range("2000-01", periods=n, freq="ME")
    return pd.Series(trend + seasonal + noise, index=dates, name="value")


def adf_test(series: pd.Series) -> dict:
    result = adfuller(series.dropna())
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Stationary": result[1] < 0.05,
    }


def save_decomposition(series: pd.Series, tag: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    try:
        decomp = seasonal_decompose(series, model="additive", period=12)
    except Exception:
        decomp = seasonal_decompose(series, model="additive")
    fig = decomp.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(f"Decomposition — {tag}", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_decomposition.png", dpi=150)
    plt.close(fig)
    print(f"  Saved decomposition plot")


def save_acf_pacf(series: pd.Series, tag: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), ax=ax1, lags=40)
    plot_pacf(series.dropna(), ax=ax2, lags=40)
    fig.suptitle(f"ACF / PACF — {tag}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_acf_pacf.png", dpi=150)
    plt.close(fig)
    print(f"  Saved ACF/PACF plot")


def fit_arima(train: pd.Series, test: pd.Series, seasonal_period: int = 12):
    """Fit SARIMAX and forecast."""
    best_aic = np.inf
    best_order = (1, 1, 1)
    for p in range(3):
        for q in range(3):
            try:
                model = SARIMAX(
                    train, order=(p, 1, q),
                    seasonal_order=(1, 1, 1, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False)
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, 1, q)
            except Exception:
                continue

    model = SARIMAX(
        train, order=best_order,
        seasonal_order=(1, 1, 1, seasonal_period),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    forecast = result.forecast(steps=len(test))
    return forecast, best_order, best_aic


def fit_holt_winters(train: pd.Series, test: pd.Series, seasonal_period: int = 12):
    model = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=seasonal_period
    ).fit()
    forecast = model.forecast(steps=len(test))
    return forecast


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical Time Series Forecasting")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--column", type=str, default=None)
    parser.add_argument("--seasonal-period", type=int, default=12)
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=True, index_col=0)
        col = args.column or df.columns[0]
        series = df[col]
        tag = Path(args.csv).stem
    else:
        print("Using synthetic monthly data")
        series = generate_synthetic(500)
        tag = "synthetic"

    print(f"Series: {len(series)} observations")
    print(f"  Range: {series.index[0]} → {series.index[-1]}")

    # stationarity
    adf = adf_test(series)
    print(f"\n  ADF test: statistic={adf['ADF Statistic']:.4f}, p={adf['p-value']:.4f}, stationary={adf['Stationary']}")

    save_decomposition(series, tag)
    save_acf_pacf(series, tag)

    # train/test split
    split = int(len(series) * 0.8)
    train, test = series.iloc[:split], series.iloc[split:]
    print(f"\n  Train: {len(train)}  Test: {len(test)}")

    # ARIMA
    print("\nFitting SARIMAX ...")
    arima_fc, order, aic = fit_arima(train, test, args.seasonal_period)
    arima_mae = np.mean(np.abs(test.values - arima_fc.values))
    print(f"  Best order: {order}, AIC={aic:.1f}")
    print(f"  ARIMA MAE: {arima_mae:.4f}")

    # Holt-Winters
    print("\nFitting Holt-Winters ...")
    hw_fc = fit_holt_winters(train, test, args.seasonal_period)
    hw_mae = np.mean(np.abs(test.values - hw_fc.values))
    print(f"  HW MAE: {hw_mae:.4f}")

    # comparison plot
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train.index, train.values, label="Train", alpha=0.7)
    ax.plot(test.index, test.values, label="Actual", color="black", linewidth=2)
    ax.plot(test.index, arima_fc.values, label=f"ARIMA {order} (MAE={arima_mae:.2f})", linestyle="--")
    ax.plot(test.index, hw_fc.values, label=f"Holt-Winters (MAE={hw_mae:.2f})", linestyle="--")
    ax.set_title("Classical Forecast Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_forecast_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved {OUTPUT_DIR / f'{tag}_forecast_comparison.png'}")
    print("Done ✓")


if __name__ == "__main__":
    main()
