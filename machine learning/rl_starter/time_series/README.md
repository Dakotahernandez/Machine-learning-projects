# Time Series Forecasting

LSTM and Transformer-based time series prediction with PyTorch, plus classical methods with statsmodels.

## Projects

### `lstm_forecaster.py` — LSTM Sequence Forecasting
- Sliding window approach for univariate/multivariate time series
- Multi-step ahead prediction
- Works with synthetic data (sine waves) or real datasets
- Rolling forecast with expanding training window
- Forecast vs actual plots with confidence intervals

### `transformer_forecaster.py` — Transformer for Time Series
- Positional encoding + multi-head self-attention
- Encoder-only architecture adapted for regression
- Demonstrates attention patterns on time series data
- Compares against LSTM baseline

### `classical_forecast.py` — ARIMA & Exponential Smoothing
- Auto ARIMA model selection
- Holt-Winters exponential smoothing
- Decomposition: trend + seasonal + residual
- ADF stationarity test
- ACF/PACF plots

## Quick Start

```powershell
pip install -r requirements.txt

# LSTM forecasting (generates synthetic data by default)
python lstm_forecaster.py --epochs 50 --device auto

# Transformer forecasting
python transformer_forecaster.py --epochs 30 --device auto

# Classical methods
python classical_forecast.py
```
