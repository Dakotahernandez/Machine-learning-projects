# Machine Learning Projects

A comprehensive collection of machine learning projects spanning classical ML, deep learning, NLP, computer vision, generative models, time series forecasting, and reinforcement learning — all built from scratch with **PyTorch** and **scikit-learn**.

---

## Projects

| # | Project | Domain | Highlights |
|:-:|---------|--------|------------|
| 1 | [**Supervised Learning**](supervised_learning/) | Classical ML | 8-algorithm pipeline, auto-preprocessing, grid search tuning, ensemble stacking |
| 2 | [**Deep Learning**](deep_learning/) | Neural Networks | MNIST CNN (99%+), CIFAR-10 ResNet with AMP, Fashion-MNIST autoencoder + t-SNE |
| 3 | [**NLP**](nlp/) | Natural Language | IMDB sentiment LSTM, AG News TextCNN, Word2Vec with analogy solving |
| 4 | [**Computer Vision**](computer_vision/) | Vision | Transfer learning (5 architectures), Grad-CAM attention maps, feature clustering + UMAP |
| 5 | [**Generative Models**](generative/) | Generative AI | DCGAN image generation, VAE with latent interpolation, t-SNE visualisation |
| 6 | [**Time Series**](time_series/) | Forecasting | LSTM forecaster, Transformer forecaster, ARIMA, Holt-Winters, decomposition |
| 7 | [**Reinforcement Learning**](reinforcement_learning/) | RL | PPO LunarLander, DQN Atari Pong, web UI dashboard, TensorBoard logging |

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch, torchvision, CUDA/AMP |
| **Classical ML** | scikit-learn, XGBoost, statsmodels |
| **Visualisation** | Matplotlib, Seaborn, TensorBoard, UMAP, t-SNE |
| **RL** | Stable-Baselines3, Gymnasium |
| **Data** | pandas, NumPy |

---

## Repository Structure

```
├── supervised_learning/          # Classical ML pipeline + ensemble methods
│   ├── ml_pipeline.py            #   Auto-preprocessing, 8 algorithms, grid search
│   ├── ensemble_stacking.py      #   Stacking & voting ensembles
│   └── requirements.txt
│
├── deep_learning/                # PyTorch from scratch
│   ├── mnist_cnn.py              #   Custom CNN, OneCycleLR, 99%+ accuracy
│   ├── cifar10_resnet.py         #   ResNet with residual blocks, AMP
│   ├── fashion_autoencoder.py    #   Conv autoencoder + latent t-SNE
│   └── requirements.txt
│
├── nlp/                          # Natural language processing
│   ├── sentiment_lstm.py         #   Bidirectional LSTM on IMDB
│   ├── text_classifier.py        #   TextCNN (Kim 2014) on AG News
│   ├── word_embeddings.py        #   Word2Vec skip-gram from scratch
│   └── requirements.txt
│
├── computer_vision/              # Transfer learning & feature extraction
│   ├── transfer_learning.py      #   Fine-tune ResNet/EfficientNet + Grad-CAM
│   ├── feature_extractor.py      #   Deep feature clustering + UMAP
│   └── requirements.txt
│
├── generative/                   # Generative models
│   ├── dcgan.py                  #   Deep Convolutional GAN
│   ├── vae.py                    #   Variational Autoencoder
│   └── requirements.txt
│
├── time_series/                  # Forecasting
│   ├── lstm_forecaster.py        #   LSTM sliding-window prediction
│   ├── transformer_forecaster.py #   Transformer encoder for time series
│   ├── classical_forecast.py     #   ARIMA + Holt-Winters + decomposition
│   └── requirements.txt
│
├── reinforcement_learning/       # Game-playing agents
│   ├── scripts/                  #   Train/eval scripts + web UI server
│   ├── rl_utils/                 #   Callbacks, seeding, path helpers
│   ├── ui/                       #   Browser-based control panel
│   └── requirements.txt
│
└── .gitignore
```

---

## Quick Start

Each project is self-contained with its own `requirements.txt`. Pick any project and go:

```powershell
# Create a shared virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install per-project deps
pip install -r supervised_learning/requirements.txt
pip install -r deep_learning/requirements.txt
# ... etc
```

Or install everything at once:

```powershell
pip install scikit-learn xgboost pandas matplotlib seaborn tqdm statsmodels umap-learn
pip install stable-baselines3 gymnasium tensorboard
```

---

## Highlights

### Supervised Learning — Compete 8 algorithms in one command
```powershell
python supervised_learning/ml_pipeline.py --dataset iris
```
Automatically preprocesses data, cross-validates Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Decision Tree, and XGBoost, then tunes the winner.

### Deep Learning — Custom ResNet on CIFAR-10
```powershell
python deep_learning/cifar10_resnet.py --epochs 50 --device auto
```
Residual blocks from scratch, cosine LR schedule, mixed precision training, saves prediction samples.

### NLP — Sentiment Analysis from Scratch
```powershell
python nlp/sentiment_lstm.py --epochs 5 --device auto
```
Downloads IMDB, builds vocabulary, trains bidirectional LSTM with packed sequences.

### Generative — DCGAN Digits
```powershell
python generative/dcgan.py --epochs 50 --device auto
```
Generates handwritten digits from noise, saves image grids and latent space interpolations.

### Time Series — LSTM vs Transformer
```powershell
python time_series/lstm_forecaster.py --epochs 50
python time_series/transformer_forecaster.py --epochs 30
```
Compare LSTM and Transformer architectures on the same forecasting task.

### Reinforcement Learning — Train a LunarLander Agent
```powershell
cd reinforcement_learning
.\scripts\install_windows.ps1
.\.venv\Scripts\python.exe .\scripts\train_lunarlander_ppo.py --timesteps 500000 --n-envs 16
.\.venv\Scripts\python.exe .\scripts\ui_server.py  # web UI at localhost:8000
```

---

## Requirements

- **Python** 3.10+
- **GPU** recommended (NVIDIA + CUDA) but all projects work on CPU
- **Windows** 10/11 (tested), Linux/macOS should work with minor path changes
