# Machine Learning Projects

A comprehensive collection of machine learning projects spanning classical ML, deep learning, NLP, computer vision, generative models, time series forecasting, and reinforcement learning — all built from scratch with PyTorch and scikit-learn.

## Projects

| # | Project | Domain | Stack | Description |
|---|---------|--------|-------|-------------|
| 1 | [Reinforcement Learning](reinforcement_learning/) | RL | PyTorch, SB3, Gymnasium | PPO LunarLander + DQN Atari Pong with web UI |

## Setup

Each project has its own `requirements.txt`. For a shared environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Structure

```
├── reinforcement_learning/   # Gymnasium + SB3 game agents with web UI
├── .gitignore
└── README.md
```
