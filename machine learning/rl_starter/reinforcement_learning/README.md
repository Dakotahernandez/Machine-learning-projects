# Reinforcement Learning

Train game-playing agents with **Gymnasium + Stable-Baselines3 + PyTorch CUDA**.

## Projects

| Agent | Environment | Algorithm | Observations |
|-------|-------------|-----------|-------------|
| LunarLander | `LunarLander-v2` | PPO | Vectorised envs, optional VecNormalize |
| Pong | `ALE/Pong-v5` | DQN | Atari frame-stacking, CNN policy |

## Quick Start

```powershell
# Install (creates .venv with CUDA PyTorch)
.\scripts\install_windows.ps1

# Train LunarLander
.\.venv\Scripts\python.exe .\scripts\train_lunarlander_ppo.py --timesteps 500000 --n-envs 16

# Evaluate (renders the game)
.\.venv\Scripts\python.exe .\scripts\eval_lunarlander_ppo.py --episodes 5
```

## Pong (Atari)

```powershell
.\.venv\Scripts\python.exe -m pip install "gymnasium[atari]==0.29.1" "autorom[accept-rom-license]"
.\.venv\Scripts\python.exe -m autorom
.\.venv\Scripts\python.exe .\scripts\train_pong_dqn.py --timesteps 1000000
.\.venv\Scripts\python.exe .\scripts\eval_pong_dqn.py --episodes 3
```

## Web UI

```powershell
.\.venv\Scripts\python.exe .\scripts\ui_server.py
# Open http://127.0.0.1:8000
```

## TensorBoard

```powershell
.\.venv\Scripts\tensorboard.exe --logdir .\runs
```

## Key Files

```
scripts/
├── train_lunarlander_ppo.py   # PPO training with checkpoints & eval callbacks
├── train_pong_dqn.py          # DQN training for Atari Pong
├── eval_lunarlander_ppo.py    # Evaluate & render LunarLander agent
├── eval_pong_dqn.py           # Evaluate & render Pong agent
├── ui_server.py               # Local web UI server
└── install_windows.ps1        # One-click Windows setup

rl_utils/
├── callbacks.py               # Checkpoint & evaluation callbacks
├── paths.py                   # Project path helpers
└── seeding.py                 # Reproducibility utilities
```
