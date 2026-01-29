# RL Starter (Gymnasium + SB3 + PyTorch CUDA)

Local Windows 11 starter for training game-playing agents with Gymnasium + Stable-Baselines3 + PyTorch (CUDA).

## Quickstart

PowerShell (from project root):

```powershell
# 1) Install dependencies (creates .venv and installs CUDA-enabled torch)
.\scripts\install_windows.ps1
```

```powershell
# 2) Train LunarLander PPO
.\.venv\Scripts\python.exe .\scripts\train_lunarlander_ppo.py --timesteps 500000 --n-envs 16
```

```powershell
# 3) Evaluate LunarLander PPO (renders)
.\.venv\Scripts\python.exe .\scripts\eval_lunarlander_ppo.py --episodes 5
```

## Atari (Pong) setup (optional)

Install Atari extras (one command):

```powershell
.\.venv\Scripts\python.exe -m pip install "gymnasium[atari,accept-rom-license]"
```

Train Pong DQN:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_pong_dqn.py --timesteps 1000000
```

Evaluate Pong DQN (renders):

```powershell
.\.venv\Scripts\python.exe .\scripts\eval_pong_dqn.py --episodes 3
```

## TensorBoard

```powershell
.\.venv\Scripts\tensorboard.exe --logdir .\runs
```

## Troubleshooting (Windows)

- **PowerShell script execution policy**: If scripts are blocked, run PowerShell as your user and execute:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```
- **Visual C++ build tools**: Some packages may require Microsoft Visual C++ Build Tools. Install from the Visual Studio Build Tools installer if you see compiler errors.
- **Rendering**: Gymnasium rendering needs a desktop session. If running headless or via remote session without GUI, use `--render-mode none` (where available) or avoid eval scripts.
- **Slow or stuck rendering**: Try reducing episode count or disabling rendering. Rendering is only enabled in the eval scripts by default.
- **GPU not used**: Ensure your NVIDIA drivers are up to date and that `torch.cuda.is_available()` returns `True`.

## Notes

- LunarLander uses vectorized environments and optional `VecNormalize`.
- Atari is optional and separated from LunarLander. If Atari dependencies are missing, the Pong scripts will exit with a clear message.
