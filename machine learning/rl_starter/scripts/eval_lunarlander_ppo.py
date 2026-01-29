from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rl_utils.paths import ensure_dirs


def pick_device(requested: str) -> str:
    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested
    if device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available. Falling back to CPU.")
            return "cpu"
        try:
            arch_list = torch.cuda.get_arch_list()
            major, minor = torch.cuda.get_device_capability()
            sm = f"sm_{major}{minor}"
            if arch_list and sm not in arch_list:
                print(f"CUDA arch {sm} not supported by this PyTorch build. Falling back to CPU.")
                return "cpu"
        except Exception:
            pass
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO on LunarLander-v2")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--model-path", type=str, default="models/lunarlander_ppo.zip")
    parser.add_argument("--vecnorm-path", type=str, default="models/lunarlander_ppo_vecnormalize.pkl")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    ensure_dirs()
    device = pick_device(args.device)

    def make_env():
        return gym.make("LunarLander-v2", render_mode="human")

    env = DummyVecEnv([make_env])

    vecnorm_path = Path(args.vecnorm_path)
    if vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False
        env.norm_reward = False

    model_path = Path(args.model_path)
    if model_path.suffix == ".zip":
        model_path = model_path.with_suffix("")
    model = PPO.load(str(model_path), env=env, device=device)

    for _ in range(args.episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)
            done = bool(dones[0])

    env.close()


if __name__ == "__main__":
    main()
