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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO on LunarLander-v3")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--model-path", type=str, default="models/lunarlander_ppo.zip")
    parser.add_argument("--vecnorm-path", type=str, default="models/lunarlander_ppo_vecnormalize.pkl")
    args = parser.parse_args()

    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_env():
        return gym.make("LunarLander-v3", render_mode="human")

    env = DummyVecEnv([make_env])

    vecnorm_path = Path(args.vecnorm_path)
    if vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(args.model_path, env=env, device=device)

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
