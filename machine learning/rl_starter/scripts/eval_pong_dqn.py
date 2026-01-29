from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


ATARI_INSTALL_HELP = (
    "Atari dependencies not found. Install with: "
    "python -m pip install \"gymnasium[atari,accept-rom-license]\""
)


def build_pong_env(render_mode: str | None):
    try:
        from stable_baselines3.common.env_util import make_atari_env
    except Exception as exc:  # pragma: no cover
        print(ATARI_INSTALL_HELP)
        raise SystemExit(1) from exc

    try:
        env = make_atari_env(
            "ALE/Pong-v5",
            n_envs=1,
            seed=0,
            env_kwargs={"render_mode": render_mode},
        )
    except Exception as exc:
        print(ATARI_INSTALL_HELP)
        raise SystemExit(1) from exc

    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env


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
    parser = argparse.ArgumentParser(description="Evaluate DQN on Atari Pong")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--model-path", type=str, default="models/pong_dqn.zip")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)

    env = build_pong_env(render_mode="human")
    model_path = Path(args.model_path)
    if model_path.suffix == ".zip":
        model_path = model_path.with_suffix("")
    model = DQN.load(str(model_path), env=env, device=device)

    for _ in range(args.episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)
            done = bool(dones[0])
            try:
                env.render()
            except Exception:
                pass

    env.close()


if __name__ == "__main__":
    main()
