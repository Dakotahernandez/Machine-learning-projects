from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rl_utils.callbacks import make_checkpoint_callback, make_eval_callback
from rl_utils.paths import ensure_dirs
from rl_utils.seeding import seed_everything


ATARI_INSTALL_HELP = (
    "Atari dependencies not found. Install with: "
    "python -m pip install \"gymnasium[atari,accept-rom-license]\""
)


def build_pong_env(n_envs: int, seed: int, render_mode: str | None):
    try:
        from stable_baselines3.common.env_util import make_atari_env
    except Exception as exc:  # pragma: no cover
        print(ATARI_INSTALL_HELP)
        raise SystemExit(1) from exc

    try:
        env = make_atari_env(
            "ALE/Pong-v5",
            n_envs=n_envs,
            seed=seed,
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
    parser = argparse.ArgumentParser(description="Train DQN on Atari Pong")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--run-name", type=str, default="pong_dqn")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    args = parser.parse_args()

    dirs = ensure_dirs()
    seed_everything(args.seed)

    env = build_pong_env(args.n_envs, args.seed, render_mode=None)
    eval_env = build_pong_env(1, args.seed + 1, render_mode=None)

    device = pick_device(args.device)

    save_path = Path(args.save_path) if args.save_path else dirs["models"] / f"{args.run_name}.zip"
    log_dir = Path(args.log_dir) if args.log_dir else dirs["runs"] / args.run_name

    checkpoint_dir = dirs["models"] / "pong_dqn_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        make_checkpoint_callback(args.checkpoint_freq, checkpoint_dir, "pong_dqn"),
        make_eval_callback(eval_env, dirs["models"], log_dir / "eval", args.eval_freq),
    ]

    model = DQN(
        "CnnPolicy",
        env,
        verbose=args.verbose,
        tensorboard_log=str(log_dir),
        device=device,
        seed=args.seed,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=32,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
    )

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
