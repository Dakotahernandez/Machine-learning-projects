from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rl_utils.callbacks import make_checkpoint_callback, make_eval_callback
from rl_utils.paths import ensure_dirs
from rl_utils.seeding import seed_everything


def build_vec_env(n_envs: int, seed: int, vec_type: str):
    def make_env(rank: int):
        def _init():
            env = gym.make("LunarLander-v2")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.reset(seed=seed + rank)
            return env

        return _init

    set_random_seed(seed)
    env_fns = [make_env(i) for i in range(n_envs)]
    if vec_type == "subproc":
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


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
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v2")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vec-env", choices=["subproc", "dummy"], default="subproc")
    parser.add_argument("--vec-normalize", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--run-name", type=str, default="lunarlander_ppo")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    args = parser.parse_args()

    dirs = ensure_dirs()
    seed_everything(args.seed)

    env = build_vec_env(args.n_envs, args.seed, args.vec_env)
    env = VecMonitor(env)

    eval_env = DummyVecEnv([lambda: gym.make("LunarLander-v2")])
    eval_env = VecMonitor(eval_env)

    save_path = Path(args.save_path) if args.save_path else dirs["models"] / f"{args.run_name}.zip"
    log_dir = Path(args.log_dir) if args.log_dir else dirs["runs"] / args.run_name
    vecnorm_path = save_path.with_suffix("").with_name(f"{save_path.stem}_vecnormalize.pkl")
    if args.vec_normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
        eval_env.obs_rms = env.obs_rms
        eval_env.ret_rms = env.ret_rms

    device = pick_device(args.device)

    checkpoint_dir = dirs["models"] / "lunarlander_ppo_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        make_checkpoint_callback(args.checkpoint_freq, checkpoint_dir, "lunarlander_ppo"),
        make_eval_callback(eval_env, dirs["models"], log_dir / "eval", args.eval_freq),
    ]

    model = PPO(
        "MlpPolicy",
        env,
        verbose=args.verbose,
        tensorboard_log=str(log_dir),
        device=device,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))

    if args.vec_normalize:
        env.save(str(vecnorm_path))

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
