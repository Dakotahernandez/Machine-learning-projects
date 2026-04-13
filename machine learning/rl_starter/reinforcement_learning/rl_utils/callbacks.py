from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


def make_checkpoint_callback(save_freq: int, save_path: Path, name_prefix: str):
    return CheckpointCallback(
        save_freq=save_freq,
        save_path=str(save_path),
        name_prefix=name_prefix,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )


def make_eval_callback(eval_env, best_model_dir: Path, log_path: Path, eval_freq: int):
    return EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_path),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )
