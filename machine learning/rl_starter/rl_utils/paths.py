from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dirs() -> dict[str, Path]:
    root = project_root()
    models = root / "models"
    runs = root / "runs"
    models.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    return {"root": root, "models": models, "runs": runs}
