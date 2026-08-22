"""Atomic full-runtime and model-only checkpoint IO."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import uuid

import torch

from .trainer import MAADPGTrainer


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_full_checkpoint(
    trainer: MAADPGTrainer,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    _atomic_torch_save(trainer.state_dict(metadata=metadata), Path(path))


def load_full_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[MAADPGTrainer, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    trainer = MAADPGTrainer.from_state_dict(payload, device=device)
    return trainer, payload["metadata"]


def save_model_checkpoint(
    trainer: MAADPGTrainer,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "checkpoint_version": 1,
        "checkpoint_kind": "model_only",
        "trainer_config": trainer.config,
        "learner": trainer.learner.state_dict(),
        "environment_steps": trainer.runtime.environment_steps,
        "gradient_steps": trainer.runtime.gradient_steps,
        "metadata": metadata or {},
    }
    _atomic_torch_save(payload, Path(path))
