#!/usr/bin/env python3
"""Bounded CPU-only preflight for exact categorical SAC over AW9."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

from cocap_voradj.models.shared_local_ac import (
    SharedLocalACNetworkConfig,
    assert_runtime_aw9,
    canonical_aw9_grid,
)
from cocap_voradj.training.discrete_sac import (
    DiscreteSACConfig,
    DiscreteSACTrainer,
    atomic_torch_save,
)
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.trainer import load_config
from tools.run_shared_local_ac_capability import make_env


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _zero_like(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.zeros_like(value) for key, value in observation.items()}


def run(config_path: Path, output_root: Path, steps: int | None = None) -> dict[str, Any]:
    config = load_config(str(config_path))
    if str(config.get("device", "cpu")).lower() != "cpu":
        raise ValueError("A2 discrete-SAC preflight is CPU-only; config device must be cpu")
    ac = config.get("ac", {}) or {}
    network = SharedLocalACNetworkConfig(**dict(ac.get("network", {}) or {}))
    trainer_config = DiscreteSACConfig(
        gamma=float(ac.get("gamma", 0.99)),
        tau=float(ac.get("tau", 0.005)),
        actor_lr=float(ac.get("actor_lr", 3e-4)),
        critic_lr=float(ac.get("critic_lr", 3e-4)),
        alpha_lr=float(ac.get("alpha_lr", 3e-4)),
        alpha_init=float(ac.get("alpha_init", 0.2)),
        adam_eps=float(ac.get("adam_eps", 1e-8)),
        max_grad_norm=float(ac["max_grad_norm"]) if ac.get("max_grad_norm") is not None else None,
    )
    trainer = DiscreteSACTrainer(network, trainer_config, device="cpu")

    env = make_env(config, "voradj_coverage", int(config.get("seed", 2026092102)))
    observations = env.reset()
    assert_runtime_aw9(env)
    np.testing.assert_allclose(np.asarray(env.pursuers[0].action_list), canonical_aw9_grid(), atol=1e-7)
    env_facts = {
        "task": "voradj_coverage",
        "action_mode": str(config.get("pursuer", {}).get("action_mode")),
        "collision_semantics": str(config.get("env", {}).get("collision_semantics")),
        "observation_mode": str(config.get("perception", {}).get("observation_mode")),
        "num_pursuers": len(env.pursuers),
        "num_evaders": len(env.evaders),
        "episode_max_length": int(env.episode_max_length),
    }

    requested_steps = int(steps if steps is not None else (ac.get("smoke", {}) or {}).get("steps", 4))
    batch_size = int((ac.get("smoke", {}) or {}).get("batch_size", ac.get("batch_size", 8)))
    rows: list[tuple[dict[str, np.ndarray], int, float, dict[str, np.ndarray], bool]] = []
    action_counts = np.zeros(9, dtype=np.int64)
    current = list(observations)
    for _ in range(max(requested_steps, 1)):
        active = [index for index, observation in enumerate(current) if observation is not None]
        if not active:
            current = list(env.reset())
            continue
        actor_batch = stack_obs([current[index] for index in active], "cpu")
        with torch.no_grad():
            actions_tensor = trainer.actor.deterministic_action(actor_batch)
        actions: list[int | None] = [None] * len(current)
        for row, index in enumerate(active):
            action = int(actions_tensor[row].item())
            actions[index] = action
            action_counts[action] += 1
        outcome = env.step(actions, [])
        for index in active:
            old = current[index]
            assert old is not None
            nxt = outcome.observations[index] if outcome.observations[index] is not None else _zero_like(old)
            info = outcome.infos[index] if index < len(outcome.infos) else {}
            rows.append((old, int(actions[index]), float(outcome.rewards[index]), nxt, bool(info.get("terminated", False))))
        current = list(outcome.observations)

    if len(rows) < batch_size:
        raise RuntimeError(f"CPU preflight collected only {len(rows)} rows; need {batch_size}")
    rows = rows[:batch_size]
    batch = {
        "obs": stack_obs([row[0] for row in rows], "cpu"),
        "actions": torch.as_tensor([row[1] for row in rows], dtype=torch.long),
        "rewards": torch.as_tensor([row[2] for row in rows], dtype=torch.float32),
        "next_obs": stack_obs([row[3] for row in rows], "cpu"),
        "terminated": torch.as_tensor([row[4] for row in rows], dtype=torch.bool),
    }
    metrics = trainer.update(batch)
    if metrics["finite"] != 1.0:
        raise RuntimeError(f"non-finite discrete SAC preflight metrics: {metrics}")

    contract = {
        "schema": "aw9-discrete-sac-preflight-contract-v1",
        "task": "coverage",
        "action_size": 9,
        "action_grid": canonical_aw9_grid().tolist(),
        "observation_contract": "existing_shared_local_aw9",
        "formal_training_started": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = output_root / "discrete_sac_aw9_cpu_smoke.pt"
    atomic_torch_save(trainer.checkpoint_payload(contract=contract, global_step=len(rows)), checkpoint)
    restored = DiscreteSACTrainer(network, trainer_config, device="cpu")
    restored.load_payload(torch.load(checkpoint, map_location="cpu", weights_only=False), contract)
    if restored.update_count != trainer.update_count:
        raise RuntimeError("atomic checkpoint restore lost update count")

    summary = {
        "schema": "discrete-sac-aw9-cpu-preflight-v1",
        "formal_training_started": False,
        "device": "cpu",
        "steps": requested_steps,
        "rows": len(rows),
        "batch_size": batch_size,
        "updates": trainer.update_count,
        "metrics": metrics,
        "action_counts": action_counts.tolist(),
        "aw9_grid": canonical_aw9_grid().tolist(),
        "environment_facts": env_facts,
        "checkpoint": str(checkpoint),
    }
    _write_json(output_root / "discrete_sac_aw9_cpu_preflight.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.output_root.resolve(), args.steps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
