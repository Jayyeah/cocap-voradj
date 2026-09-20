#!/usr/bin/env python3
"""Static sizing and checkpoint-policy audit; never writes a replay snapshot."""
from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from cocap_voradj.models.shared_local_ac import SharedLocalACNetworkConfig
from cocap_voradj.training.shared_local_ac import (
    RecoveryInitPool,
    SharedLocalACConfig,
    SharedLocalACTrainer,
    SharedLocalReplay,
    Transition,
    estimate_replay_row_bytes,
)
from cocap_voradj.training.trainer import load_config


def sample_obs() -> dict[str, np.ndarray]:
    return {
        "self": np.zeros(9, dtype=np.float32),
        "pursuers": np.zeros((8, 7), dtype=np.float32),
        "evaders": np.zeros((8, 7), dtype=np.float32),
        "obstacles": np.zeros((5, 5), dtype=np.float32),
        "masks": np.asarray([True] + [False] * 21, dtype=bool),
        "types": np.asarray([0] + [1] * 8 + [2] * 8 + [3] * 5, dtype=np.int64),
    }


def serialized_size(payload) -> int:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return int(buffer.tell())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    ac = config.get("ac", {}) or {}
    network = SharedLocalACNetworkConfig(**dict(ac.get("network", {}) or {}))
    trainer_config = SharedLocalACConfig(
        gamma=float(ac.get("gamma", 0.99)), tau=float(ac.get("tau", 0.005)),
        actor_lr=float(ac.get("actor_lr", 1e-4)), critic_lr=float(ac.get("critic_lr", 1e-4)),
        adam_eps=float(ac.get("adam_eps", 1e-8)), batch_size=int(ac.get("batch_size", 128)),
        min_replay_size=int(ac.get("min_replay_size", 3000)), train_freq=int(ac.get("train_freq", 4)),
        max_grad_norm=float(ac.get("max_grad_norm", 0.5)), replay_capacity=int(ac.get("replay_capacity", 1_000_000)),
        replay_fallback=str(ac.get("replay_fallback", "defer")),
    )
    trainer = SharedLocalACTrainer(network, trainer_config, "cpu")
    model_payload = trainer.checkpoint_payload(config=config, global_step=25000, episode=0, include_runtime=False)
    runtime_payload = trainer.checkpoint_payload(config=config, global_step=25000, episode=0, include_runtime=True, include_replay=False)
    row = Transition(
        obs=sample_obs(), action=0, reward=0.0, next_obs=sample_obs(), done=False,
        terminated=False, truncated=False, phase="pre_capture", replay_class="pre_capture_cover",
        behavior_probability=1.0 / 9.0, epsilon=0.6, actor_probability=1.0 / 9.0,
        episode_id=0, agent_id=0, metadata={"phase": "pre_capture"},
    )
    row_bytes = estimate_replay_row_bytes(row)
    disk = shutil.disk_usage(Path.cwd())
    active_agents = int((config.get("env", {}) or {}).get("num_pursuers", 4))
    def replay_bytes(steps: int) -> int:
        return int(min(trainer_config.replay_capacity, steps * active_agents) * row_bytes)
    payload = {
        "schema": "ac-capability-disk-safety-v1",
        "config": str(config_path),
        "filesystem": {"path": str(Path.cwd()), "free_bytes": int(disk.free), "total_bytes": int(disk.total), "used_bytes": int(disk.used)},
        "checkpoint_policy": {
            "historical_model_steps": [25000, 50000, 75000, 100000],
            "historical_model_only": True,
            "historical_model_copies": 4,
            "resume_latest_only": True,
            "resume_atomic_replace": True,
            "resume_contains_optimizer_rng": True,
            "replay_in_resume_default": False,
            "replay_snapshot_files": 0,
            "evaluation_artifact": "fixed_small_summary_only",
            "full_rollout_tensors": False,
            "recovery_pool": "bounded_in_memory_only_capacity_1000",
            "large_historical_resume_copies_per_run": 1,
        },
        "serialized_estimates_bytes": {
            "model_only_checkpoint": serialized_size(model_payload),
            "runtime_resume_without_replay": serialized_size(runtime_payload),
            "one_replay_row": row_bytes,
            "replay_100k_steps_if_explicitly_saved": replay_bytes(100000),
            "replay_200k_steps_if_explicitly_saved": replay_bytes(200000),
            "latest_resume_with_replay_100k": serialized_size(runtime_payload) + replay_bytes(100000),
            "latest_resume_with_replay_200k": serialized_size(runtime_payload) + replay_bytes(200000),
            "four_model_checkpoints": 4 * serialized_size(model_payload),
        },
        "growth_assumptions": {
            "active_agent_rows_per_env_step": active_agents,
            "replay_capacity_rows": trainer_config.replay_capacity,
            "100k_replay_rows": active_agents * 100000,
            "200k_replay_rows": active_agents * 200000,
            "default_disk_growth_100k": 4 * serialized_size(model_payload) + serialized_size(runtime_payload),
            "default_disk_growth_200k": 4 * serialized_size(model_payload) + serialized_size(runtime_payload),
            "explicit_replay_resume_growth_100k": 4 * serialized_size(model_payload) + serialized_size(runtime_payload) + replay_bytes(100000),
            "explicit_replay_resume_growth_200k": 4 * serialized_size(model_payload) + serialized_size(runtime_payload) + replay_bytes(200000),
        },
        "no_large_test_artifact_written": True,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
