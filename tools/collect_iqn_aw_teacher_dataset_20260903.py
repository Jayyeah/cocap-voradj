#!/usr/bin/env python3
"""Collect deterministic Final-IQN-AW supervision on the exact MAPPO-9-v2 task."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.continuous.central_schema import build_central_global_obs
from cocap_voradj.training.continuous.formal_config import scene_config
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.trainer import load_config, set_global_config
from tools.evaluate_iqn_corrected_capture import (
    DEFAULT_COLLISION_SEMANTICS,
    action_grid_from_config,
    current_apf_actions,
    map_action_index,
    validate_corrected_capture_contract,
)
from tools.evaluate_vxy_stage2_cross_retention_20260903 import ring_count
from tools.probe_cf3_policy_temperature import apply_collision_semantics_override

SCHEMA = "iqn-aw-teacher-dataset-v1"
ROLE_IDS = {"other": 0, "support": 1, "direct": 2, "pursuing": 3}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fixed_midpoint_q(
    model: CoCapIQN,
    observations: Sequence[Mapping[str, np.ndarray]],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    batch = stack_obs(list(observations), device)
    tau = (torch.arange(32, device=device, dtype=torch.float32) + 0.5) / 32.0
    with torch.no_grad():
        quantiles = model(batch, num_tau=32, mode="voradj", tau=tau)["q_values"]
        q_values = quantiles.mean(dim=1)
    greedy = q_values.argmax(dim=-1)
    return q_values.cpu().numpy().astype(np.float32), greedy.cpu().numpy().astype(np.int64)


def role_id(info: Mapping[str, Any]) -> tuple[int, bool, bool, bool]:
    metadata = dict(info.get("replay_metadata", {}) or {})
    direct = int(metadata.get("vct_ls_direct_enemy_count", 0) or 0) > 0
    pursuing = bool(metadata.get("effective_pursuing", False))
    support = bool(metadata.get("support_candidate", False))
    if direct:
        value = ROLE_IDS["direct"]
    elif support:
        value = ROLE_IDS["support"]
    elif pursuing:
        value = ROLE_IDS["pursuing"]
    else:
        value = ROLE_IDS["other"]
    return value, pursuing, direct, support


def append_rows(storage: dict[str, list[np.ndarray]], prefix: str, tree: Mapping[str, Any], count: int) -> None:
    for key, value in tree.items():
        array = np.asarray(value)
        repeated = np.repeat(array[None], int(count), axis=0)
        storage.setdefault(f"{prefix}{key}", []).append(repeated)


def flush_shard(
    output_root: Path,
    shard_index: int,
    storage: dict[str, list[np.ndarray]],
) -> tuple[str, int, dict[str, list[int]]]:
    arrays = {key: np.concatenate(values, axis=0) for key, values in storage.items()}
    row_count = int(len(arrays["episode"]))
    shapes = {key: list(value.shape[1:]) for key, value in arrays.items()}
    name = f"shard_{shard_index:04d}.npz"
    destination = output_root / "shards" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, destination)
    return str(destination.relative_to(output_root)), row_count, shapes


def load_or_initialize_manifest(
    path: Path,
    *,
    config_path: Path,
    config_hash: str,
    checkpoint: Path,
    checkpoint_sha: str,
    seed: int,
    shard_episodes: int,
    requested_episodes: int,
    max_rows: int,
    contract_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema": SCHEMA,
            "config_hash": config_hash,
            "teacher_checkpoint_sha256": checkpoint_sha,
            "seed_base": int(seed),
            "shard_episodes": int(shard_episodes),
        }
        mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
        if mismatched:
            raise ValueError(f"dataset resume contract mismatch: {mismatched}")
        return manifest
    return {
        "schema": SCHEMA,
        "status": "running",
        "created_at": now(),
        "config": str(config_path),
        "config_hash": config_hash,
        "contract_audit": contract_audit,
        "teacher_checkpoint": str(checkpoint),
        "teacher_checkpoint_sha256": checkpoint_sha,
        "teacher_q_contract": "mean_j Z_tau_j with tau_j=(j+0.5)/32; no random tau",
        "policy": {"epsilon": 0.0, "quantiles": "fixed_midpoint_32"},
        "seed_base": int(seed),
        "requested_episodes": int(requested_episodes),
        "max_active_agent_rows": int(max_rows),
        "shard_episodes": int(shard_episodes),
        "next_episode": 0,
        "row_count": 0,
        "episode_count": 0,
        "capture_count": 0,
        "collision_count": 0,
        "episodes_visited_2plus": 0,
        "episodes_visited_3plus": 0,
        "role_counts": {name: 0 for name in ROLE_IDS},
        "shards": [],
        "field_contract": {
            "local_obs.*": "per active agent/timestep",
            "teacher_q": [9],
            "greedy_action": [],
            "episode/timestep/agent": [],
            "role_id/pursuing/direct/support": [],
            "reward/done": [],
            "global_state.*": "central state repeated for each active agent row",
            "ring_count/near_capture": [],
        },
        "role_ids": ROLE_IDS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--max-active-agent-rows", type=int, default=200000)
    parser.add_argument("--shard-episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026091301)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if min(args.episodes, args.max_active_agent_rows, args.shard_episodes) <= 0:
        parser.error("episodes, max-active-agent-rows and shard-episodes must be positive")

    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_root = Path(args.output_root).resolve()
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != str(args.expected_checkpoint_sha256).lower():
        raise ValueError("teacher checkpoint SHA-256 mismatch")

    root_config = load_config(str(config_path))
    evaluation_config = apply_collision_semantics_override(root_config, DEFAULT_COLLISION_SEMANTICS)
    config = scene_config(evaluation_config, "capture")
    contract_audit = validate_corrected_capture_contract(config)
    config_hash = stable_hash(config)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest = load_or_initialize_manifest(
        manifest_path,
        config_path=config_path,
        config_hash=config_hash,
        checkpoint=checkpoint,
        checkpoint_sha=checkpoint_sha,
        seed=args.seed,
        shard_episodes=args.shard_episodes,
        requested_episodes=args.episodes,
        max_rows=args.max_active_agent_rows,
        contract_audit=contract_audit,
    )
    if manifest.get("status") == "complete":
        print(json.dumps({"status": "already_complete", "row_count": manifest["row_count"]}))
        return 0

    random.seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    torch.manual_seed(int(args.seed))
    model = CoCapIQN.load(str(checkpoint), device=str(args.device)).eval()
    grid = action_grid_from_config(config)
    tau_probe = (torch.arange(32, device=args.device, dtype=torch.float32) + 0.5) / 32.0
    if not torch.equal(tau_probe.cpu(), (torch.arange(32, dtype=torch.float32) + 0.5) / 32.0):
        raise RuntimeError("midpoint tau construction failed")

    storage: dict[str, list[np.ndarray]] = {}
    shard_episode_count = 0
    shard_index = len(manifest["shards"])
    started = time.time()
    start_episode = int(manifest["next_episode"])
    for episode_index in range(start_episode, int(args.episodes)):
        if int(manifest["row_count"]) >= int(args.max_active_agent_rows):
            break
        episode_seed = int(args.seed) + episode_index
        set_global_config(config)
        env = VorAdjEnv(copy.deepcopy(config), seed=episode_seed)
        observations = list(env.reset())
        apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
        capture_types: list[str] = []
        collision = False
        ring_counts: list[int] = []
        episode_rows = 0
        for timestep in range(1, int(config["env"]["episode_max_length"]) + 1):
            active = [index for index, obs in enumerate(observations) if obs is not None]
            if not active:
                break
            active_obs = [observations[index] for index in active]
            q_values, greedy = fixed_midpoint_q(model, active_obs, str(args.device))
            global_state = build_central_global_obs(
                env,
                max_agents=int(config["training"]["max_agents"]),
                max_evaders=int(config["central_critic"].get("max_evaders", 8)),
                max_obstacles=int(config["central_critic"].get("max_obstacles", 5)),
                self_feature_dim=int(config["central_critic"].get("self_feature_dim", 9)),
            )
            commands: list[np.ndarray | None] = [None] * len(observations)
            for row, agent_index in enumerate(active):
                raw = map_action_index(int(greedy[row]), grid)
                commands[agent_index] = np.asarray(env.action_adapter.validate(raw), dtype=np.float32)
            outcome = env.step(commands, current_apf_actions(env, apf_agents))
            capture_types.extend(
                str(event.get("capture_type", "unknown"))
                for event in list(getattr(env, "last_capture_events", []) or [])
            )
            collision = collision or bool(getattr(env, "last_collision_events", []) or [])
            current_ring = int(ring_count(env))
            ring_counts.append(current_ring)

            local_tree = {
                key: np.stack([np.asarray(active_obs[row][key]) for row in range(len(active))])
                for key in active_obs[0]
            }
            for key, value in local_tree.items():
                storage.setdefault(f"local_obs.{key}", []).append(value)
            append_rows(storage, "global_state.", global_state, len(active))
            roles = [role_id(outcome.infos[index]) for index in active]
            scalars = {
                "teacher_q": q_values,
                "greedy_action": greedy,
                "episode": np.full(len(active), episode_index, dtype=np.int32),
                "environment_seed": np.full(len(active), episode_seed, dtype=np.int64),
                "timestep": np.full(len(active), timestep, dtype=np.int32),
                "agent": np.asarray(active, dtype=np.int16),
                "role_id": np.asarray([value[0] for value in roles], dtype=np.uint8),
                "pursuing": np.asarray([value[1] for value in roles], dtype=np.bool_),
                "direct": np.asarray([value[2] for value in roles], dtype=np.bool_),
                "support": np.asarray([value[3] for value in roles], dtype=np.bool_),
                "reward": np.asarray([outcome.rewards[index] for index in active], dtype=np.float32),
                "done": np.asarray([outcome.dones[index] for index in active], dtype=np.bool_),
                "ring_count": np.full(len(active), current_ring, dtype=np.int8),
                "near_capture": np.full(len(active), current_ring >= 2, dtype=np.bool_),
            }
            for key, value in scalars.items():
                storage.setdefault(key, []).append(value)
            for value in roles:
                role_name = next(name for name, index in ROLE_IDS.items() if index == value[0])
                manifest["role_counts"][role_name] += 1
            episode_rows += len(active)
            observations = list(outcome.observations)
            if all(outcome.dones):
                break

        record = env.episode_record(task="capture")
        manifest["row_count"] += int(episode_rows)
        manifest["episode_count"] += 1
        manifest["capture_count"] += int(bool(record.get("captured", False)))
        manifest["collision_count"] += int(bool(collision or record.get("collision_event", False)))
        manifest["episodes_visited_2plus"] += int(any(value >= 2 for value in ring_counts))
        manifest["episodes_visited_3plus"] += int(any(value >= 3 for value in ring_counts))
        manifest["next_episode"] = episode_index + 1
        shard_episode_count += 1

        should_flush = (
            shard_episode_count >= int(args.shard_episodes)
            or int(manifest["row_count"]) >= int(args.max_active_agent_rows)
            or episode_index + 1 >= int(args.episodes)
        )
        if should_flush:
            relative, rows, shapes = flush_shard(output_root, shard_index, storage)
            manifest["shards"].append({
                "path": relative,
                "rows": rows,
                "episode_end_exclusive": episode_index + 1,
                "sha256": sha256_file(output_root / relative),
            })
            manifest["field_shapes"] = shapes
            manifest["updated_at"] = now()
            manifest["elapsed_seconds"] = time.time() - started
            atomic_json(manifest_path, manifest)
            storage = {}
            shard_episode_count = 0
            shard_index += 1

    manifest["status"] = "complete"
    manifest["completed_at"] = now()
    episodes = max(int(manifest["episode_count"]), 1)
    manifest["summary"] = {
        "capture_rate": manifest["capture_count"] / episodes,
        "collision_rate": manifest["collision_count"] / episodes,
        "visited_2plus_rate": manifest["episodes_visited_2plus"] / episodes,
        "visited_3plus_rate": manifest["episodes_visited_3plus"] / episodes,
        "normal_rollouts_only": True,
        "successful_episodes_only": False,
    }
    manifest["coverage_checks"] = {
        "support_rows": int(manifest["role_counts"]["support"]),
        "near_capture_rows": int(sum(
            int(np.load(output_root / shard["path"], allow_pickle=False)["near_capture"].sum())
            for shard in manifest["shards"]
        )),
    }
    atomic_json(manifest_path, manifest)
    (output_root / "DATASET_DONE").write_text(now() + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "summary": manifest["summary"], "rows": manifest["row_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
