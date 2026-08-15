#!/usr/bin/env python3
"""Fair deterministic IQN evaluation on the corrected CF3 capture contract.

This tool deliberately separates the policy checkpoint from the evaluation
environment.  The historical IQN chooses one of the legacy 3x3 ``(a, omega)``
grid points, but the command is executed through the current continuous-AW
adapter and the current moving-APF CF3 environment.  It therefore answers a
zero-shot policy question; it does not claim that IQN and CF3 were trained on
the same task history.

Important safeguards:

* the historical checkpoint SHA-256 is verified before it is loaded;
* environment seeds are used exactly as requested (there is no legacy ``+43``);
* midpoint IQN quantiles and epsilon=0 make the policy deterministic;
* the moving evader must use the current APF helper on every live step;
* post-fix synchronized/swept collision semantics are the default;
* capture type, ring/hold geometry, near-capture action modes, collision
  taxonomy, and an A4-compatible last-N-step physical trace are retained.

The evaluator never mutates a training bundle and is safe to run on CPU.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np
import torch

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.continuous.formal_config import scene_config
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.trainer import load_config, set_global_config
from tools.audit_cf3_success_consolidation import (
    action_rows,
    max_hold,
    physical_snapshot,
    summarize_trace,
)
from tools.probe_cf3_policy_temperature import apply_collision_semantics_override
from tools.run_continuous_ctde_training import _evader_actions_for_env


DEFAULT_CHECKPOINT = Path(
    "/home/yjq/rl/CoCap1/cocap-voradj/runs/"
    "iqn_scratch_200k_20260808/checkpoints/step_125000.pt"
)
DEFAULT_CHECKPOINT_SHA256 = (
    "5a0ad1c1400d0004334669908c85db6f7b8496fb2987f36f992040c26bd0344d"
)
DEFAULT_SEED = 2026081201
DEFAULT_COLLISION_SEMANTICS = "synchronized_swept_v1"
TRACE_SCHEMA = "cf3_success_consolidation.frames.v1"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_hash(path: str | Path, expected: str) -> str:
    normalized = str(expected).strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("expected checkpoint SHA-256 must be 64 lowercase hex characters")
    actual = checkpoint_sha256(path)
    if actual != normalized:
        raise ValueError(
            f"IQN checkpoint SHA-256 mismatch: expected={normalized}, actual={actual}"
        )
    return actual


def episode_seeds(base: int, episodes: int) -> list[int]:
    """Return exact paired seeds; intentionally contains no historical offset."""
    if int(episodes) <= 0:
        raise ValueError("episodes must be positive")
    return [int(base) + index for index in range(int(episodes))]


def action_grid_from_config(config: Mapping[str, Any]) -> list[tuple[float, float]]:
    pursuer = dict(config.get("pursuer", {}) or {})
    accelerations = [float(value) for value in pursuer.get("a", [])]
    angular_velocities = [float(value) for value in pursuer.get("w", [])]
    grid = [(a, omega) for a in accelerations for omega in angular_velocities]
    if len(accelerations) != 3 or len(angular_velocities) != 3 or len(grid) != 9:
        raise ValueError(
            "fair IQN bridge requires the exact 3x3 pursuer.a x pursuer.w grid"
        )
    if len(set(grid)) != 9:
        raise ValueError("IQN action grid contains duplicate commands")
    return grid


def map_action_index(index: int, grid: Sequence[tuple[float, float]]) -> np.ndarray:
    value = int(index)
    if value < 0 or value >= len(grid):
        raise ValueError(f"IQN action index {value} is outside [0, {len(grid) - 1}]")
    return np.asarray(grid[value], dtype=np.float32)


def validate_corrected_capture_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    env = dict(config.get("env", {}) or {})
    reward = dict(config.get("reward", {}) or {})
    perception = dict(config.get("perception", {}) or {})
    voradj = dict(config.get("voradj", {}) or {})
    action = dict(config.get("action", {}) or {})
    evader = dict(config.get("evader", {}) or {})
    expected = {
        "num_pursuers": int(env.get("num_pursuers", -1)) == 4,
        "num_evaders": int(env.get("num_evaders", -1)) == 1,
        "num_obstacles": int(env.get("num_obstacles", -1)) == 1,
        "horizon_1000": int(env.get("episode_max_length", -1)) == 1000,
        "pre_capture_horizon_1000": int(env.get("pre_capture_max_length", -1)) == 1000,
        "moving_apf": bool(evader.get("autonomous", False)),
        "local_enemy": not bool(perception.get("global_evader_visibility", False)),
        "legacy_voradj": str(voradj.get("perception_topology_version", "")) == "legacy_voradj",
        "k_required_3": int(reward.get("k_required", -1)) == 3,
        "stationary_fallback": bool(reward.get("capture_stationary_enabled", False)),
        "min_active_2": int(reward.get("min_active_pursuers", -1)) == 2,
        "coverage_min_active_2": int(reward.get("coverage_ce_min_active_pursuers", -1)) == 2,
        "k10_effective_role": int(voradj.get("is_pursuing_release_delay_steps", -1)) == 10,
        "capture_terminal": bool(voradj.get("capture_episode_ends_on_capture", False)),
        "continuous_aw": str(action.get("mode", "")) == "acceleration_angular_velocity_body",
        "postfix_collision": str(env.get("collision_semantics", "")) == DEFAULT_COLLISION_SEMANTICS,
    }
    failed = [name for name, passed in expected.items() if not passed]
    if failed:
        raise ValueError(f"not the corrected CF3 capture contract: failed={failed}")
    return {"passed": True, "checks": expected}


def initial_state_hash(env: VorAdjEnv) -> str:
    state = {
        "pursuers": [
            [float(p.x), float(p.y), float(p.theta), float(p.speed)]
            for p in env.pursuers
        ],
        "evaders": [
            [float(e.x), float(e.y), float(e.theta), float(e.speed)]
            for e in env.evaders
        ],
        "obstacles": [
            [float(o.x), float(o.y), float(o.r)]
            for o in env.obstacles
        ],
    }
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_apf_actions(
    env: VorAdjEnv,
    apf_agents: Sequence[ApfAgent],
) -> list[int | None]:
    """Use the current runner APF path and fail closed on an accidental static target."""
    if not bool((env.config.get("evader", {}) or {}).get("autonomous", False)):
        raise ValueError("corrected IQN evaluation requires evader.autonomous=true")
    actions = list(_evader_actions_for_env(env, list(apf_agents)))
    for index, evader in enumerate(env.evaders):
        if not bool(evader.deactivated) and (index >= len(actions) or actions[index] is None):
            raise RuntimeError("current APF returned no action for an active evader")
    return actions


@torch.no_grad()
def select_iqn_actions(
    model: CoCapIQN,
    observations: Sequence[Mapping[str, np.ndarray] | None],
    *,
    device: str,
    grid: Sequence[tuple[float, float]],
    adapter: Any,
) -> tuple[list[np.ndarray | None], list[int | None], np.ndarray, float]:
    active = [index for index, observation in enumerate(observations) if observation is not None]
    commands: list[np.ndarray | None] = [None] * len(observations)
    indices: list[int | None] = [None] * len(observations)
    dense = np.zeros((len(observations), 2), dtype=np.float32)
    if not active:
        return commands, indices, dense, 0.0
    batch = stack_obs([observations[index] for index in active], device)  # type: ignore[list-item]
    selected = model.act(
        batch,
        mode="voradj",
        epsilon=0.0,
        deterministic_quantiles=True,
    ).detach().cpu().tolist()
    rejected = 0
    for pursuer_index, raw_index in zip(active, selected):
        action_index = int(raw_index)
        raw = map_action_index(action_index, grid)
        validated, diagnostics = adapter.validate_with_diagnostics(raw)
        rejected += int(bool(diagnostics.action_rejected))
        command = np.asarray(validated, dtype=np.float32)
        commands[pursuer_index] = command
        indices[pursuer_index] = action_index
        dense[pursuer_index] = command
    return commands, indices, dense, float(rejected / max(len(active), 1))


def capture_flags(
    capture_types: Sequence[str],
    record: Mapping[str, Any],
) -> Dict[str, bool]:
    normal = any(str(value) != "stationary" for value in capture_types)
    stationary = any(str(value) == "stationary" for value in capture_types)
    captured = bool(record.get("captured", False))
    if captured != bool(normal or stationary):
        raise RuntimeError(
            "capture record/event mismatch: "
            f"record={captured}, capture_types={list(capture_types)}"
        )
    return {
        "normal_capture": bool(normal),
        "stationary_capture": bool(stationary),
        "captured": captured,
    }


def ring_metrics(counts: Sequence[int]) -> Dict[str, Any]:
    steps = max(len(counts), 1)
    return {
        "max_num_in_ring": int(max(counts, default=0)),
        "visited_2plus_ring": bool(any(int(value) >= 2 for value in counts)),
        "visited_3plus_ring": bool(any(int(value) >= 3 for value in counts)),
        "fraction_steps_2plus_in_ring": float(
            sum(int(value) >= 2 for value in counts) / steps
        ),
        "fraction_steps_3plus_in_ring": float(
            sum(int(value) >= 3 for value in counts) / steps
        ),
        "max_2plus_ring_hold_steps": max_hold(counts, 2),
        "max_3plus_ring_hold_steps": max_hold(counts, 3),
    }


def _third_distance(snapshot: Mapping[str, Any]) -> float | None:
    distances = sorted(
        float(row["distance_to_enemy"])
        for row in snapshot.get("pursuers", [])
        if row.get("distance_to_enemy") is not None
    )
    return distances[2] if len(distances) >= 3 else None


def is_near_capture_frame(frame: Mapping[str, Any]) -> tuple[bool, Dict[str, Any]]:
    pre_ring = int(frame.get("ring_count_pre", 0))
    post_ring = int(frame.get("ring_count_post_was_active", 0))
    pre_d3 = _third_distance(frame.get("pre", {}) or {})
    post_d3 = _third_distance(frame.get("post_was_active", {}) or {})
    d3_near = bool(
        (pre_d3 is not None and pre_d3 <= 10.5)
        or (post_d3 is not None and post_d3 <= 10.5)
    )
    ring_near = max(pre_ring, post_ring) >= 2
    return bool(ring_near or d3_near), {
        "ring_2plus": ring_near,
        "d3_le_10_5": d3_near,
        "pre_d3": pre_d3,
        "post_d3": post_d3,
    }


def summarize_action_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    grid_size: int = 9,
) -> Dict[str, Any]:
    histogram = {str(index): 0 for index in range(int(grid_size))}
    a_histogram: Counter[str] = Counter()
    omega_histogram: Counter[str] = Counter()
    for row in rows:
        index = int(row["action_index"])
        histogram[str(index)] = histogram.get(str(index), 0) + 1
        a_histogram[format(float(row["a"]), ".9g")] += 1
        omega_histogram[format(float(row["omega"]), ".9g")] += 1
    count = len(rows)

    def rate(predicate) -> float:
        return float(np.mean([bool(predicate(row)) for row in rows])) if rows else 0.0

    return {
        "agent_steps": int(count),
        "action_index_histogram": histogram,
        "a_value_histogram": dict(sorted(a_histogram.items())),
        "omega_value_histogram": dict(sorted(omega_histogram.items())),
        "accelerate_rate": rate(lambda row: float(row["a"]) > 1e-9),
        "coast_rate": rate(lambda row: abs(float(row["a"])) <= 1e-9),
        "brake_rate": rate(lambda row: float(row["a"]) < -1e-9),
        "left_rate": rate(lambda row: float(row["omega"]) > 1e-9),
        "straight_rate": rate(lambda row: abs(float(row["omega"])) <= 1e-9),
        "right_rate": rate(lambda row: float(row["omega"]) < -1e-9),
        "a_saturation_rate": rate(lambda row: row.get("a_saturated", False)),
        "omega_saturation_rate": rate(lambda row: row.get("omega_saturated", False)),
        "omega_sign_flip_rate": rate(lambda row: row.get("omega_sign_flip", False)),
        "action_repeat_rate": rate(lambda row: row.get("action_repeat", False)),
    }


def near_capture_action_modes(
    frames: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    selected: list[Mapping[str, Any]] = []
    ring_frames = d3_frames = 0
    for frame in frames:
        near, reasons = is_near_capture_frame(frame)
        if not near:
            continue
        selected.append(frame)
        ring_frames += int(bool(reasons["ring_2plus"]))
        d3_frames += int(bool(reasons["d3_le_10_5"]))
    rows = [row for frame in selected for row in frame.get("actions", [])]
    members = [row for row in rows if bool(row.get("pre_in_ring", False))]
    nonmembers = [row for row in rows if not bool(row.get("pre_in_ring", False))]
    return {
        "definition": "max(pre,post-was-active) ring>=2 OR pre/post third distance<=10.5m",
        "near_capture_steps": int(len(selected)),
        "ring_2plus_steps": int(ring_frames),
        "d3_le_10_5_steps": int(d3_frames),
        "all": summarize_action_rows(rows),
        "pre_ring_members": summarize_action_rows(members),
        "pre_non_ring_agents": summarize_action_rows(nonmembers),
    }


def last50_trace_payload(
    frames: Sequence[Mapping[str, Any]],
    *,
    requested_tail_steps: int = 50,
) -> Dict[str, Any]:
    """Package a tail with the exact frame vocabulary consumed by A4."""
    kept = list(frames)[-int(requested_tail_steps):]
    return {
        "schema": TRACE_SCHEMA,
        "requested_tail_steps": int(requested_tail_steps),
        "frames": kept,
        "trace_summary": summarize_trace(kept),
    }


def _augment_action_rows(
    rows: list[Dict[str, Any]],
    indices: Sequence[int | None],
    pre: Mapping[str, Any],
    post: Mapping[str, Any],
    previous_index: MutableMapping[int, int],
) -> list[Dict[str, Any]]:
    pre_by_id = {int(row["id"]): row for row in pre.get("pursuers", [])}
    post_by_id = {int(row["id"]): row for row in post.get("pursuers", [])}
    for row in rows:
        pursuer_id = int(row["pursuer_id"])
        action_index = indices[pursuer_id]
        if action_index is None:
            raise RuntimeError(f"missing IQN action index for active pursuer {pursuer_id}")
        previous = previous_index.get(pursuer_id)
        row["action_index"] = int(action_index)
        row["action_repeat"] = bool(previous is not None and previous == int(action_index))
        row["pre_in_ring"] = bool(pre_by_id.get(pursuer_id, {}).get("in_ring_8_10_5", False))
        row["post_in_ring"] = bool(post_by_id.get(pursuer_id, {}).get("in_ring_8_10_5", False))
        row["pre_distance_to_enemy"] = pre_by_id.get(pursuer_id, {}).get("distance_to_enemy")
        row["post_distance_to_enemy"] = post_by_id.get(pursuer_id, {}).get("distance_to_enemy")
        previous_index[pursuer_id] = int(action_index)
    return rows


def run_episode(
    model: CoCapIQN,
    root_config: Mapping[str, Any],
    *,
    seed: int,
    device: str,
    max_steps: int,
    tail_steps: int,
) -> Dict[str, Any]:
    config = scene_config(dict(root_config), "capture")
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=int(seed))
    observations = list(env.reset())
    state_hash = initial_state_hash(env)
    apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    grid = action_grid_from_config(config)
    tail: deque[Dict[str, Any]] = deque(maxlen=int(tail_steps))
    diagnostic_frames: list[Dict[str, Any]] = []
    capture_types: list[str] = []
    ring_counts: list[int] = []
    ring_counts_was_active: list[int] = []
    collision_ever = False
    previous_omega: Dict[int, float] = {}
    previous_index: Dict[int, int] = {}
    rejected_rates: list[float] = []
    horizon = min(int(max_steps), int(config["env"]["episode_max_length"]))

    for episode_step in range(1, horizon + 1):
        active_ids = [int(p.id) for p in env.pursuers if not p.deactivated]
        pre = physical_snapshot(env, pursuer_ids=active_ids)
        commands, indices, dense, rejected = select_iqn_actions(
            model,
            observations,
            device=device,
            grid=grid,
            adapter=env.action_adapter,
        )
        base_rows = action_rows(env, dense, active_ids, previous_omega)
        outcome = env.step(commands, current_apf_actions(env, apf_agents))
        events = [dict(event) for event in getattr(env, "last_capture_events", [])]
        capture_types.extend(str(event.get("capture_type", "unknown")) for event in events)
        collision_events = [dict(event) for event in getattr(env, "last_collision_events", [])]
        collision_now = bool(collision_events)
        collision_ever = collision_ever or collision_now
        post_alive = physical_snapshot(env)
        post_was_active = physical_snapshot(env, pursuer_ids=active_ids)
        actions = _augment_action_rows(
            base_rows, indices, pre, post_was_active, previous_index,
        )
        ring_counts.append(int(post_alive["ring_count"]))
        ring_counts_was_active.append(int(post_was_active["ring_count"]))
        rejected_rates.append(float(rejected))
        frame = {
            "episode_step": int(episode_step),
            "pre": pre,
            "post": post_alive,
            "post_was_active": post_was_active,
            "ring_count_pre": int(pre["ring_count"]),
            "ring_count_post_alive": int(post_alive["ring_count"]),
            "ring_count_post_was_active": int(post_was_active["ring_count"]),
            "actions": actions,
            "action_rejected_rate": float(rejected),
            "capture_events": events,
            "collision_events": collision_events,
            "collision_now": bool(collision_now),
            "dones": [bool(value) for value in outcome.dones],
        }
        diagnostic_frames.append(frame)
        tail.append(frame)
        observations = list(outcome.observations)
        if all(outcome.dones):
            break

    record = env.episode_record(task="capture")
    flags = capture_flags(capture_types, record)
    collision = bool(collision_ever or record.get("collision_event", False))
    frames = list(tail)
    return {
        "seed": int(seed),
        "environment_seed_used": int(seed),
        "initial_state_hash": state_hash,
        "length": int(record.get("episode_length", len(diagnostic_frames))),
        **flags,
        "collision": collision,
        "collision_semantics": str(record.get("collision_semantics", "")),
        "collision_type_counts": dict(record.get("collision_type_counts", {}) or {}),
        "capture_types": capture_types,
        **ring_metrics(ring_counts),
        "max_num_in_ring_was_active": int(max(ring_counts_was_active, default=0)),
        "max_3plus_ring_hold_steps_was_active": max_hold(ring_counts_was_active, 3),
        "mean_action_rejected_rate": float(np.mean(rejected_rates)) if rejected_rates else 0.0,
        "action_modes_all_steps": summarize_action_rows([
            row for frame in diagnostic_frames for row in frame["actions"]
        ]),
        "near_capture_action_modes": near_capture_action_modes(diagnostic_frames),
        "last50_trace": last50_trace_payload(
            frames, requested_tail_steps=int(tail_steps),
        ),
    }


def _sum_histograms(records: Sequence[Mapping[str, Any]], path: str) -> Dict[str, int]:
    result = {str(index): 0 for index in range(9)}
    for record in records:
        value: Any = record
        for key in path.split("."):
            value = (value or {}).get(key, {})
        for key, count in dict(value or {}).items():
            result[str(key)] = result.get(str(key), 0) + int(count)
    return result


def summarize(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    episodes = len(records)
    if not episodes:
        return {"episodes": 0}

    def count(key: str) -> int:
        return int(sum(bool(record.get(key, False)) for record in records))

    def mean(key: str) -> float:
        return float(np.mean([float(record.get(key, 0.0)) for record in records]))

    collision_types: Counter[str] = Counter()
    for record in records:
        collision_types.update({
            str(key): int(value)
            for key, value in (record.get("collision_type_counts", {}) or {}).items()
        })
    return {
        "episodes": int(episodes),
        "normal_capture_count": count("normal_capture"),
        "stationary_capture_count": count("stationary_capture"),
        "capture_count": count("captured"),
        "collision_count": count("collision"),
        "normal_capture_rate": count("normal_capture") / episodes,
        "stationary_capture_rate": count("stationary_capture") / episodes,
        "capture_rate": count("captured") / episodes,
        "collision_rate": count("collision") / episodes,
        "episodes_with_2plus_ring": count("visited_2plus_ring"),
        "episodes_with_3plus_ring": count("visited_3plus_ring"),
        "episode_2plus_ring_rate": count("visited_2plus_ring") / episodes,
        "episode_3plus_ring_rate": count("visited_3plus_ring") / episodes,
        "mean_fraction_steps_2plus_in_ring": mean("fraction_steps_2plus_in_ring"),
        "mean_fraction_steps_3plus_in_ring": mean("fraction_steps_3plus_in_ring"),
        "max_3plus_ring_hold_steps": int(max(
            int(record.get("max_3plus_ring_hold_steps", 0)) for record in records
        )),
        "mean_episode_length": mean("length"),
        "collision_type_counts": dict(sorted(collision_types.items())),
        "action_index_histogram_all_steps": _sum_histograms(
            records, "action_modes_all_steps.action_index_histogram",
        ),
        "action_index_histogram_near_capture": _sum_histograms(
            records, "near_capture_action_modes.all.action_index_histogram",
        ),
        "near_capture_steps": int(sum(
            int(record.get("near_capture_action_modes", {}).get("near_capture_steps", 0))
            for record in records
        )),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    actual_hash = verify_checkpoint_hash(checkpoint, args.expected_checkpoint_sha256)
    root_config = load_config(str(config_path))
    evaluation_config = apply_collision_semantics_override(
        root_config, str(args.collision_semantics),
    )
    capture_config = scene_config(evaluation_config, "capture")
    contract_audit = validate_corrected_capture_contract(capture_config)
    model = CoCapIQN.load(str(checkpoint), device=str(args.device)).eval()
    if int(model.config.action_size) != 9:
        raise ValueError(f"historical IQN action_size must be 9, got {model.config.action_size}")
    if int(model.config.self_feature_dim) != int(capture_config["perception"]["self_feature_dim"]):
        raise ValueError("IQN/current observation self_feature_dim mismatch")
    records = [
        run_episode(
            model,
            evaluation_config,
            seed=seed,
            device=str(args.device),
            max_steps=int(args.max_steps),
            tail_steps=int(args.tail_steps),
        )
        for seed in episode_seeds(int(args.seed), int(args.episodes))
    ]
    return {
        "schema_version": 1,
        "kind": "iqn_corrected_cf3_capture_eval",
        "created_at": now(),
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_hash,
        "checkpoint_hash_verified": True,
        "device": str(args.device),
        "environment_seed_contract": "exact base+episode_index; no legacy +43 offset",
        "environment_seed_base": int(args.seed),
        "episode_seeds": episode_seeds(int(args.seed), int(args.episodes)),
        "max_steps": min(int(args.max_steps), int(capture_config["env"]["episode_max_length"])),
        "policy_contract": {
            "epsilon": 0.0,
            "quantiles": "fixed_midpoint_32",
            "action_bridge": "IQN index -> pursuer.a x pursuer.w -> current AW adapter",
            "action_grid": [list(value) for value in action_grid_from_config(capture_config)],
        },
        "evader_contract": "current moving APF via _evader_actions_for_env; fail closed on None",
        "collision_semantics": str(args.collision_semantics),
        "contract_audit": contract_audit,
        "summary": summarize(records),
        "records": records,
    }


def _write_payload(payload: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")

    def json_default(value: Any):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, set):
            return sorted(value)
        raise TypeError(f"cannot serialize {type(value).__name__}")

    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--expected-checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--tail-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--collision-semantics",
        choices=(DEFAULT_COLLISION_SEMANTICS,),
        default=DEFAULT_COLLISION_SEMANTICS,
        help="Fixed corrected simulator contract; legacy semantics are intentionally rejected.",
    )
    args = parser.parse_args()
    for name in ("episodes", "max_steps", "tail_steps"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    payload = evaluate(args)
    _write_payload(payload, Path(args.output))
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
