#!/usr/bin/env python3
"""Paired formal100 Stage2 cross/retention evaluation for IQN-VXY.

This evaluator is deliberately inference-only.  It runs checkpoints under one
resolved Stage2 8P/2E/2obs environment, uses epsilon=0 and midpoint-32 IQN
quantiles, and writes compact per-episode records instead of full frame dumps.
The same exact seed is used for every checkpoint/scenario pair.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.dynamics.continuous_action import body_to_world, vxy9_body_grid
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.trainer import set_global_config
from tools.batch_rollouts import coverage_rollout_config
from tools.rollout_voradj_visual import act_evaders, act_pursuers, load_config, scenario_config


SCHEMA = "iqn-vxy-stage2-cross-retention-formal100-v1"
SCENARIOS = ("capture", "coverage", "mix")
BINARY_METRICS = {
    "capture": ("captured", "normal_capture", "stationary_capture", "collision"),
    "coverage": ("coverage_success", "collision", "cv015_success"),
    "mix": (
        "captured", "normal_capture", "stationary_capture", "coverage_success",
        "collision", "post_capture_survival", "post_capture_settled",
    ),
}
CONTINUOUS_METRICS = {
    "capture": (
        "length", "first_detect_to_capture_steps", "first_2plus_ring_step",
        "first_3plus_ring_step", "two_plus_to_three_plus_steps",
        "support_to_direct_upgrade_steps_mean", "support_to_pursuing_upgrade_steps_mean",
        "direct_speed_mean", "support_speed_mean",
    ),
    "coverage": ("length", "final_cv", "best_cv", "centroid_rms", "centroid_max"),
    "mix": (
        "length", "post_capture_steps", "post_capture_ce_step", "final_cv",
        "best_cv", "centroid_rms", "centroid_max", "first_detect_to_capture_steps",
        "two_plus_to_three_plus_steps", "support_to_direct_upgrade_steps_mean",
        "support_to_pursuing_upgrade_steps_mean", "direct_speed_mean", "support_speed_mean",
    ),
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean_present(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [finite_or_none(row.get(key)) for row in rows]
    kept = [value for value in values if value is not None]
    return float(np.mean(kept)) if kept else None


def max_hold(values: Sequence[int], threshold: int) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if int(value) >= int(threshold) else 0
        best = max(best, current)
    return int(best)


def first_step(values: Sequence[int], threshold: int) -> int | None:
    return next((index + 1 for index, value in enumerate(values) if int(value) >= int(threshold)), None)


def min_surface_clearance(env: VorAdjEnv, pursuer_index: int) -> float:
    pursuer = env.pursuers[pursuer_index]
    point = np.asarray([float(pursuer.x), float(pursuer.y)], dtype=float)
    candidates: list[float] = [
        float(pursuer.x), float(pursuer.y),
        float(env.width) - float(pursuer.x), float(env.height) - float(pursuer.y),
    ]
    for other_index, other in enumerate(env.pursuers):
        if other_index == pursuer_index or other.deactivated:
            continue
        other_point = np.asarray([float(other.x), float(other.y)], dtype=float)
        candidates.append(float(np.linalg.norm(point - other_point) - float(pursuer.r) - float(other.r)))
    for obstacle in env.obstacles:
        obstacle_point = np.asarray([float(obstacle.x), float(obstacle.y)], dtype=float)
        candidates.append(float(np.linalg.norm(point - obstacle_point) - float(pursuer.r) - float(obstacle.r)))
    return float(min(candidates))


def ring_count(env: VorAdjEnv) -> int:
    """Legacy outer-annulus diagnostic; NOT same-target capture closure."""
    active_evaders = [evader for evader in env.evaders if not evader.deactivated]
    if not active_evaders:
        return 0
    counts: list[int] = []
    for evader in active_evaders:
        target = np.asarray([float(evader.x), float(evader.y)], dtype=float)
        count = 0
        for pursuer in env.pursuers:
            if pursuer.deactivated:
                continue
            point = np.asarray([float(pursuer.x), float(pursuer.y)], dtype=float)
            distance = float(np.linalg.norm(point - target))
            count += int(8.0 <= distance < 10.5)
        counts.append(count)
    return int(max(counts, default=0))


def role_audit_template(num_pursuers: int) -> dict[str, Any]:
    return {
        "first_direct": [None] * num_pursuers,
        "first_pursuing": [None] * num_pursuers,
        "first_support_neighbor": [None] * num_pursuers,
        "first_support_target": [None] * num_pursuers,
        "direct_speeds": [], "support_speeds": [],
        "support_enemy_progress": [], "support_friend_progress": [],
        "support_capture_rewards": [], "support_coverage_rewards": [],
        "support_competing_reward_steps": 0, "support_agent_steps": 0,
        "direct_velocity_errors": [], "support_velocity_errors": [],
        "direct_rate_limit_pressure": 0, "support_rate_limit_pressure": 0,
        "direct_agent_steps": 0,
        "support_near_hazard_speeds": [], "support_clear_speeds": [],
        "support_near_hazard_steps": 0,
    }


def support_context_snapshot(env: VorAdjEnv) -> list[dict[str, Any]]:
    """Resolve support friends/targets by the same VCT-LS path as reward."""
    pursuer_positions = np.asarray(
        [[float(p.x), float(p.y)] for p in env.pursuers], dtype=float,
    )
    evader_positions = np.asarray(
        [[float(e.x), float(e.y)] for e in env.evaders], dtype=float,
    ) if env.evaders else np.zeros((0, 2), dtype=float)
    data = env._capture_voronoi_map(pursuer_positions, evader_positions)
    raw_labels = env._raw_task_labels_from_map(data)
    labels = env._task_labels_from_map(data, update_effective=False, raw_labels=raw_labels)
    contexts: list[dict[str, Any]] = []
    for index in range(len(env.pursuers)):
        friend_ids = env._support_pursuing_friend_ids(index, labels, data)
        target_ids: set[int] = set()
        for friend_id in friend_ids:
            target_ids.update(
                env._vct_ls_direct_enemy_ids_for_pursuer(
                    friend_id, pursuer_positions, evader_positions,
                )
            )
        contexts.append({
            "friend_ids": [int(value) for value in friend_ids],
            "target_ids": sorted(int(value) for value in target_ids),
            "pursuer_positions": pursuer_positions,
            "evader_positions": evader_positions,
        })
    return contexts


def update_role_audit(
    audit: dict[str, Any], infos: Sequence[Mapping[str, Any]], actions: Sequence[int | None],
    speeds: Sequence[float], velocities: Sequence[np.ndarray], thetas: Sequence[float],
    clearances: Sequence[float], contexts: Sequence[Mapping[str, Any]],
    after_pursuer_positions: np.ndarray, after_evader_positions: np.ndarray,
    step: int, grid: np.ndarray | None, acceleration_budget: float,
) -> None:
    for index, info in enumerate(infos):
        metadata = dict(info.get("replay_metadata", {}) or {})
        direct = int(metadata.get("vct_ls_direct_enemy_count", 0) or 0) > 0
        pursuing = bool(metadata.get("effective_pursuing", False))
        context = contexts[index]
        friend_ids = [int(value) for value in context.get("friend_ids", [])]
        target_ids = [int(value) for value in context.get("target_ids", [])]
        support_neighbor = bool(friend_ids)
        support = bool(metadata.get("support_candidate", False))
        support_target = bool(support and target_ids)
        for key, condition in (
            ("first_direct", direct), ("first_pursuing", pursuing),
            ("first_support_neighbor", support_neighbor), ("first_support_target", support_target),
        ):
            if condition and audit[key][index] is None:
                audit[key][index] = int(step)
        action = actions[index] if index < len(actions) else None
        velocity_error = None
        if action is not None and grid is not None:
            desired_world = body_to_world(grid[int(action)], float(thetas[index]))
            velocity_error = float(np.linalg.norm(desired_world - np.asarray(velocities[index], dtype=float)))
        if direct:
            audit["direct_agent_steps"] += 1
            audit["direct_speeds"].append(float(speeds[index]))
            if velocity_error is not None:
                audit["direct_velocity_errors"].append(velocity_error)
                audit["direct_rate_limit_pressure"] += int(velocity_error > acceleration_budget + 1e-9)
        if support:
            audit["support_agent_steps"] += 1
            audit["support_speeds"].append(float(speeds[index]))
            before_p = np.asarray(context["pursuer_positions"], dtype=float)
            before_e = np.asarray(context["evader_positions"], dtype=float)
            if target_ids:
                target_id = min(
                    target_ids,
                    key=lambda value: np.linalg.norm(after_pursuer_positions[index] - after_evader_positions[value]),
                )
                enemy_progress = float(
                    np.linalg.norm(before_p[index] - before_e[target_id])
                    - np.linalg.norm(after_pursuer_positions[index] - after_evader_positions[target_id])
                )
                audit["support_enemy_progress"].append(enemy_progress)
            if friend_ids:
                before_friend_distance = min(float(np.linalg.norm(before_p[index] - before_p[value])) for value in friend_ids)
                after_friend_distance = min(float(np.linalg.norm(after_pursuer_positions[index] - after_pursuer_positions[value])) for value in friend_ids)
                audit["support_friend_progress"].append(before_friend_distance - after_friend_distance)
            capture_reward = float(metadata.get("reward_support_blend_capture", 0.0) or 0.0)
            coverage_reward = float(metadata.get("reward_support_blend_coverage", 0.0) or 0.0)
            audit["support_capture_rewards"].append(capture_reward)
            audit["support_coverage_rewards"].append(coverage_reward)
            audit["support_competing_reward_steps"] += int(capture_reward * coverage_reward < 0.0)
            if velocity_error is not None:
                audit["support_velocity_errors"].append(velocity_error)
                audit["support_rate_limit_pressure"] += int(velocity_error > acceleration_budget + 1e-9)
            if float(clearances[index]) <= 2.0:
                audit["support_near_hazard_steps"] += 1
                audit["support_near_hazard_speeds"].append(float(speeds[index]))
            else:
                audit["support_clear_speeds"].append(float(speeds[index]))


def finish_role_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    def first_any(key: str) -> int | None:
        values = [value for value in audit[key] if value is not None]
        return int(min(values)) if values else None

    def delays(target: str) -> list[int]:
        values: list[int] = []
        for start, end in zip(audit["first_support_neighbor"], audit[target]):
            if start is not None and end is not None and int(end) >= int(start):
                values.append(int(end) - int(start))
        return values

    direct_delays = delays("first_direct")
    pursuing_delays = delays("first_pursuing")
    support_steps = int(audit["support_agent_steps"])
    direct_steps = int(audit["direct_agent_steps"])
    return {
        "first_direct_enemy_seen_step": first_any("first_direct"),
        "first_support_pursuing_neighbor_step": first_any("first_support_neighbor"),
        "first_support_neighbor_visible_target_step": first_any("first_support_target"),
        "agents_ever_direct": int(sum(value is not None for value in audit["first_direct"])),
        "agents_ever_support": int(sum(value is not None for value in audit["first_support_neighbor"])),
        "support_to_direct_upgrade_count": len(direct_delays),
        "support_to_direct_upgrade_steps_mean": float(np.mean(direct_delays)) if direct_delays else None,
        "support_to_pursuing_upgrade_count": len(pursuing_delays),
        "support_to_pursuing_upgrade_steps_mean": float(np.mean(pursuing_delays)) if pursuing_delays else None,
        "direct_speed_mean": float(np.mean(audit["direct_speeds"])) if audit["direct_speeds"] else None,
        "support_speed_mean": float(np.mean(audit["support_speeds"])) if audit["support_speeds"] else None,
        "support_enemy_progress_mean": float(np.mean(audit["support_enemy_progress"])) if audit["support_enemy_progress"] else None,
        "support_friend_progress_mean": float(np.mean(audit["support_friend_progress"])) if audit["support_friend_progress"] else None,
        "support_capture_reward_mean": float(np.mean(audit["support_capture_rewards"])) if audit["support_capture_rewards"] else None,
        "support_coverage_reward_mean": float(np.mean(audit["support_coverage_rewards"])) if audit["support_coverage_rewards"] else None,
        "support_reward_sign_competition_rate": float(audit["support_competing_reward_steps"] / max(support_steps, 1)),
        "direct_velocity_error_mean": float(np.mean(audit["direct_velocity_errors"])) if audit["direct_velocity_errors"] else None,
        "support_velocity_error_mean": float(np.mean(audit["support_velocity_errors"])) if audit["support_velocity_errors"] else None,
        "direct_rate_limit_pressure_rate": float(audit["direct_rate_limit_pressure"] / max(direct_steps, 1)),
        "support_rate_limit_pressure_rate": float(audit["support_rate_limit_pressure"] / max(support_steps, 1)),
        "support_near_hazard_rate": float(audit["support_near_hazard_steps"] / max(support_steps, 1)),
        "support_speed_near_hazard_mean": float(np.mean(audit["support_near_hazard_speeds"])) if audit["support_near_hazard_speeds"] else None,
        "support_speed_clear_mean": float(np.mean(audit["support_clear_speeds"])) if audit["support_clear_speeds"] else None,
        "support_agent_steps": support_steps,
        "direct_agent_steps": direct_steps,
    }


def prepare_config(
    base: Mapping[str, Any], scenario: str, episode_index: int,
    *, num_evaders: int = 2, capture_horizon: int = 1000,
    coverage_horizon: int = 1500, mix_horizon: int = 2500,
) -> tuple[dict[str, Any], str, int]:
    if scenario == "capture":
        return scenario_config(dict(base), int(num_evaders), task_name="voradj"), "", int(capture_horizon)
    if scenario == "coverage":
        coverage_base, source = coverage_rollout_config(dict(base), int(episode_index))
        return scenario_config(coverage_base, 0, task_name="voradj_coverage"), source, int(coverage_horizon)
    return scenario_config(dict(base), int(num_evaders), mix_mode=True, task_name="voradj"), "", int(mix_horizon)


@torch.no_grad()
def run_episode(
    model: CoCapIQN, base: Mapping[str, Any], scenario: str, seed: int,
    episode_index: int, device: str, *, action_contract: str = "vxy9",
    num_evaders: int = 2, capture_horizon: int = 1000,
    coverage_horizon: int = 1500, mix_horizon: int = 2500,
) -> dict[str, Any]:
    cfg, coverage_source, horizon = prepare_config(
        base, scenario, episode_index, num_evaders=num_evaders,
        capture_horizon=capture_horizon, coverage_horizon=coverage_horizon,
        mix_horizon=mix_horizon,
    )
    set_global_config(cfg)
    env = VorAdjEnv(copy.deepcopy(cfg), seed=int(seed))
    observations = env.reset()
    from cocap_voradj.evaluation.mission_events import MissionEventTracker, snapshot
    mission_tracker = MissionEventTracker(env.pursuers[0].dt * env.pursuers[0].N)
    mission_tracker.observe(snapshot(env, observations), 0)
    apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    grid = (
        vxy9_body_grid(float((cfg.get("pursuer", {}) or {}).get("v_max", 3.0)))
        if str(action_contract) == "vxy9" else None
    )
    decision_dt = float(env.pursuers[0].dt) * int(env.pursuers[0].N)
    acceleration_limit = float((cfg.get("action", {}) or {}).get("servo_acceleration_limit", 0.4))
    acceleration_budget = acceleration_limit * decision_dt
    audit = role_audit_template(len(env.pursuers))
    capture_types: list[str] = []
    ring_counts: list[int] = []
    cv_values: list[float] = []
    capture_step: int | None = None
    post_capture_collision = False
    captured_before = False
    episode_return = np.zeros(len(env.pursuers), dtype=np.float64)

    for step in range(1, int(horizon) + 1):
        set_global_config(cfg)
        speeds = [float(p.speed) for p in env.pursuers]
        velocities = [np.asarray(p.velocity, dtype=float).copy() for p in env.pursuers]
        thetas = [float(p.theta) for p in env.pursuers]
        clearances = [min_surface_clearance(env, index) for index in range(len(env.pursuers))]
        contexts = support_context_snapshot(env) if not captured_before else []
        actions = act_pursuers(model, observations, device)
        outcome = env.step(actions, act_evaders(env, apf_agents))
        episode_return += np.asarray(outcome.rewards, dtype=np.float64)
        if not captured_before:
            after_pursuer_positions = np.asarray(
                [[float(p.x), float(p.y)] for p in env.pursuers], dtype=float,
            )
            after_evader_positions = np.asarray(
                [[float(e.x), float(e.y)] for e in env.evaders], dtype=float,
            ) if env.evaders else np.zeros((0, 2), dtype=float)
            update_role_audit(
                audit, outcome.infos, actions, speeds, velocities, thetas, clearances, contexts,
                after_pursuer_positions, after_evader_positions, step, grid, acceleration_budget,
            )
        events = list(getattr(env, "last_capture_events", []) or [])
        capture_types.extend(str(event.get("capture_type", "unknown")) for event in events)
        captured_now = bool(env.evaders and all(evader.deactivated and not evader.collision for evader in env.evaders))
        if captured_now and capture_step is None:
            capture_step = int(step)
        if captured_before and any(bool(getattr(pursuer, "collision", False)) for pursuer in env.pursuers):
            post_capture_collision = True
        captured_before = bool(captured_before or captured_now)
        if not captured_before:
            ring_counts.append(ring_count(env))
        cv = finite_or_none((getattr(env, "last_distribution_metrics", {}) or {}).get("area_cv"))
        if (scenario == "coverage" or captured_before) and cv is not None:
            cv_values.append(cv)
        observations = outcome.observations
        mission_tracker.observe(snapshot(env, observations), step, env.last_capture_events)
        if scenario == "capture" and captured_now:
            break
        if all(outcome.dones):
            break

    record = env.episode_record(task=scenario)
    metrics = dict(getattr(env, "last_voradj_metrics", {}) or {})
    captured = bool(record.get("captured", False))
    # A multi-evader episode can contain a partial capture event without satisfying
    # the formal all-evaders capture criterion. Only classify capture type after
    # the episode-level capture contract is met.
    normal = bool(captured and any(value != "stationary" for value in capture_types))
    stationary = bool(captured and any(value == "stationary" for value in capture_types))
    if captured and not capture_types:
        raise RuntimeError(f"captured episode has no capture event seed={seed} scenario={scenario}")
    collision = bool(record.get("collision_event", False))
    coverage_success = bool(
        record.get("coverage_strict_success", False)
        or metrics.get("post_capture_coverage_success", False)
        or metrics.get("pure_coverage_success", False)
    )
    post_capture_survival = bool(
        captured and record.get("all_pursuers_active", False) and not post_capture_collision
    )
    role_result = finish_role_audit(audit)
    first_detect = role_result["first_direct_enemy_seen_step"]
    first_2plus = first_step(ring_counts, 2)
    first_3plus = first_step(ring_counts, 3)
    result = {
        "seed": int(seed), "episode_index": int(episode_index), "scenario": scenario,
        "mission_events": mission_tracker.finish(step),
        "legacy_latency_warning": "outer annulus / first-ever agent diagnostics; use mission_events for same-target closure/support",
        "coverage_init_source": coverage_source, "length": int(record.get("length", env.episode_step)),
        "captured": captured, "normal_capture": bool(normal), "stationary_capture": bool(stationary),
        "capture_types": capture_types, "capture_step": capture_step,
        "first_detect_to_capture_steps": (
            int(capture_step - first_detect)
            if capture_step is not None and first_detect is not None and capture_step >= first_detect
            else None
        ),
        "coverage_success": coverage_success, "collision": collision,
        "boundary_collision": bool(record.get("boundary_collision_event", False)),
        "agent_agent_collision": bool(record.get("agent_agent_collision_event", False)),
        "collision_type_counts": dict(record.get("collision_type_counts", {}) or {}),
        "all_pursuers_active": bool(record.get("all_pursuers_active", False)),
        "post_capture_survival": post_capture_survival,
        "post_capture_collision": bool(post_capture_collision),
        "post_capture_settled": bool(captured and record.get("episode_end_stationary", False)),
        "post_capture_steps": int(getattr(env, "post_capture_step", 0)),
        "post_capture_ce_step": (
            int(metrics.get("post_capture_coverage_step", -1))
            if int(metrics.get("post_capture_coverage_step", -1)) >= 0 else None
        ),
        "post_capture_window_expired": bool(metrics.get("post_capture_window_expired", False)),
        "cv015_success": bool(record.get("coverage_cv015_success", False)),
        "final_cv": finite_or_none(record.get("coverage_strict_area_cv")),
        "best_cv": float(min(cv_values)) if cv_values else None,
        "centroid_rms": finite_or_none(record.get("coverage_ce_center_rms")),
        "centroid_max": finite_or_none(record.get("coverage_ce_center_max")),
        "episode_end_mean_speed": finite_or_none(record.get("episode_end_mean_speed")),
        "episode_end_max_speed": finite_or_none(record.get("episode_end_max_speed")),
        "episode_return_mean": float(np.mean(episode_return)),
        "episode_return_sum": float(np.sum(episode_return)),
        "first_2plus_ring_step": first_2plus,
        "first_3plus_ring_step": first_3plus,
        "two_plus_to_three_plus_steps": (
            int(first_3plus - first_2plus)
            if first_2plus is not None and first_3plus is not None and first_3plus >= first_2plus
            else None
        ),
        "steps_2plus_ring": int(sum(value >= 2 for value in ring_counts)),
        "steps_3plus_ring": int(sum(value >= 3 for value in ring_counts)),
        "max_2plus_ring_hold": max_hold(ring_counts, 2),
        "max_3plus_ring_hold": max_hold(ring_counts, 3),
        **role_result,
    }
    return result


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    episodes = len(records)
    if not episodes:
        return {"episodes": 0}
    collision_types: Counter[str] = Counter()
    for row in records:
        collision_types.update({str(key): int(value) for key, value in dict(row.get("collision_type_counts", {}) or {}).items()})
    rate_keys = sorted({key for keys in BINARY_METRICS.values() for key in keys} | {
        "boundary_collision", "agent_agent_collision", "all_pursuers_active",
        "post_capture_collision", "post_capture_window_expired",
    })
    mean_keys = sorted({key for keys in CONTINUOUS_METRICS.values() for key in keys} | {
        "episode_end_mean_speed", "episode_end_max_speed", "episode_return_mean",
        "steps_2plus_ring", "steps_3plus_ring", "max_2plus_ring_hold", "max_3plus_ring_hold",
        "first_direct_enemy_seen_step", "first_support_pursuing_neighbor_step",
        "first_support_neighbor_visible_target_step", "agents_ever_direct", "agents_ever_support",
        "first_detect_to_capture_steps", "two_plus_to_three_plus_steps",
        "support_to_direct_upgrade_steps_mean", "support_to_pursuing_upgrade_steps_mean",
        "direct_speed_mean", "support_speed_mean", "support_enemy_progress_mean",
        "support_friend_progress_mean", "support_capture_reward_mean", "support_coverage_reward_mean",
        "support_reward_sign_competition_rate", "direct_velocity_error_mean",
        "support_velocity_error_mean", "direct_rate_limit_pressure_rate",
        "support_rate_limit_pressure_rate", "support_near_hazard_rate",
        "support_speed_near_hazard_mean", "support_speed_clear_mean",
    })
    from cocap_voradj.evaluation.mission_events import summarize_events
    return {
        "episodes": episodes,
        "legacy_latency_warning": "Use mission_event_summary; old outer-ring and first-ever means are not same-target closure/support.",
        "mission_event_summary": summarize_events([e for row in records for e in row.get("mission_events", {}).get("events", [])]),
        "coverage_init_counts": dict(Counter(str(row.get("coverage_init_source", "")) for row in records)),
        **{f"{key}_rate": float(np.mean([bool(row.get(key, False)) for row in records])) for key in rate_keys},
        **{f"mean_{key}": mean_present(records, key) for key in mean_keys},
        "collision_type_counts": dict(sorted(collision_types.items())),
    }


def bootstrap_delta(a: Sequence[float], b: Sequence[float], seed: int) -> dict[str, Any]:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    diffs = right[mask] - left[mask]
    if not len(diffs):
        return {"paired_n": 0, "candidate_minus_baseline_mean": None, "bootstrap95": [None, None]}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(diffs), size=(10000, len(diffs)))
    samples = np.mean(diffs[indices], axis=1)
    return {
        "paired_n": int(len(diffs)),
        "candidate_minus_baseline_mean": float(np.mean(diffs)),
        "bootstrap95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
    }


def paired_comparison(
    baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]], scenario: str,
) -> dict[str, Any]:
    left = {int(row["seed"]): row for row in baseline}
    right = {int(row["seed"]): row for row in candidate}
    seeds = sorted(set(left).intersection(right))
    binary: dict[str, Any] = {}
    for metric in BINARY_METRICS[scenario]:
        pairs = [(bool(left[seed].get(metric, False)), bool(right[seed].get(metric, False))) for seed in seeds]
        binary[metric] = {
            "paired_n": len(pairs),
            "baseline_rate": float(np.mean([a for a, _ in pairs])),
            "candidate_rate": float(np.mean([b for _, b in pairs])),
            "candidate_minus_baseline": float(np.mean([int(b) - int(a) for a, b in pairs])),
            "candidate_only": int(sum((not a) and b for a, b in pairs)),
            "baseline_only": int(sum(a and (not b) for a, b in pairs)),
            "both": int(sum(a and b for a, b in pairs)),
            "neither": int(sum((not a) and (not b) for a, b in pairs)),
            "bootstrap95": bootstrap_delta([float(a) for a, _ in pairs], [float(b) for _, b in pairs], 20260903)["bootstrap95"],
        }
    continuous: dict[str, Any] = {}
    for metric in CONTINUOUS_METRICS[scenario]:
        pairs = [
            (finite_or_none(left[seed].get(metric)), finite_or_none(right[seed].get(metric)))
            for seed in seeds
        ]
        kept = [(a, b) for a, b in pairs if a is not None and b is not None]
        continuous[metric] = bootstrap_delta([a for a, _ in kept], [b for _, b in kept], 20260903)
    return {"scenario": scenario, "seeds": seeds, "binary": binary, "continuous": continuous}


def parse_model_spec(value: str) -> tuple[str, Path, str]:
    fields = value.split("=", 2)
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("--model must be LABEL=PATH=SHA256")
    label, path, expected = fields
    if not label or len(expected) != 64:
        raise argparse.ArgumentTypeError("invalid model label or SHA256")
    return label, Path(path), expected.lower()


def validate_stage2_contract(config: Mapping[str, Any]) -> dict[str, bool]:
    return validate_matched_contract(
        config, expected_pursuers=8, expected_evaders=2, expected_obstacles=2,
        action_contract="vxy9",
    )


def validate_matched_contract(
    config: Mapping[str, Any], *, expected_pursuers: int, expected_evaders: int,
    expected_obstacles: int, action_contract: str,
) -> dict[str, bool]:
    resolved_capture = scenario_config(dict(config), int(expected_evaders), task_name="voradj")
    env = dict(resolved_capture.get("env", {}) or {})
    action = dict(config.get("action", {}) or {})
    pursuer = dict(config.get("pursuer", {}) or {})
    iqn = dict(config.get("iqn", {}) or {})
    if action_contract not in {"aw9", "vxy9"}:
        raise ValueError(f"unsupported action contract: {action_contract}")
    action_ok = (
        str(action.get("mode", "")) == "discrete_desired_velocity_2d_body"
        and str(env.get("action_mode", "")) == "vxy9"
        if action_contract == "vxy9"
        else str(pursuer.get("action_mode", "unicycle")).lower() in {"unicycle", "unicycle_discrete"}
    )
    checks = {
        "stage_shape": int(env.get("num_pursuers", -1)) == int(expected_pursuers) and int(env.get("num_evaders", -1)) == int(expected_evaders) and int(env.get("num_obstacles", -1)) == int(expected_obstacles),
        "action_contract": bool(action_ok),
        "legacy_collision": str(env.get("collision_semantics", "legacy_end_step")) == "legacy_end_step",
        "iqn32": int(iqn.get("action_quantile_samples", -1)) == 32,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f"not the frozen matched formal contract: {failed}")
    return checks


def worker_main(args: argparse.Namespace) -> int:
    torch.set_num_threads(1)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    torch.manual_seed(int(args.seed))
    base = load_config(Path(args.config))
    model = CoCapIQN.load(str(Path(args.checkpoint)), device=str(args.device))
    model.eval()
    payload: dict[str, Any] = {scenario: [] for scenario in args.scenarios}
    started = time.time()
    for scenario in args.scenarios:
        for episode_index in range(int(args.start_index), int(args.end_index)):
            seed = int(args.seed) + episode_index
            payload[scenario].append(run_episode(
                model, base, scenario, seed, episode_index, str(args.device),
                action_contract=str(args.action_contract), num_evaders=int(args.expected_evaders),
                capture_horizon=int(args.capture_horizon), coverage_horizon=int(args.coverage_horizon),
                mix_horizon=int(args.mix_horizon),
            ))
    atomic_json(Path(args.worker_output), {
        "schema": SCHEMA, "model": args.worker_model, "start_index": args.start_index,
        "end_index": args.end_index, "records": payload, "wall_seconds": time.time() - started,
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", action="append", type=parse_model_spec)
    parser.add_argument("--output-root")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026083201)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers-per-model", type=int, default=10)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--baseline", default="stage2_600k")
    parser.add_argument("--action-contract", choices=("aw9", "vxy9"), default="vxy9")
    parser.add_argument("--expected-pursuers", type=int, default=8)
    parser.add_argument("--expected-evaders", type=int, default=2)
    parser.add_argument("--expected-obstacles", type=int, default=2)
    parser.add_argument("--capture-horizon", type=int, default=1000)
    parser.add_argument("--coverage-horizon", type=int, default=1500)
    parser.add_argument("--mix-horizon", type=int, default=2500)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", help=argparse.SUPPRESS)
    parser.add_argument("--worker-model", help=argparse.SUPPRESS)
    parser.add_argument("--start-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--end-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker_main(args)
    if not args.model or not args.output_root:
        parser.error("parent mode requires --model and --output-root")
    if int(args.episodes) <= 0 or int(args.workers_per_model) <= 0:
        parser.error("episodes and workers-per-model must be positive")

    config_path = Path(args.config).resolve()
    base = load_config(config_path)
    checks = validate_matched_contract(
        base, expected_pursuers=int(args.expected_pursuers), expected_evaders=int(args.expected_evaders),
        expected_obstacles=int(args.expected_obstacles), action_contract=str(args.action_contract),
    )
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_specs: list[tuple[str, Path, str]] = []
    for label, raw_path, expected in args.model:
        path = raw_path.resolve()
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"checkpoint SHA mismatch for {label}: expected={expected}, actual={actual}")
        model_specs.append((label, path, actual))
    labels = [label for label, _, _ in model_specs]
    if len(labels) != len(set(labels)) or args.baseline not in labels:
        raise ValueError("model labels must be unique and include --baseline")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        revision = "unknown"
    manifest = {
        "schema": SCHEMA, "status": "running", "created_at": now(), "git_revision": revision,
        "config": str(config_path), "config_file_sha256": sha256_file(config_path),
        "resolved_config_sha256": stable_hash(base), "matched_contract_checks": checks,
        "policy": {"epsilon": 0.0, "quantiles": "fixed_midpoint_32", "action_contract": str(args.action_contract)},
        "paired_seed_base": int(args.seed), "paired_seeds": [int(args.seed) + i for i in range(int(args.episodes))],
        "episodes_per_model_scenario": int(args.episodes), "scenarios": list(args.scenarios),
        "stage_shape": {"pursuers": int(args.expected_pursuers), "evaders": int(args.expected_evaders), "obstacles": int(args.expected_obstacles)},
        "caps": {"capture": int(args.capture_horizon), "coverage": int(args.coverage_horizon), "mix": int(args.mix_horizon)},
        "coverage_initialization": "global episode-index alternating map_random/inner_random_cluster",
        "device": str(args.device), "workers_per_model": int(args.workers_per_model),
        "models": {label: {"checkpoint": str(path), "sha256": sha} for label, path, sha in model_specs},
        "training_or_weight_updates": False,
    }
    atomic_json(output_root / "manifest.json", manifest)

    chunks: list[tuple[int, int]] = []
    for worker_index in range(min(int(args.workers_per_model), int(args.episodes))):
        start = worker_index * int(args.episodes) // min(int(args.workers_per_model), int(args.episodes))
        end = (worker_index + 1) * int(args.episodes) // min(int(args.workers_per_model), int(args.episodes))
        chunks.append((start, end))
    processes: list[tuple[str, Path, Any, Any]] = []
    for label, checkpoint, _ in model_specs:
        for worker_index, (start, end) in enumerate(chunks):
            worker_dir = output_root / "_workers" / label
            worker_dir.mkdir(parents=True, exist_ok=True)
            worker_output = worker_dir / f"worker_{worker_index:02d}.json"
            worker_log = worker_dir / f"worker_{worker_index:02d}.log"
            command = [
                sys.executable, str(Path(__file__).resolve()), "--worker", "--config", str(config_path),
                "--checkpoint", str(checkpoint), "--worker-model", label,
                "--start-index", str(start), "--end-index", str(end),
                "--worker-output", str(worker_output), "--seed", str(args.seed),
                "--device", str(args.device), "--scenarios", *list(args.scenarios),
                "--action-contract", str(args.action_contract),
                "--expected-evaders", str(args.expected_evaders),
                "--capture-horizon", str(args.capture_horizon),
                "--coverage-horizon", str(args.coverage_horizon),
                "--mix-horizon", str(args.mix_horizon),
            ]
            handle = worker_log.open("wb")
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            processes.append((label, worker_output, handle, process))
    failures: list[dict[str, Any]] = []
    for label, worker_output, handle, process in processes:
        returncode = process.wait()
        handle.close()
        if returncode != 0 or not worker_output.is_file():
            failures.append({"model": label, "worker_output": str(worker_output), "returncode": returncode})
    if failures:
        manifest["status"] = "failed"
        manifest["failures"] = failures
        atomic_json(output_root / "manifest.json", manifest)
        raise RuntimeError(f"formal workers failed: {failures}")

    model_records: dict[str, dict[str, list[dict[str, Any]]]] = {
        label: {scenario: [] for scenario in args.scenarios} for label in labels
    }
    for label, worker_output, _, _ in processes:
        payload = json.loads(worker_output.read_text(encoding="utf-8"))
        for scenario in args.scenarios:
            model_records[label][scenario].extend(payload["records"][scenario])
    summaries: dict[str, Any] = {}
    for label in labels:
        summaries[label] = {}
        for scenario in args.scenarios:
            rows = sorted(model_records[label][scenario], key=lambda row: int(row["seed"]))
            if len(rows) != int(args.episodes) or [int(row["seed"]) for row in rows] != manifest["paired_seeds"]:
                raise RuntimeError(f"incomplete/non-paired records for {label}/{scenario}")
            model_records[label][scenario] = rows
            summaries[label][scenario] = summarize(rows)
        atomic_json(output_root / label / "records.json", model_records[label])
        atomic_json(output_root / label / "summary.json", summaries[label])

    comparisons: dict[str, Any] = {}
    baseline = args.baseline
    for candidate in labels:
        if candidate == baseline:
            continue
        comparisons[f"{candidate}_vs_{baseline}"] = {
            scenario: paired_comparison(model_records[baseline][scenario], model_records[candidate][scenario], scenario)
            for scenario in args.scenarios
        }
    atomic_json(output_root / "summaries.json", summaries)
    atomic_json(output_root / "paired_comparisons.json", comparisons)
    manifest["status"] = "complete"
    manifest["completed_at"] = now()
    manifest["record_counts"] = {
        label: {scenario: len(model_records[label][scenario]) for scenario in args.scenarios}
        for label in labels
    }
    atomic_json(output_root / "manifest.json", manifest)
    (output_root / "FORMAL_DONE").write_text(now() + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output_root": str(output_root), "summaries": summaries}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
