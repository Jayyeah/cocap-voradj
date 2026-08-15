#!/usr/bin/env python3
"""Collect CF3 near-success trajectory tails for capture consolidation.

This is a read-only checkpoint diagnostic.  It runs paired-seed rollouts under
one explicitly selected policy temperature and retains the last ``N`` decision
steps for three mutually useful outcome classes:

* a real (non-stationary) formal capture;
* a failure which reached 3+ agents in the capture ring and then collided;
* a non-capture failure with a sustained 2+ ring hold.

Each saved transition contains enough physical state to answer where the
coordination broke: ring membership and angles, angular gaps, pair/obstacle/
boundary clearance, radial and tangential relative velocity, and the applied
``(a, omega)`` command including saturation and omega sign-flip flags.  The
tool never writes into, or resumes, the training bundle.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.continuous.formal_config import scene_config
from cocap_voradj.training.trainer import load_config, set_global_config
from tools.probe_cf3_policy_temperature import (
    TemperatureMode,
    apply_collision_semantics_override,
    epsilon_stream,
    parse_temperature_modes,
    sample_temperature_actions,
)
from tools.run_continuous_ctde_training import (
    _evader_actions_for_env,
    _make_trainer,
    _pad_local_obs_tree,
)
from tools.run_masac_rollout_gifs import verify_config_contract


CATEGORY_NORMAL_CAPTURE = "normal_capture"
CATEGORY_THREE_PLUS_COLLISION = "three_plus_then_collision"
CATEGORY_LONG_TWO_PLUS = "long_two_plus_failure"
CATEGORIES = (
    CATEGORY_NORMAL_CAPTURE,
    CATEGORY_THREE_PLUS_COLLISION,
    CATEGORY_LONG_TWO_PLUS,
)


def _position(robot: Any) -> np.ndarray:
    return np.asarray([float(robot.x), float(robot.y)], dtype=float)


def _velocity(robot: Any) -> np.ndarray:
    value = getattr(robot, "velocity", None)
    if value is None:
        return np.zeros(2, dtype=float)
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size < 2:
        return np.zeros(2, dtype=float)
    return result[:2]


def _finite(value: float | np.floating) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _circular_separation(a: float, b: float) -> float:
    return float(abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi))


def angular_geometry(angles: Sequence[float]) -> Dict[str, Any]:
    """Return rotation-invariant angular gap diagnostics in radians."""
    ordered = sorted(float(value) % (2.0 * math.pi) for value in angles)
    if not ordered:
        return {
            "angles_rad": [],
            "adjacent_gaps_rad": [],
            "pairwise_separations_rad": [],
            "largest_angular_gap_rad": None,
            "smallest_adjacent_gap_rad": None,
            "minimum_pairwise_separation_rad": None,
        }
    if len(ordered) == 1:
        gaps = [2.0 * math.pi]
        pairwise: list[float] = []
    else:
        gaps = [
            float((ordered[(index + 1) % len(ordered)] - ordered[index]) % (2.0 * math.pi))
            for index in range(len(ordered))
        ]
        pairwise = [
            _circular_separation(ordered[left], ordered[right])
            for left in range(len(ordered))
            for right in range(left + 1, len(ordered))
        ]
    return {
        "angles_rad": ordered,
        "adjacent_gaps_rad": gaps,
        "pairwise_separations_rad": pairwise,
        "largest_angular_gap_rad": float(max(gaps)),
        "smallest_adjacent_gap_rad": float(min(gaps)),
        "minimum_pairwise_separation_rad": float(min(pairwise)) if pairwise else None,
    }


def relative_velocity_components(
    pursuer_position: Sequence[float],
    pursuer_velocity: Sequence[float],
    enemy_position: Sequence[float],
    enemy_velocity: Sequence[float],
) -> Dict[str, float]:
    """Resolve relative velocity in the enemy-centred radial frame.

    ``radial_velocity_outward`` is positive when the pursuer moves away from
    the enemy. ``closing_speed`` is its negation. Tangential velocity is
    positive counter-clockwise around the enemy.
    """
    ppos = np.asarray(pursuer_position, dtype=float)
    pvel = np.asarray(pursuer_velocity, dtype=float)
    epos = np.asarray(enemy_position, dtype=float)
    evel = np.asarray(enemy_velocity, dtype=float)
    delta = ppos - epos
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        radial = tangential = 0.0
    else:
        outward = delta / distance
        relative = pvel - evel
        radial = float(np.dot(relative, outward))
        tangential = float(outward[0] * relative[1] - outward[1] * relative[0])
    return {
        "distance_to_enemy": distance,
        "radial_velocity_outward": radial,
        "closing_speed": -radial,
        "tangential_velocity_ccw": tangential,
    }


def clearance_geometry(
    pursuers: Sequence[Any],
    obstacles: Sequence[Any],
    *,
    width: float,
    height: float,
) -> Dict[str, Any]:
    """Compute explicit surface clearances without filtering dead robots."""
    pair_rows: list[Dict[str, Any]] = []
    obstacle_rows: list[Dict[str, Any]] = []
    boundary_rows: list[Dict[str, Any]] = []
    for left in range(len(pursuers)):
        for right in range(left + 1, len(pursuers)):
            p = pursuers[left]
            q = pursuers[right]
            clearance = float(np.linalg.norm(_position(p) - _position(q)) - float(p.r) - float(q.r))
            pair_rows.append({
                "pursuer_i": int(p.id),
                "pursuer_j": int(q.id),
                "surface_clearance": clearance,
            })
    for pursuer in pursuers:
        ppos = _position(pursuer)
        for obstacle_index, obstacle in enumerate(obstacles):
            clearance = float(
                np.linalg.norm(ppos - _position(obstacle))
                - float(pursuer.r) - float(obstacle.r)
            )
            obstacle_rows.append({
                "pursuer_id": int(pursuer.id),
                "obstacle_id": int(getattr(obstacle, "id", obstacle_index)),
                "surface_clearance": clearance,
            })
        center_clearance = float(min(
            float(pursuer.x),
            float(pursuer.y),
            float(width) - float(pursuer.x),
            float(height) - float(pursuer.y),
        ))
        boundary_rows.append({
            "pursuer_id": int(pursuer.id),
            # This matches the simulator's hard-boundary test, which is on the
            # robot centre rather than on the collision circle.
            "center_clearance": center_clearance,
        })
    return {
        "agent_agent": pair_rows,
        "agent_obstacle": obstacle_rows,
        "boundary": boundary_rows,
        "min_agent_agent_surface_clearance": _finite(
            min((row["surface_clearance"] for row in pair_rows), default=math.inf)
        ),
        "min_obstacle_surface_clearance": _finite(
            min((row["surface_clearance"] for row in obstacle_rows), default=math.inf)
        ),
        "min_boundary_center_clearance": _finite(
            min((row["center_clearance"] for row in boundary_rows), default=math.inf)
        ),
    }


def _active_enemy(env: Any) -> Any | None:
    active = [enemy for enemy in env.evaders if not bool(enemy.deactivated)]
    return active[0] if active else (env.evaders[0] if env.evaders else None)


def physical_snapshot(
    env: Any,
    *,
    pursuer_ids: Iterable[int] | None = None,
) -> Dict[str, Any]:
    """Snapshot enemy-centred capture geometry at the current env state."""
    selected_ids = None if pursuer_ids is None else {int(value) for value in pursuer_ids}
    pursuers = [
        pursuer for pursuer in env.pursuers
        if (
            (selected_ids is None and not bool(pursuer.deactivated))
            or (selected_ids is not None and int(pursuer.id) in selected_ids)
        )
    ]
    enemy = _active_enemy(env)
    if enemy is None:
        return {
            "enemy": None,
            "pursuers": [],
            "ring_count": 0,
            "angular": angular_geometry([]),
            "clearance": clearance_geometry(
                pursuers, env.obstacles, width=float(env.width), height=float(env.height),
            ),
        }
    epos = _position(enemy)
    evel = _velocity(enemy)
    rows: list[Dict[str, Any]] = []
    ring_angles: list[float] = []
    all_angles: list[float] = []
    for pursuer in pursuers:
        ppos = _position(pursuer)
        angle = float(math.atan2(ppos[1] - epos[1], ppos[0] - epos[0]) % (2.0 * math.pi))
        components = relative_velocity_components(ppos, _velocity(pursuer), epos, evel)
        in_ring = 8.0 <= components["distance_to_enemy"] < 10.5
        all_angles.append(angle)
        if in_ring:
            ring_angles.append(angle)
        rows.append({
            "id": int(pursuer.id),
            "active": not bool(pursuer.deactivated),
            "collision": bool(getattr(pursuer, "collision", False)),
            "collision_types": sorted(str(value) for value in (getattr(pursuer, "collision_types", set()) or set())),
            "position": ppos.tolist(),
            "velocity": _velocity(pursuer).tolist(),
            "theta": float(pursuer.theta),
            "speed": float(pursuer.speed),
            "enemy_centric_angle_rad": angle,
            "in_ring_8_10_5": bool(in_ring),
            **components,
        })
    ring_rows = [row for row in rows if row["in_ring_8_10_5"]]
    ring_pairs = [
        {
            "pursuer_i": int(ring_rows[left]["id"]),
            "pursuer_j": int(ring_rows[right]["id"]),
            "separation_rad": _circular_separation(
                float(ring_rows[left]["enemy_centric_angle_rad"]),
                float(ring_rows[right]["enemy_centric_angle_rad"]),
            ),
        }
        for left in range(len(ring_rows))
        for right in range(left + 1, len(ring_rows))
    ]
    return {
        "enemy": {
            "id": int(enemy.id),
            "active": not bool(enemy.deactivated),
            "position": epos.tolist(),
            "velocity": evel.tolist(),
            "speed": float(enemy.speed),
        },
        "pursuers": rows,
        "ring_count": int(sum(row["in_ring_8_10_5"] for row in rows)),
        "angular": angular_geometry(ring_angles),
        "ring_pairwise_angular_separations": ring_pairs,
        "all_active_angular": angular_geometry(all_angles),
        "clearance": clearance_geometry(
            pursuers, env.obstacles, width=float(env.width), height=float(env.height),
        ),
    }


def action_rows(
    env: Any,
    actions: np.ndarray,
    active_ids: Sequence[int],
    previous_omega: MutableMapping[int, float],
    *,
    saturation_fraction: float = 0.98,
    omega_flip_threshold: float = 0.02,
) -> list[Dict[str, Any]]:
    a_max = float(env.action_adapter.a_max)
    w_max = float(env.action_adapter.w_max)
    active = {int(value) for value in active_ids}
    rows: list[Dict[str, Any]] = []
    for index, pursuer in enumerate(env.pursuers):
        if int(pursuer.id) not in active or index >= len(actions):
            continue
        a = float(actions[index][0])
        omega = float(actions[index][1])
        previous = previous_omega.get(int(pursuer.id))
        sign_flip = bool(
            previous is not None
            and abs(previous) >= float(omega_flip_threshold)
            and abs(omega) >= float(omega_flip_threshold)
            and previous * omega < 0.0
        )
        previous_omega[int(pursuer.id)] = omega
        rows.append({
            "pursuer_id": int(pursuer.id),
            "a": a,
            "omega": omega,
            "a_fraction_of_limit": float(abs(a) / max(a_max, 1e-12)),
            "omega_fraction_of_limit": float(abs(omega) / max(w_max, 1e-12)),
            "a_saturated": bool(abs(a) >= float(saturation_fraction) * a_max),
            "omega_saturated": bool(abs(omega) >= float(saturation_fraction) * w_max),
            "omega_sign_flip": sign_flip,
        })
    return rows


def max_hold(counts: Sequence[int], minimum: int) -> int:
    current = best = 0
    for value in counts:
        current = current + 1 if int(value) >= int(minimum) else 0
        best = max(best, current)
    return int(best)


def classify_episode(
    frames: Sequence[Mapping[str, Any]],
    *,
    normal_capture: bool,
    stationary_capture: bool,
    collision: bool,
    long_two_plus_steps: int,
) -> str | None:
    """Choose one exclusive consolidation class, in priority order."""
    if normal_capture:
        return CATEGORY_NORMAL_CAPTURE
    reached_three = False
    three_then_collision = False
    for frame in frames:
        reached_three = reached_three or max(
            int(frame.get("ring_count_pre", 0)),
            int(frame.get("ring_count_post_alive", 0)),
            int(frame.get("ring_count_post_was_active", 0)),
        ) >= 3
        three_then_collision = three_then_collision or (
            reached_three and bool(frame.get("collision_now", False))
        )
    # A plain aggregate collision flag is retained for small synthetic callers
    # which do not include per-step collision events. Runtime traces always do.
    if collision and not any("collision_now" in frame for frame in frames):
        three_then_collision = reached_three
    if collision and not stationary_capture and three_then_collision:
        return CATEGORY_THREE_PLUS_COLLISION
    post_counts = [int(frame.get("ring_count_post_alive", 0)) for frame in frames]
    if not normal_capture and not stationary_capture and max_hold(post_counts, 2) >= int(long_two_plus_steps):
        return CATEGORY_LONG_TWO_PLUS
    return None


def _numeric(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def _distribution(values: Iterable[Any]) -> Dict[str, float | int | None]:
    clean = _numeric(values)
    if not clean:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    array = np.asarray(clean, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_trace(frames: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    post_counts = [int(frame.get("ring_count_post_alive", 0)) for frame in frames]
    agents = [agent for frame in frames for agent in frame.get("post", {}).get("pursuers", [])]
    actions = [action for frame in frames for action in frame.get("actions", [])]
    return {
        "steps_saved": int(len(frames)),
        "first_episode_step": int(frames[0]["episode_step"]) if frames else None,
        "last_episode_step": int(frames[-1]["episode_step"]) if frames else None,
        "max_num_in_ring": int(max(post_counts, default=0)),
        "max_2plus_ring_hold_steps": max_hold(post_counts, 2),
        "max_3plus_ring_hold_steps": max_hold(post_counts, 3),
        "largest_angular_gap_when_2plus": _distribution(
            frame.get("post", {}).get("angular", {}).get("largest_angular_gap_rad")
            for frame in frames if int(frame.get("ring_count_post_alive", 0)) >= 2
        ),
        "minimum_pairwise_separation_when_2plus": _distribution(
            frame.get("post", {}).get("angular", {}).get("minimum_pairwise_separation_rad")
            for frame in frames if int(frame.get("ring_count_post_alive", 0)) >= 2
        ),
        "agent_agent_surface_clearance": _distribution(
            frame.get("post", {}).get("clearance", {}).get("min_agent_agent_surface_clearance")
            for frame in frames
        ),
        "obstacle_surface_clearance": _distribution(
            frame.get("post", {}).get("clearance", {}).get("min_obstacle_surface_clearance")
            for frame in frames
        ),
        "boundary_center_clearance": _distribution(
            frame.get("post", {}).get("clearance", {}).get("min_boundary_center_clearance")
            for frame in frames
        ),
        "radial_velocity_outward": _distribution(agent.get("radial_velocity_outward") for agent in agents),
        "tangential_velocity_ccw": _distribution(agent.get("tangential_velocity_ccw") for agent in agents),
        "a": _distribution(action.get("a") for action in actions),
        "omega": _distribution(action.get("omega") for action in actions),
        "a_saturation_rate": float(np.mean([bool(action.get("a_saturated")) for action in actions])) if actions else 0.0,
        "omega_saturation_rate": float(np.mean([bool(action.get("omega_saturated")) for action in actions])) if actions else 0.0,
        "omega_sign_flip_rate": float(np.mean([bool(action.get("omega_sign_flip")) for action in actions])) if actions else 0.0,
        "collision_event_count": int(sum(len(frame.get("collision_events", [])) for frame in frames)),
    }


def summarize_categories(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    scalar_paths = (
        "max_num_in_ring",
        "max_2plus_ring_hold_steps",
        "max_3plus_ring_hold_steps",
        "a_saturation_rate",
        "omega_saturation_rate",
        "omega_sign_flip_rate",
    )
    for category in CATEGORIES:
        selected = [case for case in cases if case.get("category") == category]
        result[category] = {
            "cases": int(len(selected)),
            **{
                key: _distribution(case.get("trace_summary", {}).get(key) for case in selected)
                for key in scalar_paths
            },
            "min_agent_agent_surface_clearance": _distribution(
                case.get("trace_summary", {}).get("agent_agent_surface_clearance", {}).get("min")
                for case in selected
            ),
            "largest_gap_mean_when_2plus": _distribution(
                case.get("trace_summary", {}).get("largest_angular_gap_when_2plus", {}).get("mean")
                for case in selected
            ),
        }
    return result


def run_episode(
    trainer: Any,
    root_config: Dict[str, Any],
    *,
    mode: TemperatureMode,
    seed: int,
    noise_seed: int,
    max_steps: int,
    tail_steps: int,
    long_two_plus_steps: int,
) -> Dict[str, Any]:
    config = scene_config(root_config, "capture")
    set_global_config(config)
    env = VorAdjEnv(config, seed=int(seed))
    observations = list(env.reset())
    apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    max_agents = int(root_config["central_critic"]["max_agents"])
    actor_max_pursuers = int(root_config["actor"]["max_pursuers"])
    self_dim = int(root_config["actor"].get("self_feature_dim", 9))
    noises = epsilon_stream(noise_seed, max_agents)
    tail: deque[Dict[str, Any]] = deque(maxlen=int(tail_steps))
    previous_omega: Dict[int, float] = {}
    capture_types: list[str] = []
    collision_ever = False
    full_post_ring_counts: list[int] = []
    full_reached_three = False
    full_three_then_collision = False

    horizon = min(int(max_steps), int(config["env"]["episode_max_length"]))
    for episode_step in range(1, horizon + 1):
        active_ids = [int(pursuer.id) for pursuer in env.pursuers if not pursuer.deactivated]
        pre = physical_snapshot(env, pursuer_ids=active_ids)
        padded = _pad_local_obs_tree(observations, max_agents, actor_max_pursuers, self_dim)
        sampled, rejected = sample_temperature_actions(
            trainer, padded, len(env.pursuers), env.action_adapter,
            mode.temperature, next(noises),
        )
        actions = [
            None if observations[index] is None else sampled[index]
            for index in range(len(env.pursuers))
        ]
        action_diagnostics = action_rows(env, sampled, active_ids, previous_omega)
        outcome = env.step(actions, _evader_actions_for_env(env, apf_agents))
        events = [dict(event) for event in getattr(env, "last_capture_events", [])]
        capture_types.extend(str(event.get("capture_type", "unknown")) for event in events)
        collision_events = [dict(event) for event in getattr(env, "last_collision_events", [])]
        collision_now = bool(collision_events) or any(
            bool(getattr(pursuer, "collision", False)) and int(pursuer.id) in active_ids
            for pursuer in env.pursuers
        )
        collision_ever = collision_ever or collision_now
        post_alive = physical_snapshot(env)
        post_was_active = physical_snapshot(env, pursuer_ids=active_ids)
        full_post_ring_counts.append(int(post_alive["ring_count"]))
        full_reached_three = full_reached_three or max(
            int(pre["ring_count"]),
            int(post_alive["ring_count"]),
            int(post_was_active["ring_count"]),
        ) >= 3
        full_three_then_collision = full_three_then_collision or (
            full_reached_three and collision_now
        )
        tail.append({
            "episode_step": int(episode_step),
            "pre": pre,
            "post": post_alive,
            "post_was_active": post_was_active,
            "ring_count_pre": int(pre["ring_count"]),
            "ring_count_post_alive": int(post_alive["ring_count"]),
            "ring_count_post_was_active": int(post_was_active["ring_count"]),
            "actions": action_diagnostics,
            "action_rejected_rate": float(rejected),
            "capture_events": events,
            "collision_events": collision_events,
            "collision_now": bool(collision_now),
            "dones": [bool(value) for value in outcome.dones],
        })
        observations = list(outcome.observations)
        if all(outcome.dones):
            break

    record = env.episode_record(task="capture")
    normal_capture = any(value != "stationary" for value in capture_types)
    stationary_capture = any(value == "stationary" for value in capture_types)
    frames = list(tail)
    # Classification must consider the full episode rather than only the saved
    # tail. Inject the full-series hold/reached-three result explicitly.
    full_two_hold = max_hold(full_post_ring_counts, 2)
    if normal_capture:
        category = CATEGORY_NORMAL_CAPTURE
    elif collision_ever and not stationary_capture and full_three_then_collision:
        category = CATEGORY_THREE_PLUS_COLLISION
    elif not stationary_capture and full_two_hold >= int(long_two_plus_steps):
        category = CATEGORY_LONG_TWO_PLUS
    else:
        category = None
    return {
        "seed": int(seed),
        "noise_seed": int(noise_seed),
        "mode": mode.name,
        "temperature": float(mode.temperature),
        "length": int(record.get("episode_length", len(full_post_ring_counts))),
        "normal_capture": bool(normal_capture),
        "stationary_capture": bool(stationary_capture),
        "collision": bool(collision_ever or record.get("collision_event", False)),
        "collision_type_counts": dict(record.get("collision_type_counts", {}) or {}),
        "max_num_in_ring_full_episode": int(max(full_post_ring_counts, default=0)),
        "reached_three_full_episode": bool(full_reached_three),
        "three_plus_then_collision_full_episode": bool(full_three_then_collision),
        "max_2plus_ring_hold_steps_full_episode": int(full_two_hold),
        "max_3plus_ring_hold_steps_full_episode": max_hold(full_post_ring_counts, 3),
        "category": category,
        "capture_types": capture_types,
        "trace_summary": summarize_trace(frames),
        "frames": frames,
    }


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    config = load_config(str(config_path))
    capture_config = scene_config(config, "capture")
    horizon = min(int(args.max_steps), int(capture_config["env"]["episode_max_length"]))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = dict(saved.get("contract", {}) or {})
    verification = verify_config_contract(
        config, contract, [{"scene": "capture", "max_steps": horizon}],
    )
    evaluation_config = apply_collision_semantics_override(config, args.collision_semantics)
    trainer = _make_trainer(config, str(args.device))
    trainer.load_checkpoint(checkpoint, contract)
    trainer.actor.eval()
    modes = parse_temperature_modes([float(args.temperature)])
    mode = modes[0]

    cases: list[Dict[str, Any]] = []
    episode_summaries: list[Dict[str, Any]] = []
    category_counts = {name: 0 for name in CATEGORIES}
    for episode_index in range(int(args.episodes)):
        episode = run_episode(
            trainer,
            evaluation_config,
            mode=mode,
            seed=int(args.seed) + episode_index,
            noise_seed=int(args.noise_seed) + episode_index,
            max_steps=horizon,
            tail_steps=int(args.tail_steps),
            long_two_plus_steps=int(args.long_two_plus_steps),
        )
        summary = {key: value for key, value in episode.items() if key != "frames"}
        episode_summaries.append(summary)
        category = episode.get("category")
        if category in category_counts and category_counts[category] < int(args.max_cases_per_category):
            cases.append(episode)
            category_counts[category] += 1
        if all(value >= int(args.max_cases_per_category) for value in category_counts.values()):
            break

    return {
        "schema_version": 1,
        "kind": "cf3_success_consolidation_audit",
        "created_at": now(),
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "device": str(args.device),
        "episodes_attempted": int(len(episode_summaries)),
        "tail_steps": int(args.tail_steps),
        "long_two_plus_hold_threshold_steps": int(args.long_two_plus_steps),
        "max_cases_per_category": int(args.max_cases_per_category),
        "mode": {"name": mode.name, "temperature": mode.temperature},
        "collision_semantics_override": args.collision_semantics,
        "sign_conventions": {
            "radial_velocity_outward": "positive moves away from enemy; closing_speed is its negation",
            "tangential_velocity_ccw": "positive is counter-clockwise around enemy",
            "angular_units": "radians",
            "boundary_clearance": "robot-centre clearance, matching hard-boundary predicate",
        },
        "config_verification": verification,
        "category_comparison": summarize_categories(cases),
        "episode_summaries": episode_summaries,
        "cases": cases,
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--tail-steps", type=int, default=50)
    parser.add_argument("--long-two-plus-steps", type=int, default=20)
    parser.add_argument("--max-cases-per-category", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026081601)
    parser.add_argument("--noise-seed", type=int, default=2026081651)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--collision-semantics",
        choices=("legacy_end_step", "synchronized_swept_v1"),
        default=None,
        help="Optional simulator-only intervention, applied after checkpoint hash verification.",
    )
    args = parser.parse_args()
    positive = (
        "episodes", "max_steps", "tail_steps", "long_two_plus_steps",
        "max_cases_per_category",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    payload = evaluate(args)
    _write_payload(payload, Path(args.output))
    print(json.dumps(payload["category_comparison"], indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
