"""Legacy-IQN-derived curriculum snapshot schema and geometry/full restore."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.trainer import set_global_config


SNAPSHOT_SCHEMA_VERSION = 1


def _vector(robot: Any) -> List[float]:
    return [float(robot.x), float(robot.y)]


def _velocity(robot: Any) -> List[float]:
    value = np.asarray(getattr(robot, "velocity", (0.0, 0.0)), dtype=float).reshape(-1)
    return [float(value[0]) if len(value) > 0 else 0.0, float(value[1]) if len(value) > 1 else 0.0]


def build_snapshot(
    env: VorAdjEnv,
    *,
    scene: str,
    phase: str,
    step: int,
    source: Dict[str, Any],
    task_labels: Optional[List[str]] = None,
    post_capture_elapsed: int = 0,
) -> Dict[str, Any]:
    labels = list(task_labels) if task_labels is not None else list(getattr(env, "last_task_labels", []))
    pursuer_count = len(env.pursuers)
    evader_count = len(env.evaders)
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source": dict(source),
        "scene": str(scene),
        "phase": str(phase),
        "step": int(step),
        "pursuers": {
            "position": [_vector(p) for p in env.pursuers],
            "world_velocity": [_velocity(p) for p in env.pursuers],
            "speed": [float(p.speed) for p in env.pursuers],
            "theta": [float(p.theta) for p in env.pursuers],
            "active_mask": [bool(not p.deactivated) for p in env.pursuers],
            "collision_mask": [bool(getattr(p, "collision", False)) for p in env.pursuers],
        },
        "evaders": {
            "position": [_vector(e) for e in env.evaders],
            "world_velocity": [_velocity(e) for e in env.evaders],
            "speed": [float(e.speed) for e in env.evaders],
            "theta": [float(e.theta) for e in env.evaders],
            "active_mask": [bool(not e.deactivated) for e in env.evaders],
            "captured": [bool(e.deactivated) for e in env.evaders],
        },
        "obstacles": {
            "position": [[float(o.x), float(o.y)] for o in env.obstacles],
            "radius": [float(o.r) for o in env.obstacles],
        },
        "task_state": {
            "task_labels": list(labels),
            "pursuing_mask": [bool(label == "capture") for label in labels],
            "phase_counter": int(step),
            "post_capture_elapsed": int(post_capture_elapsed),
            "captured_target_ids": [int(i) for i, e in enumerate(env.evaders) if e.deactivated],
        },
        "entity_counts": {"pursuers": pursuer_count, "evaders": evader_count, "obstacles": len(env.obstacles)},
    }


def restore_snapshot(
    snapshot: Dict[str, Any],
    config: Dict[str, Any],
    seed: int,
    *,
    mode: str = "geometry_reset",
) -> Tuple[VorAdjEnv, Dict[str, Any]]:
    if int(snapshot.get("schema_version", 0)) != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("snapshot schema version mismatch")
    if mode not in {"geometry_reset", "full_state"}:
        raise ValueError("snapshot restore mode must be geometry_reset or full_state")
    cfg = dict(config)
    set_global_config(cfg)
    env = VorAdjEnv(cfg, seed=seed)
    pursuer_positions = snapshot["pursuers"]["position"]
    pursuer_active = snapshot["pursuers"]["active_mask"]
    if len(pursuer_positions) != int(env.env_cfg.get("num_pursuers", len(pursuer_positions))):
        raise ValueError("snapshot pursuer count does not match env")
    env.reset()
    for i, pursuer in enumerate(env.pursuers):
        x, y = pursuer_positions[i]
        pursuer.start = np.asarray([float(x), float(y)], dtype=float)
        pursuer.x, pursuer.y = float(x), float(y)
        pursuer.deactivated = not bool(pursuer_active[i])
        pursuer.collision = False
        pursuer.boundary_collision = False
        if mode == "full_state":
            vx, vy = snapshot["pursuers"]["world_velocity"][i]
            pursuer.velocity = np.asarray([float(vx), float(vy)], dtype=float)
            pursuer.speed = float(snapshot["pursuers"]["speed"][i])
            pursuer.theta = float(snapshot["pursuers"]["theta"][i])
        else:
            pursuer.velocity = np.zeros(2, dtype=float)
            pursuer.speed = 0.0
            pursuer.theta = 0.0
    for i, evader in enumerate(env.evaders):
        if i >= len(snapshot["evaders"]["position"]):
            continue
        x, y = snapshot["evaders"]["position"][i]
        evader.x, evader.y = float(x), float(y)
        evader.deactivated = not bool(snapshot["evaders"]["active_mask"][i])
        if mode == "full_state":
            vx, vy = snapshot["evaders"]["world_velocity"][i]
            evader.velocity = np.asarray([float(vx), float(vy)], dtype=float)
            evader.speed = float(snapshot["evaders"]["speed"][i])
            evader.theta = float(snapshot["evaders"]["theta"][i])
        else:
            evader.velocity = np.zeros(2, dtype=float)
            evader.speed = 0.0
            evader.theta = 0.0
    for i, obstacle in enumerate(env.obstacles):
        if i >= len(snapshot["obstacles"]["position"]):
            continue
        x, y = snapshot["obstacles"]["position"][i]
        obstacle.x, obstacle.y = float(x), float(y)
        obstacle.r = float(snapshot["obstacles"]["radius"][i])
    phase = str(snapshot.get("phase", "pre_capture"))
    if phase in {"post_capture", "pure_recovery"}:
        env.post_capture_started = True
        env.post_capture_step = int(snapshot.get("task_state", {}).get("post_capture_elapsed", 0))
        env.post_capture_grace_remaining = 0
        if phase == "post_capture":
            env.capture_snapshot = {
                "step": int(snapshot.get("step", 0)),
                "positions": [[float(x), float(y)] for x, y in pursuer_positions],
                "active_mask": [bool(v) for v in pursuer_active],
            }
    env._invalidate_voronoi_cache()
    env.last_task_labels = env._task_labels_from_map(env._capture_voronoi_map(), update_effective=True)
    env.last_reward_terms = {}
    diagnostics = {
        "restore_mode": mode,
        "scene": str(snapshot.get("scene", "")),
        "phase": phase,
        "pursuer_positions": [[float(p.x), float(p.y)] for p in env.pursuers],
        "evader_positions": [[float(e.x), float(e.y)] for e in env.evaders],
        "obstacles": [[float(o.x), float(o.y), float(o.r)] for o in env.obstacles],
        "task_labels": list(env.last_task_labels),
    }
    return env, diagnostics
