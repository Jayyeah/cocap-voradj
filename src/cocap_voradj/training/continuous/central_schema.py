"""Training-only global entity schema for the continuous CTDE path.

The schema intentionally contains geometry, velocity and active masks only.
It does not expose phase, origin, event labels or any other oracle metadata.
Each focal row is expressed in that pursuer's body frame, while the critic
receives all globally available entities rather than the actor's local crop.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _position(robot: Any) -> np.ndarray:
    return np.asarray([float(robot.x), float(robot.y)], dtype=np.float32)


def _velocity(robot: Any) -> np.ndarray:
    value = getattr(robot, "velocity", (0.0, 0.0))
    return np.asarray(value, dtype=np.float32).reshape(2)


def build_central_global_obs(
    env: Any,
    *,
    max_agents: int = 12,
    max_evaders: int = 8,
    max_obstacles: int = 5,
    self_feature_dim: int = 9,
) -> Dict[str, np.ndarray]:
    """Build one padded central observation from an environment snapshot.

The global token layout is fixed in the world frame: pursuers are
``[x, y, vx, vy, sin(yaw), cos(yaw), role]``, evaders are
``[x, y, vx, vy, sin(yaw), cos(yaw), 1]``, and obstacles are
``[x, y, radius, 0, 0]``.  Positions, radii and velocities are normalized by
the environment scale.  Focal self features retain the local actor contract;
the central critic therefore has a consistent global entity set plus a query
for each focal pursuer.
    """
    pursuers = list(getattr(env, "pursuers", []))
    evaders = list(getattr(env, "evaders", []))
    obstacles = list(getattr(env, "obstacles", []))
    if len(pursuers) > int(max_agents) or len(evaders) > int(max_evaders) or len(obstacles) > int(max_obstacles):
        raise ValueError("environment entity count exceeds central schema padding")
    active = np.asarray([not bool(getattr(item, "deactivated", False)) for item in pursuers], dtype=bool)
    active_mask = np.zeros(int(max_agents), dtype=bool)
    active_mask[: len(pursuers)] = active
    self_features = np.zeros((int(max_agents), int(self_feature_dim)), dtype=np.float32)
    pursuer_features = np.zeros((int(max_agents), int(max_agents), 7), dtype=np.float32)
    evader_features = np.zeros((int(max_evaders), 7), dtype=np.float32)
    obstacle_features = np.zeros((int(max_obstacles), 5), dtype=np.float32)

    width = max(float(getattr(env, "width", 1.0)), 1e-6)
    height = max(float(getattr(env, "height", 1.0)), 1e-6)
    position_scale = max(width, height)
    velocity_scale = max(
        [float(getattr(item, "max_speed", 0.0) or 0.0) for item in [*pursuers, *evaders]] + [1.0]
    )
    for i, focal in enumerate(pursuers):
        if not active[i]:
            continue
        local = env._pack_agent_obs(i)
        if local is not None:
            self_value = np.asarray(local["self"], dtype=np.float32).reshape(-1)
            self_features[i, : min(len(self_value), self_feature_dim)] = self_value[:self_feature_dim]
        for j, other in enumerate(pursuers):
            if not active[j]:
                continue
            role = float(getattr(other, "is_pursuing", False)) if j != i else float(getattr(focal, "is_pursuing", False))
            yaw = float(getattr(other, "theta", 0.0) or 0.0)
            position = _position(other)
            velocity = _velocity(other)
            pursuer_features[i, j] = [
                float(position[0] / width),
                float(position[1] / height),
                float(velocity[0] / velocity_scale),
                float(velocity[1] / velocity_scale),
                float(np.sin(yaw)),
                float(np.cos(yaw)),
                role,
            ]

    evader_mask = np.zeros(int(max_evaders), dtype=bool)
    for j, evader in enumerate(evaders):
        if bool(getattr(evader, "deactivated", False)):
            continue
        evader_mask[j] = True
        position = _position(evader)
        velocity = _velocity(evader)
        yaw = float(getattr(evader, "theta", 0.0) or 0.0)
        evader_features[j] = [
            float(position[0] / width),
            float(position[1] / height),
            float(velocity[0] / velocity_scale),
            float(velocity[1] / velocity_scale),
            float(np.sin(yaw)),
            float(np.cos(yaw)),
            1.0,
        ]

    obstacle_mask = np.zeros(int(max_obstacles), dtype=bool)
    for j, obstacle in enumerate(obstacles):
        obstacle_mask[j] = True
        position = _position(obstacle)
        obstacle_features[j] = [
            float(position[0] / width),
            float(position[1] / height),
            float(getattr(obstacle, "r", 0.0) / position_scale),
            0.0,
            0.0,
        ]

    # Current critic contract expands these shared evader/obstacle tokens for
    # each focal row.  Keep the explicit masks in the replay schema.
    pursuer_mask = np.broadcast_to(active_mask[None, :], (int(max_agents), int(max_agents))).copy()
    pursuer_mask &= active_mask[:, None]
    return {
        "self": self_features,
        "pursuers": pursuer_features,
        "evaders": evader_features,
        "obstacles": obstacle_features,
        "pursuer_mask": pursuer_mask,
        "evader_mask": evader_mask,
        "obstacle_mask": obstacle_mask,
        "active_mask": active_mask,
    }
