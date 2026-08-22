"""Analytic boundary/circular-obstacle lidar used by Eq. 24."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import LidarConfig, WorkspaceConfig


FloatArray = NDArray[np.float64]


def ray_circle_distance(
    origin: ArrayLike,
    direction: ArrayLike,
    centre: ArrayLike,
    radius: float,
    tolerance: float = 1.0e-12,
) -> float:
    """Nearest non-negative ray/circle hit, or infinity for no hit."""

    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    c = np.asarray(centre, dtype=np.float64)
    if o.shape != (2,) or d.shape != (2,) or c.shape != (2,):
        raise ValueError("ray and circle values must be two-vectors")
    d_norm = float(np.linalg.norm(d))
    if d_norm <= tolerance:
        raise ValueError("ray direction cannot be zero")
    d = d / d_norm
    relative = o - c
    c_term = float(np.dot(relative, relative) - radius * radius)
    if c_term <= 0.0:
        return 0.0
    b_term = float(np.dot(d, relative))
    discriminant = b_term * b_term - c_term
    if discriminant < -tolerance:
        return math.inf
    root = math.sqrt(max(0.0, discriminant))
    hits = (-b_term - root, -b_term + root)
    forward = [value for value in hits if value >= -tolerance]
    return max(0.0, min(forward)) if forward else math.inf


def ray_workspace_distance(
    origin: ArrayLike,
    direction: ArrayLike,
    workspace: WorkspaceConfig,
    tolerance: float = 1.0e-12,
) -> float:
    """Distance to the first square-boundary intersection."""

    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    if o.shape != (2,) or d.shape != (2,):
        raise ValueError("origin and direction must be two-vectors")
    norm = float(np.linalg.norm(d))
    if norm <= tolerance:
        raise ValueError("ray direction cannot be zero")
    d = d / norm
    candidates: list[float] = []
    for axis in range(2):
        if d[axis] > tolerance:
            candidates.append((workspace.high - o[axis]) / d[axis])
        elif d[axis] < -tolerance:
            candidates.append((workspace.low - o[axis]) / d[axis])
    forward = [value for value in candidates if value >= -tolerance]
    if not forward:
        return 0.0
    return max(0.0, min(forward))


def cast_lidar(
    position: ArrayLike,
    heading: float,
    obstacles: Iterable[tuple[ArrayLike, float]],
    workspace: WorkspaceConfig,
    config: LidarConfig,
) -> FloatArray:
    """Return normalized nearest-hit ranges in counter-clockwise ray order."""

    pos = np.asarray(position, dtype=np.float64)
    if pos.shape != (2,):
        raise ValueError("position must be a two-vector")
    obstacle_list = list(obstacles)
    values = np.empty(config.rays, dtype=np.float64)
    for index in range(config.rays):
        angle = heading + 2.0 * math.pi * index / config.rays
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
        nearest = min(
            config.max_range,
            ray_workspace_distance(pos, direction, workspace),
            *(
                ray_circle_distance(pos, direction, centre, radius)
                for centre, radius in obstacle_list
            ),
        )
        values[index] = np.clip(nearest / config.max_range, 0.0, 1.0)
    return values
