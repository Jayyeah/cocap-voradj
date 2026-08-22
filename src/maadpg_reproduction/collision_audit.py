"""Audit-only swept collision checks; never controls Paper-primary termination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


BoolArray = NDArray[np.bool_]


def point_segment_distance(
    point: ArrayLike, start: ArrayLike, end: ArrayLike
) -> float:
    p = np.asarray(point, dtype=np.float64)
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(end, dtype=np.float64)
    displacement = b - a
    squared_length = float(np.dot(displacement, displacement))
    if squared_length == 0.0:
        return float(np.linalg.norm(p - a))
    fraction = float(np.dot(p - a, displacement) / squared_length)
    closest = a + np.clip(fraction, 0.0, 1.0) * displacement
    return float(np.linalg.norm(p - closest))


def moving_points_minimum_distance(
    left_start: ArrayLike,
    left_end: ArrayLike,
    right_start: ArrayLike,
    right_end: ArrayLike,
) -> float:
    relative_start = np.asarray(left_start, dtype=np.float64) - np.asarray(
        right_start, dtype=np.float64
    )
    relative_delta = (
        np.asarray(left_end, dtype=np.float64)
        - np.asarray(left_start, dtype=np.float64)
        - np.asarray(right_end, dtype=np.float64)
        + np.asarray(right_start, dtype=np.float64)
    )
    squared_speed = float(np.dot(relative_delta, relative_delta))
    if squared_speed == 0.0:
        return float(np.linalg.norm(relative_start))
    fraction = float(-np.dot(relative_start, relative_delta) / squared_speed)
    relative = relative_start + np.clip(fraction, 0.0, 1.0) * relative_delta
    return float(np.linalg.norm(relative))


@dataclass(frozen=True)
class SweptCollisionAudit:
    obstacle_by_agent: BoolArray
    teammate_by_agent: BoolArray
    target_by_agent: BoolArray

    @property
    def any_by_agent(self) -> BoolArray:
        return self.obstacle_by_agent | self.teammate_by_agent | self.target_by_agent


def audit_swept_collisions(
    previous_pursuers: ArrayLike,
    current_pursuers: ArrayLike,
    previous_target: ArrayLike,
    current_target: ArrayLike,
    obstacles: Iterable[tuple[ArrayLike, float]],
    *,
    obstacle_clearance: float,
    teammate_separation: float,
    target_hard_radius: float,
) -> SweptCollisionAudit:
    previous = np.asarray(previous_pursuers, dtype=np.float64)
    current = np.asarray(current_pursuers, dtype=np.float64)
    target_start = np.asarray(previous_target, dtype=np.float64)
    target_end = np.asarray(current_target, dtype=np.float64)
    if previous.shape != (3, 2) or current.shape != (3, 2):
        raise ValueError("swept audit requires three fixed pursuer slots")
    obstacle_hits = np.zeros(3, dtype=bool)
    obstacle_list = list(obstacles)
    for index in range(3):
        obstacle_hits[index] = any(
            point_segment_distance(centre, previous[index], current[index])
            - radius
            < obstacle_clearance
            for centre, radius in obstacle_list
        )
    teammate_hits = np.zeros(3, dtype=bool)
    for left in range(3):
        for right in range(left + 1, 3):
            if (
                moving_points_minimum_distance(
                    previous[left], current[left], previous[right], current[right]
                )
                < teammate_separation
            ):
                teammate_hits[left] = teammate_hits[right] = True
    target_hits = np.array(
        [
            moving_points_minimum_distance(
                previous[index], current[index], target_start, target_end
            )
            < target_hard_radius
            for index in range(3)
        ],
        dtype=bool,
    )
    return SweptCollisionAudit(obstacle_hits, teammate_hits, target_hits)
