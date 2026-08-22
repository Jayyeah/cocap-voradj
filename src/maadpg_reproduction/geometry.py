"""Standalone geometry primitives for the MAADPG contract."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def unit_vector(vector: ArrayLike, epsilon: float = 1.0e-12) -> FloatArray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= epsilon:
        return np.zeros_like(value, dtype=np.float64)
    return value / norm


def pairwise_distances(points: ArrayLike) -> FloatArray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape (n,2)")
    delta = values[:, None, :] - values[None, :, :]
    return np.linalg.norm(delta, axis=-1)


def triangle_area(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> float:
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    cv = np.asarray(c, dtype=np.float64)
    cross = np.cross(bv - av, cv - av)
    return 0.5 * abs(float(cross))


def target_fan_area(pursuers: ArrayLike, target: ArrayLike) -> float:
    values = np.asarray(pursuers, dtype=np.float64)
    centre = np.asarray(target, dtype=np.float64)
    if values.shape != (3, 2) or centre.shape != (2,):
        raise ValueError("target fan area requires three pursuers and one target")
    angles = np.arctan2(values[:, 1] - centre[1], values[:, 0] - centre[0])
    order = np.argsort(angles)
    ordered = values[order]
    return sum(
        triangle_area(centre, ordered[index], ordered[(index + 1) % 3])
        for index in range(3)
    )


def point_in_triangle(
    point: ArrayLike,
    triangle: ArrayLike,
    tolerance: float = 1.0e-9,
) -> bool:
    """Return whether point is inside/on a non-degenerate three-point hull."""

    p = np.asarray(point, dtype=np.float64)
    tri = np.asarray(triangle, dtype=np.float64)
    if p.shape != (2,) or tri.shape != (3, 2):
        raise ValueError("expected point (2,) and triangle (3,2)")
    signed = []
    for index in range(3):
        a = tri[index]
        b = tri[(index + 1) % 3]
        signed.append(float(np.cross(b - a, p - a)))
    if triangle_area(tri[0], tri[1], tri[2]) <= tolerance:
        return False
    return (min(signed) >= -tolerance) or (max(signed) <= tolerance)


def maximum_angular_gap(points: ArrayLike, centre: ArrayLike) -> float:
    values = np.asarray(points, dtype=np.float64)
    origin = np.asarray(centre, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or origin.shape != (2,):
        raise ValueError("invalid point/centre shapes")
    offsets = values - origin
    if np.any(np.linalg.norm(offsets, axis=1) == 0.0):
        return 2.0 * math.pi
    angles = np.sort(np.mod(np.arctan2(offsets[:, 1], offsets[:, 0]), 2.0 * math.pi))
    wrapped = np.concatenate([angles, angles[:1] + 2.0 * math.pi])
    return float(np.max(np.diff(wrapped)))


def obstacle_surface_clearance(
    point: ArrayLike, centre: ArrayLike, radius: float
) -> float:
    p = np.asarray(point, dtype=np.float64)
    c = np.asarray(centre, dtype=np.float64)
    return float(np.linalg.norm(p - c) - radius)


def minimum_obstacle_clearance(
    points: ArrayLike, obstacles: Iterable[tuple[ArrayLike, float]]
) -> float:
    values = np.asarray(points, dtype=np.float64)
    obstacle_list = list(obstacles)
    if not obstacle_list:
        return math.inf
    return min(
        obstacle_surface_clearance(point, centre, radius)
        for point in values
        for centre, radius in obstacle_list
    )
