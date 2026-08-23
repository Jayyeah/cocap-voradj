"""Numerically stable geometry metrics for three-hunter encirclement."""

from __future__ import annotations

import math

import numpy as np


def signed_twice_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return twice the oriented area of triangle ``abc``."""

    ab = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    ac = np.asarray(c, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def point_in_triangle(
    point: np.ndarray,
    triangle: np.ndarray,
    *,
    rel_tol: float = 1e-9,
) -> bool:
    """Return whether ``point`` lies inside or on a non-degenerate triangle.

    The tolerance scales with the squared coordinate span.  Degenerate and
    collinear hunter formations are deliberately not accepted as enclosure.
    """

    tri = np.asarray(triangle, dtype=np.float64)
    p = np.asarray(point, dtype=np.float64)
    if tri.shape != (3, 2) or p.shape != (2,):
        raise ValueError(f"expected triangle=(3,2), point=(2,), got {tri.shape}, {p.shape}")
    span = max(float(np.ptp(np.vstack((tri, p)), axis=0).max()), 1.0)
    tol = rel_tol * span * span
    area2 = signed_twice_area(tri[0], tri[1], tri[2])
    if abs(area2) <= tol:
        return False
    s0 = signed_twice_area(tri[0], tri[1], p)
    s1 = signed_twice_area(tri[1], tri[2], p)
    s2 = signed_twice_area(tri[2], tri[0], p)
    if area2 < 0:
        s0, s1, s2 = -s0, -s1, -s2
    return bool(s0 >= -tol and s1 >= -tol and s2 >= -tol)


def angular_metrics(hunters: np.ndarray, target: np.ndarray) -> tuple[float, list[float]]:
    """Return largest angular gap and sorted circular pairwise gaps in radians."""

    hunters = np.asarray(hunters, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    angles = np.mod(np.arctan2(hunters[:, 1] - target[1], hunters[:, 0] - target[0]), 2 * np.pi)
    angles.sort()
    gaps = np.diff(np.concatenate((angles, angles[:1] + 2 * np.pi)))
    return float(np.max(gaps)), [float(v) for v in np.sort(gaps)]


def triangle_area(triangle: np.ndarray) -> float:
    triangle = np.asarray(triangle, dtype=np.float64)
    return abs(signed_twice_area(triangle[0], triangle[1], triangle[2])) * 0.5


def area_containment_residual(point: np.ndarray, triangle: np.ndarray) -> float:
    """Diagnostic only: sum of subareas minus enclosing triangle area."""

    tri = np.asarray(triangle, dtype=np.float64)
    p = np.asarray(point, dtype=np.float64)
    outer = triangle_area(tri)
    inner = sum(
        triangle_area(np.stack((tri[i], tri[(i + 1) % 3], p)))
        for i in range(3)
    )
    return float(inner - outer)


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2 * math.pi) - math.pi)
