"""Strict Eq. 18 capture predicate with inspectable clause values."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import CaptureConfig, WorkspaceConfig
from .geometry import (
    maximum_angular_gap,
    minimum_obstacle_clearance,
    pairwise_distances,
    point_in_triangle,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CaptureEvaluation:
    success: bool
    target_in_hull: bool
    radial_band_ok: bool
    angular_gap_ok: bool
    boundary_ok: bool
    separation_ok: bool
    obstacle_clearance_ok: bool
    target_distances: FloatArray
    maximum_angular_gap: float
    minimum_pursuer_separation: float
    minimum_obstacle_clearance: float

    def clauses(self) -> dict[str, bool]:
        return {
            "target_in_hull": self.target_in_hull,
            "radial_band_ok": self.radial_band_ok,
            "angular_gap_ok": self.angular_gap_ok,
            "boundary_ok": self.boundary_ok,
            "separation_ok": self.separation_ok,
            "obstacle_clearance_ok": self.obstacle_clearance_ok,
        }


def evaluate_capture(
    pursuer_positions: ArrayLike,
    target_position: ArrayLike,
    obstacles: Iterable[tuple[ArrayLike, float]],
    workspace: WorkspaceConfig,
    config: CaptureConfig,
) -> CaptureEvaluation:
    pursuers = np.asarray(pursuer_positions, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    if pursuers.shape != (3, 2) or target.shape != (2,):
        raise ValueError("capture requires pursuers (3,2) and target (2,)")
    distances = np.linalg.norm(pursuers - target, axis=1)
    target_in_hull = point_in_triangle(target, pursuers, config.tolerance)
    radial_band_ok = bool(
        np.all(distances >= config.rho_min - config.tolerance)
        and np.all(distances <= config.rho_max + config.tolerance)
    )
    angular_gap = maximum_angular_gap(pursuers, target)
    angular_gap_ok = angular_gap <= config.max_angular_gap + config.tolerance
    all_positions = np.concatenate([pursuers, target[None, :]], axis=0)
    boundary_ok = bool(
        np.all(all_positions >= workspace.low - config.tolerance)
        and np.all(all_positions <= workspace.high + config.tolerance)
    )
    pairwise = pairwise_distances(pursuers)
    triangle_pairs = pairwise[np.triu_indices(3, k=1)]
    min_separation = float(np.min(triangle_pairs))
    separation_ok = min_separation >= (
        config.min_pursuer_separation - config.tolerance
    )
    min_obstacle = minimum_obstacle_clearance(pursuers, obstacles)
    obstacle_ok = min_obstacle >= config.min_obstacle_clearance - config.tolerance
    clauses = (
        target_in_hull,
        radial_band_ok,
        angular_gap_ok,
        boundary_ok,
        separation_ok,
        obstacle_ok,
    )
    return CaptureEvaluation(
        success=all(clauses),
        target_in_hull=target_in_hull,
        radial_band_ok=radial_band_ok,
        angular_gap_ok=angular_gap_ok,
        boundary_ok=boundary_ok,
        separation_ok=separation_ok,
        obstacle_clearance_ok=obstacle_ok,
        target_distances=distances,
        maximum_angular_gap=angular_gap,
        minimum_pursuer_separation=min_separation,
        minimum_obstacle_clearance=min_obstacle,
    )
