"""Non-learning controls used only for R0 task-solvability gates."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dynamics import canonical_normalized_action


FloatArray = NDArray[np.float64]


def fixed_ring_pd_actions(
    pursuer_positions: ArrayLike,
    pursuer_velocities: ArrayLike,
    target_position: ArrayLike,
    *,
    acceleration_limit: float,
    desired_radius: float = 0.115,
    kp: float = 0.025,
    kd: float = 0.9,
) -> FloatArray:
    """Drive fixed pursuer slots to a 120-degree ring around a stationary target.

    This controller is an environment oracle, not a Paper-primary algorithm and
    never participates in MADDPG data collection or evaluation.
    """

    positions = np.asarray(pursuer_positions, dtype=np.float64)
    velocities = np.asarray(pursuer_velocities, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    if positions.shape != (3, 2) or velocities.shape != (3, 2):
        raise ValueError("ring oracle expects three pursuer slots")
    if target.shape != (2,) or acceleration_limit <= 0.0:
        raise ValueError("invalid target or acceleration limit")
    angles = np.arange(3, dtype=np.float64) * 2.0 * math.pi / 3.0
    desired = target + desired_radius * np.stack(
        [np.cos(angles), np.sin(angles)], axis=1
    )
    physical = kp * (desired - positions) - kd * velocities
    return np.stack(
        [
            canonical_normalized_action(value / acceleration_limit)
            for value in physical
        ]
    )
