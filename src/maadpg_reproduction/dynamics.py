"""Paper-primary point-mass USV dynamics and canonical action handling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import AgentLimits, WorkspaceConfig


FloatArray = NDArray[np.float64]


def _vec2(value: ArrayLike, *, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def project_l2(value: ArrayLike, radius: float = 1.0) -> FloatArray:
    """Project a two-vector onto a closed L2 ball."""

    vector = _vec2(value, name="value")
    if radius < 0.0 or not np.isfinite(radius):
        raise ValueError("radius must be finite and non-negative")
    norm = float(np.linalg.norm(vector))
    if norm <= radius or norm == 0.0:
        return vector.copy()
    return vector * (radius / norm)


def canonical_normalized_action(action: ArrayLike) -> FloatArray:
    """Clip the network box and then enforce the paper's acceleration norm."""

    boxed = np.clip(_vec2(action, name="action"), -1.0, 1.0)
    return project_l2(boxed, 1.0)


@dataclass(frozen=True)
class DynamicsTransition:
    position: FloatArray
    velocity: FloatArray
    proposed_position: FloatArray
    executed_normalized_action: FloatArray
    physical_acceleration: FloatArray
    boundary_crossing: bool


def integrate_usv(
    position: ArrayLike,
    velocity: ArrayLike,
    normalized_action: ArrayLike,
    limits: AgentLimits,
    workspace: WorkspaceConfig,
    dt: float,
) -> DynamicsTransition:
    """Apply Eq. 12 using its displayed velocity-then-position update order."""

    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    pos = _vec2(position, name="position")
    vel = _vec2(velocity, name="velocity")
    action = canonical_normalized_action(normalized_action)
    acceleration = action * limits.amax
    next_velocity = project_l2(vel + acceleration * dt, limits.vmax)
    proposed = pos + next_velocity * dt
    crossing = bool(
        np.any(proposed < workspace.low) or np.any(proposed > workspace.high)
    )
    next_position = np.clip(proposed, workspace.low, workspace.high)
    return DynamicsTransition(
        position=next_position,
        velocity=next_velocity,
        proposed_position=proposed,
        executed_normalized_action=action,
        physical_acceleration=acceleration,
        boundary_crossing=crossing,
    )
