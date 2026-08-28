"""Contracts for explicit two-dimensional body-frame acceleration actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class ActionContractError(ValueError):
    """Raised when an already-executed command violates the action contract."""


@dataclass(frozen=True)
class AccelerationActionDiagnostics:
    commanded_acceleration: float
    validated_acceleration: float
    action_rejected: bool
    validation_delta: float
    speed_before: float = 0.0
    speed_after: float = 0.0
    speed_limited: bool = False
    actual_acceleration: float = 0.0
    jerk: float = 0.0
    validation_rate: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "commanded_acceleration": float(self.commanded_acceleration),
            "validated_acceleration": float(self.validated_acceleration),
            "action_rejected": bool(self.action_rejected),
            "validation_delta": float(self.validation_delta),
            "speed_before": float(self.speed_before),
            "speed_after": float(self.speed_after),
            "speed_limited": bool(self.speed_limited),
            "actual_acceleration": float(self.actual_acceleration),
            "jerk": float(self.jerk),
            "validation_rate": float(self.validation_rate),
        }


def _vector2(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (2,):
        raise ActionContractError(f"{name} must have shape (2,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ActionContractError(f"{name} contains NaN or Inf")
    return vector.copy()


def body_to_world(vector_body: Any, yaw: float) -> np.ndarray:
    vector = _vector2(vector_body, "vector_body")
    angle = float(yaw)
    if not np.isfinite(angle):
        raise ActionContractError("yaw must be finite")
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]], dtype=float)


def world_to_body(vector_world: Any, yaw: float) -> np.ndarray:
    vector = _vector2(vector_world, "vector_world")
    angle = float(yaw)
    if not np.isfinite(angle):
        raise ActionContractError("yaw must be finite")
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([c * vector[0] + s * vector[1], -s * vector[0] + c * vector[1]], dtype=float)


class AccelerationActionAdapter:
    """Validate an explicit body-frame acceleration action.

    The policy owns the hard ``||a|| <= a_max`` bound.  This adapter deliberately
    does not enforce ``v_max`` and never silently clips an action.  The latter is
    an environment/dynamics constraint applied while integrating the velocity.
    ``project`` remains as a compatibility-shaped method, but it is a validating
    identity operation and raises on a materially invalid action.
    """

    def __init__(self, a_max: float, decision_dt: float, atol: float = 1e-6):
        self.a_max = float(a_max)
        self.decision_dt = float(decision_dt)
        self.atol = float(atol)
        if not np.isfinite(self.a_max) or self.a_max <= 0.0:
            raise ValueError("a_max must be finite and positive")
        if not np.isfinite(self.decision_dt) or self.decision_dt <= 0.0:
            raise ValueError("decision_dt must be finite and positive")

    def validate(self, action_body: Any) -> np.ndarray:
        action = _vector2(action_body, "acceleration_body")
        if float(np.linalg.norm(action)) > self.a_max + self.atol:
            raise ActionContractError("acceleration_body violates a_max")
        return action

    def validate_with_diagnostics(self, action_body: Any) -> tuple[np.ndarray, AccelerationActionDiagnostics]:
        action = self.validate(action_body)
        magnitude = float(np.linalg.norm(action))
        diagnostics = AccelerationActionDiagnostics(
            commanded_acceleration=magnitude,
            validated_acceleration=magnitude,
            action_rejected=False,
            validation_delta=0.0,
            actual_acceleration=magnitude,
        )
        return action.astype(float, copy=False), diagnostics

    def project(self, action_body: Any, previous_body: Any = None) -> tuple[np.ndarray, AccelerationActionDiagnostics]:
        """Compatibility alias; validation is an identity operation."""
        del previous_body
        return self.validate_with_diagnostics(action_body)


@dataclass(frozen=True)
class AccelerationAngularVelocityDiagnostics:
    """Validation diagnostics for the scalar ``(a, w)`` bridge action."""

    commanded_acceleration: float
    validated_acceleration: float
    commanded_angular_velocity: float
    validated_angular_velocity: float
    action_rejected: bool
    validation_delta: float
    speed_before: float = 0.0
    speed_after: float = 0.0
    speed_limited: bool = False
    actual_acceleration: float = 0.0
    jerk: float = 0.0
    validation_rate: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "commanded_acceleration": float(self.commanded_acceleration),
            "validated_acceleration": float(self.validated_acceleration),
            "commanded_angular_velocity": float(self.commanded_angular_velocity),
            "validated_angular_velocity": float(self.validated_angular_velocity),
            "action_rejected": bool(self.action_rejected),
            "validation_delta": float(self.validation_delta),
            "speed_before": float(self.speed_before),
            "speed_after": float(self.speed_after),
            "speed_limited": bool(self.speed_limited),
            "actual_acceleration": float(self.actual_acceleration),
            "jerk": float(self.jerk),
            "validation_rate": float(self.validation_rate),
        }


class AccelerationAngularVelocityActionAdapter:
    """Validate a bounded scalar acceleration/angular-velocity command.

    The first coordinate is the forward acceleration in m/s² and the second
    is yaw rate in rad/s.  Bounds are component-wise, matching the legacy IQN
    action grid while retaining a continuous SAC action and replay contract.
    """

    def __init__(self, a_max: float, w_max: float, decision_dt: float, atol: float = 1e-6):
        self.a_max = float(a_max)
        self.w_max = float(w_max)
        self.decision_dt = float(decision_dt)
        self.atol = float(atol)
        if not np.isfinite(self.a_max) or self.a_max <= 0.0:
            raise ValueError("a_max must be finite and positive")
        if not np.isfinite(self.w_max) or self.w_max <= 0.0:
            raise ValueError("w_max must be finite and positive")
        if not np.isfinite(self.decision_dt) or self.decision_dt <= 0.0:
            raise ValueError("decision_dt must be finite and positive")

    def validate(self, action: Any) -> np.ndarray:
        value = _vector2(action, "acceleration_angular_velocity")
        if abs(float(value[0])) > self.a_max + self.atol:
            raise ActionContractError("forward acceleration violates a_max")
        if abs(float(value[1])) > self.w_max + self.atol:
            raise ActionContractError("angular velocity violates w_max")
        return value

    def validate_with_diagnostics(self, action: Any) -> tuple[np.ndarray, AccelerationAngularVelocityDiagnostics]:
        value = self.validate(action)
        diagnostics = AccelerationAngularVelocityDiagnostics(
            commanded_acceleration=abs(float(value[0])),
            validated_acceleration=abs(float(value[0])),
            commanded_angular_velocity=float(value[1]),
            validated_angular_velocity=float(value[1]),
            action_rejected=False,
            validation_delta=0.0,
            actual_acceleration=abs(float(value[0])),
        )
        return value.astype(float, copy=False), diagnostics

    def project(self, action: Any, previous_body: Any = None) -> tuple[np.ndarray, AccelerationAngularVelocityDiagnostics]:
        del previous_body
        return self.validate_with_diagnostics(action)


class DesiredBodyVelocityActionAdapter:
    """Validate a rate-limited body-frame velocity target for VXY9."""

    def __init__(self, v_max: float, decision_dt: float, atol: float = 1e-6):
        self.v_max = float(v_max)
        self.decision_dt = float(decision_dt)
        self.atol = float(atol)
        if not np.isfinite(self.v_max) or self.v_max <= 0.0:
            raise ValueError("v_max must be finite and positive")
        if not np.isfinite(self.decision_dt) or self.decision_dt <= 0.0:
            raise ValueError("decision_dt must be finite and positive")

    def validate(self, action: Any) -> np.ndarray:
        value = _vector2(action, "desired_velocity_body")
        if float(np.linalg.norm(value)) > self.v_max + self.atol:
            raise ActionContractError("desired_velocity_body violates v_max")
        return value

    def validate_with_diagnostics(self, action: Any) -> tuple[np.ndarray, AccelerationActionDiagnostics]:
        value = self.validate(action)
        magnitude = float(np.linalg.norm(value))
        return value, AccelerationActionDiagnostics(
            commanded_acceleration=magnitude,
            validated_acceleration=magnitude,
            action_rejected=False,
            validation_delta=0.0,
        )


# Compatibility aliases for old isolated artifacts. New code must use the
# acceleration names above; aliases do not preserve the old velocity semantics.
VelocityCommandDiagnostics = AccelerationActionDiagnostics
FeasibleVelocityAdapter = AccelerationActionAdapter
