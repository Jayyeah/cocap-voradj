"""Translate legacy IQN discrete (a,w) actions to world acceleration [ax,ay]."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ActionTranslationResult:
    action: np.ndarray
    commanded_terminal_velocity: np.ndarray
    achieved_terminal_velocity: np.ndarray
    terminal_velocity_error: float
    position_error: float
    rejected: bool
    clipped: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.tolist(),
            "terminal_velocity_error": float(self.terminal_velocity_error),
            "position_error": float(self.position_error),
            "rejected": bool(self.rejected),
            "clipped": bool(self.clipped),
        }


class LegacyIQNActionToWorldAccelerationAdapter:
    """Solve a constrained world acceleration matching legacy terminal velocity."""

    def __init__(
        self,
        *,
        a_max: float = 0.4,
        drag_coefficient: float = 0.4 / 3.0,
        dt: float = 0.05,
        substeps: int = 10,
        max_speed: float = 3.0,
        atol: float = 1e-6,
    ):
        self.a_max = float(a_max)
        self.drag = float(drag_coefficient)
        self.dt = float(dt)
        self.substeps = max(int(substeps), 1)
        self.max_speed = float(max_speed)
        self.atol = float(atol)

    def _legacy_terminal(self, velocity: np.ndarray, theta: float, a_scalar: float, omega: float) -> Tuple[np.ndarray, np.ndarray]:
        speed = float(np.linalg.norm(velocity))
        angle = float(theta)
        delta = np.zeros(2, dtype=float)
        for _ in range(self.substeps):
            before = speed * np.asarray([np.cos(angle), np.sin(angle)], dtype=float)
            speed = max(0.0, speed + (a_scalar - self.drag * speed) * self.dt)
            speed = min(speed, self.max_speed)
            angle = angle + omega * self.dt
            after = speed * np.asarray([np.cos(angle), np.sin(angle)], dtype=float)
            delta += 0.5 * (before + after) * self.dt
        return after, delta

    def _world_terminal(self, velocity: np.ndarray, acceleration: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        velocity = np.asarray(velocity, dtype=float).copy()
        delta = np.zeros(2, dtype=float)
        for _ in range(self.substeps):
            before = velocity.copy()
            velocity = velocity + (acceleration - self.drag * velocity) * self.dt
            speed = float(np.linalg.norm(velocity))
            if speed > self.max_speed:
                velocity *= self.max_speed / max(speed, 1e-12)
            delta += 0.5 * (before + velocity) * self.dt
        return velocity, delta

    def translate(self, legacy_action: Any, state: Dict[str, Any]) -> ActionTranslationResult:
        command = np.asarray(legacy_action, dtype=float).reshape(-1)
        if command.shape != (2,) or not np.all(np.isfinite(command)):
            return ActionTranslationResult(
                action=np.zeros(2),
                commanded_terminal_velocity=np.zeros(2),
                achieved_terminal_velocity=np.zeros(2),
                terminal_velocity_error=float("inf"),
                position_error=float("inf"),
                rejected=True,
                clipped=False,
            )
        velocity = np.asarray(state.get("velocity", np.zeros(2)), dtype=float).reshape(-1)
        theta = float(state.get("theta", 0.0))
        v_target, legacy_delta = self._legacy_terminal(velocity, theta, float(command[0]), float(command[1]))
        base_v, _ = self._world_terminal(velocity, np.zeros(2))
        bx, _ = self._world_terminal(velocity, np.asarray([1.0, 0.0]))
        by, _ = self._world_terminal(velocity, np.asarray([0.0, 1.0]))
        basis = np.stack([bx - base_v, by - base_v], axis=1)
        try:
            solution = np.linalg.solve(basis, v_target - base_v)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(basis, v_target - base_v, rcond=None)[0]
        if not np.all(np.isfinite(solution)):
            return ActionTranslationResult(
                action=np.zeros(2),
                commanded_terminal_velocity=v_target,
                achieved_terminal_velocity=base_v,
                terminal_velocity_error=float("inf"),
                position_error=float("inf"),
                rejected=True,
                clipped=False,
            )
        action = solution.astype(float)
        clipped = bool(float(np.linalg.norm(action)) > self.a_max + self.atol)
        if clipped:
            action = action * (self.a_max / max(float(np.linalg.norm(action)), 1e-12))
        achieved_v, achieved_delta = self._world_terminal(velocity, action)
        return ActionTranslationResult(
            action=action,
            commanded_terminal_velocity=v_target,
            achieved_terminal_velocity=achieved_v,
            terminal_velocity_error=float(np.linalg.norm(achieved_v - v_target)),
            position_error=float(np.linalg.norm(achieved_delta - legacy_delta)),
            rejected=False,
            clipped=clipped,
        )
