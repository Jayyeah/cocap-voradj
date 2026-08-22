"""Replaceable, restorable target policies for an unreleased paper dependency."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import TargetPolicyConfig, WorkspaceConfig
from .dynamics import canonical_normalized_action
from .geometry import obstacle_surface_clearance, unit_vector


FloatArray = NDArray[np.float64]


class TargetPolicy(Protocol):
    policy_id: str

    def reset(self, seed: int) -> None: ...

    def action(
        self,
        target_position: ArrayLike,
        pursuer_positions: ArrayLike,
        obstacles: Iterable[tuple[ArrayLike, float]],
        workspace: WorkspaceConfig,
    ) -> FloatArray: ...

    def snapshot(self) -> dict[str, Any]: ...

    def restore(self, snapshot: dict[str, Any]) -> None: ...


@dataclass
class StationaryTargetPolicy:
    policy_id: str = "stationary_target_v1"

    def reset(self, seed: int) -> None:
        del seed

    def action(
        self,
        target_position: ArrayLike,
        pursuer_positions: ArrayLike,
        obstacles: Iterable[tuple[ArrayLike, float]],
        workspace: WorkspaceConfig,
    ) -> FloatArray:
        del target_position, pursuer_positions, obstacles, workspace
        return np.zeros(2, dtype=np.float64)

    def snapshot(self) -> dict[str, Any]:
        return {"policy_id": self.policy_id}

    def restore(self, snapshot: dict[str, Any]) -> None:
        if snapshot != {"policy_id": self.policy_id}:
            raise ValueError("invalid stationary target snapshot")


class ScriptedEvasiveOUTargetPolicy:
    policy_id = "scripted_evasive_ou_v1"

    def __init__(self, config: TargetPolicyConfig):
        if config.policy_id != self.policy_id:
            raise ValueError(f"target config requests {config.policy_id}")
        self.config = config
        self._rng = np.random.default_rng(0)
        self._ou_state = np.zeros(2, dtype=np.float64)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._ou_state = np.zeros(2, dtype=np.float64)

    def _obstacle_force(
        self,
        target: FloatArray,
        obstacles: Iterable[tuple[ArrayLike, float]],
    ) -> FloatArray:
        force = np.zeros(2, dtype=np.float64)
        for centre, radius in obstacles:
            centre_array = np.asarray(centre, dtype=np.float64)
            clearance = obstacle_surface_clearance(target, centre_array, radius)
            if clearance >= self.config.obstacle_influence:
                continue
            denominator = max(clearance, self.config.inverse_distance_floor) ** 2
            force += unit_vector(target - centre_array) / denominator
        return unit_vector(force)

    def _boundary_force(
        self, target: FloatArray, workspace: WorkspaceConfig
    ) -> FloatArray:
        force = np.zeros(2, dtype=np.float64)
        distances = (
            target[0] - workspace.low,
            workspace.high - target[0],
            target[1] - workspace.low,
            workspace.high - target[1],
        )
        directions = (
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
        )
        for distance, direction in zip(distances, directions, strict=True):
            if distance < self.config.boundary_influence:
                denominator = max(distance, self.config.inverse_distance_floor) ** 2
                force += direction / denominator
        return unit_vector(force)

    def action(
        self,
        target_position: ArrayLike,
        pursuer_positions: ArrayLike,
        obstacles: Iterable[tuple[ArrayLike, float]],
        workspace: WorkspaceConfig,
    ) -> FloatArray:
        target = np.asarray(target_position, dtype=np.float64)
        pursuers = np.asarray(pursuer_positions, dtype=np.float64)
        if target.shape != (2,) or pursuers.shape != (3, 2):
            raise ValueError("target action expects target (2,) and pursuers (3,2)")
        obstacle_list = list(obstacles)
        flee = unit_vector(target - np.mean(pursuers, axis=0))
        deterministic = (
            self.config.flee_weight * flee
            + self.config.obstacle_weight * self._obstacle_force(target, obstacle_list)
            + self.config.boundary_weight * self._boundary_force(target, workspace)
        )
        normal = self._rng.normal(size=2)
        self._ou_state += (
            self.config.ou_theta * (-self._ou_state) * self.config.ou_dt
            + self.config.ou_sigma * np.sqrt(self.config.ou_dt) * normal
        )
        return canonical_normalized_action(deterministic + self._ou_state)

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "ou_state": self._ou_state.copy(),
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("policy_id") != self.policy_id:
            raise ValueError("target-policy snapshot id mismatch")
        state = np.asarray(snapshot["ou_state"], dtype=np.float64)
        if state.shape != (2,):
            raise ValueError("invalid target OU state")
        self._ou_state = state.copy()
        self._rng.bit_generator.state = deepcopy(snapshot["rng_state"])


def build_target_policy(config: TargetPolicyConfig) -> TargetPolicy:
    if config.policy_id == ScriptedEvasiveOUTargetPolicy.policy_id:
        return ScriptedEvasiveOUTargetPolicy(config)
    if config.policy_id == "stationary_target_v1":
        return StationaryTargetPolicy()
    raise ValueError(f"unknown target policy: {config.policy_id}")
