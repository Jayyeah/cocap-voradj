"""Typed executable assumptions for ``maadpg-paper-v1``.

The paper-derived facts and the justification for every assumed value live in
``docs/maadpg_reproduction_20260823``.  This module intentionally has no import
from the legacy CoCap package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any

from . import SPEC_VERSION


@dataclass(frozen=True)
class WorkspaceConfig:
    low: float = 0.0
    high: float = 2.0

    @property
    def side_length(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class AgentLimits:
    vmax: float
    amax: float


@dataclass(frozen=True)
class ResetConfig:
    boundary_margin: float = 0.05
    obstacle_inter_clearance: float = 0.02
    spawn_obstacle_clearance: float = 0.02
    spawn_agent_clearance: float = 0.04
    max_rejection_attempts: int = 10_000


@dataclass(frozen=True)
class LidarConfig:
    rays: int = 16
    max_range: float = 0.20


@dataclass(frozen=True)
class CaptureConfig:
    rho_min: float = 0.08
    rho_max: float = 0.15
    max_angular_gap: float = 5.0 * math.pi / 6.0
    min_pursuer_separation: float = 0.04
    min_obstacle_clearance: float = 0.02
    tolerance: float = 1.0e-9


@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float = 1.0
    safety_weight: float = 1.0
    stage_weight: float = 1.0
    separation_weight: float = 1.0
    target_hit_weight: float = 1.0
    collision_penalty: float = 10.0
    success_bonus: float = 100.0
    stage_d_cap: float = 0.15
    stage_d_limit_sum: float = 0.45
    stage_d_max_sum: float = 6.0 * math.sqrt(2.0)
    stage_s4: float = 3.0 * math.sqrt(3.0) * 0.15**2 / 4.0
    separation_radius: float = 0.04
    separation_coefficient: float = 1.0
    target_soft_radius: float = 0.05
    target_hard_radius: float = 0.02
    target_soft_coefficient: float = 1.0
    target_hard_penalty: float = 10.0


@dataclass(frozen=True)
class TargetPolicyConfig:
    policy_id: str = "scripted_evasive_ou_v1"
    flee_weight: float = 0.75
    obstacle_weight: float = 0.15
    boundary_weight: float = 0.10
    obstacle_influence: float = 0.20
    boundary_influence: float = 0.20
    inverse_distance_floor: float = 0.01
    ou_theta: float = 0.15
    ou_sigma: float = 0.20
    ou_dt: float = 1.0


@dataclass(frozen=True)
class EnvironmentConfig:
    spec_version: str = SPEC_VERSION
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    pursuer_limits: AgentLimits = field(
        default_factory=lambda: AgentLimits(vmax=0.010, amax=0.004)
    )
    target_limits: AgentLimits = field(
        default_factory=lambda: AgentLimits(vmax=0.011, amax=0.005)
    )
    reset: ResetConfig = field(default_factory=ResetConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    target_policy: TargetPolicyConfig = field(default_factory=TargetPolicyConfig)
    dt: float = 1.0
    horizon_steps: int = 300
    pursuer_count: int = 3
    obstacle_count_min: int = 0
    obstacle_count_max: int = 3
    obstacle_radius_min: float = 0.10
    obstacle_radius_max: float = 0.15
    distance_normalizer: float = 4.0
    initial_heading: float = 0.0

    @property
    def pursuer_observation_dim(self) -> int:
        return 4 + 2 * (self.pursuer_count - 1) + self.lidar.rays + 2

    def validate(self) -> None:
        if self.spec_version != SPEC_VERSION:
            raise ValueError(f"unsupported spec_version: {self.spec_version}")
        if self.pursuer_count != 3:
            raise ValueError("maadpg-paper-v1 requires exactly three pursuers")
        if self.pursuer_observation_dim != 26:
            raise ValueError("Eq. 24 observation must contain exactly 26 scalars")
        if self.dt <= 0.0 or self.horizon_steps <= 0:
            raise ValueError("dt and horizon_steps must be positive")
        if self.workspace.side_length <= 0.0:
            raise ValueError("workspace must have positive extent")
        if not 0.0 <= self.capture.rho_min <= self.capture.rho_max:
            raise ValueError("capture radial band is invalid")
        if self.reward.target_hard_radius > self.reward.target_soft_radius:
            raise ValueError("hard target radius cannot exceed soft radius")
        weights = (
            self.target_policy.flee_weight,
            self.target_policy.obstacle_weight,
            self.target_policy.boundary_weight,
        )
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("target deterministic force weights must sum to one")

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def default_environment_config() -> EnvironmentConfig:
    config = EnvironmentConfig()
    config.validate()
    return config
