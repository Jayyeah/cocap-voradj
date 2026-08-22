"""Inspectable implementation of the paper's Eq. 28-33 reward family."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import AgentLimits, RewardConfig
from .geometry import pairwise_distances, target_fan_area, unit_vector


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RewardTerms:
    progress: FloatArray
    safety: FloatArray
    stage: FloatArray
    separation: FloatArray
    target_hit: FloatArray
    terminal: FloatArray
    total: FloatArray
    stage_branch: str
    target_fan_area: float
    target_distance_sum: float


def _stage_reward(
    pursuer_positions: FloatArray,
    target_position: FloatArray,
    previous_target_distances: FloatArray,
    limits: AgentLimits,
    config: RewardConfig,
) -> tuple[float, str, float, float]:
    distances = np.linalg.norm(pursuer_positions - target_position, axis=1)
    distance_sum = float(np.sum(distances))
    previous_sum = float(np.sum(previous_target_distances))
    area = target_fan_area(pursuer_positions, target_position)
    if (
        area > config.stage_s4
        and distance_sum >= config.stage_d_limit_sum
        and float(np.min(distances)) >= config.stage_d_cap
    ):
        return -distance_sum / config.stage_d_max_sum, "approach", area, distance_sum
    if area > config.stage_s4 and (
        distance_sum < config.stage_d_limit_sum
        or float(np.min(distances)) < config.stage_d_cap
    ):
        return -(1.0 / 3.0) * math.log(area - config.stage_s4 + 1.0), "encircle", area, distance_sum
    if area <= config.stage_s4 and float(np.max(distances)) > config.stage_d_cap:
        exponent = (previous_sum - distance_sum) / (3.0 * limits.vmax)
        return math.exp(exponent), "close", area, distance_sum
    return 0.0, "fallback", area, distance_sum


def compute_reward_terms(
    pursuer_positions: ArrayLike,
    pursuer_velocities: ArrayLike,
    target_position: ArrayLike,
    previous_target_distances: ArrayLike,
    minimum_lidar_ranges: ArrayLike,
    collision_by_agent: Sequence[bool],
    success: bool,
    limits: AgentLimits,
    sensor_range: float,
    config: RewardConfig,
) -> RewardTerms:
    pursuers = np.asarray(pursuer_positions, dtype=np.float64)
    velocities = np.asarray(pursuer_velocities, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    previous_distances = np.asarray(previous_target_distances, dtype=np.float64)
    lidar_ranges = np.asarray(minimum_lidar_ranges, dtype=np.float64)
    collisions = np.asarray(collision_by_agent, dtype=bool)
    if pursuers.shape != (3, 2) or velocities.shape != (3, 2):
        raise ValueError("reward expects three pursuer positions and velocities")
    if previous_distances.shape != (3,) or lidar_ranges.shape != (3,):
        raise ValueError("reward distance/lidar vectors must have shape (3,)")

    direction = np.stack([unit_vector(target - position) for position in pursuers])
    progress = np.sum(velocities * direction, axis=1) / limits.vmax
    safety = np.where(
        collisions,
        -config.collision_penalty,
        (lidar_ranges - sensor_range) / sensor_range,
    )

    stage_scalar, stage_branch, area, distance_sum = _stage_reward(
        pursuers, target, previous_distances, limits, config
    )
    stage = np.full(3, stage_scalar, dtype=np.float64)

    distances_between = pairwise_distances(pursuers)
    separation = np.zeros(3, dtype=np.float64)
    for index in range(3):
        neighbours = np.delete(distances_between[index], index)
        deficits = np.maximum(0.0, config.separation_radius - neighbours)
        separation[index] = -config.separation_coefficient * float(
            np.sum(deficits / config.separation_radius)
        )

    target_distances = np.linalg.norm(pursuers - target, axis=1)
    target_hit = np.zeros(3, dtype=np.float64)
    hard = target_distances < config.target_hard_radius
    soft = (~hard) & (target_distances < config.target_soft_radius)
    target_hit[hard] = -config.target_hard_penalty
    target_hit[soft] = -config.target_soft_coefficient * (
        config.target_soft_radius - target_distances[soft]
    ) / config.target_soft_radius

    terminal = np.full(3, config.success_bonus if success else 0.0)
    total = (
        config.progress_weight * progress
        + config.safety_weight * safety
        + config.stage_weight * stage
        + config.separation_weight * separation
        + config.target_hit_weight * target_hit
        + terminal
    )
    return RewardTerms(
        progress=progress,
        safety=safety,
        stage=stage,
        separation=separation,
        target_hit=target_hit,
        terminal=terminal,
        total=total,
        stage_branch=stage_branch,
        target_fan_area=area,
        target_distance_sum=distance_sum,
    )
