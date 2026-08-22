"""Frozen PFM and adaptive-difference gate assumptions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PotentialFieldConfig:
    target_gain: float = 1.0
    obstacle_gain: float = 1.0
    teammate_gain: float = 1.0
    epsilon: float = 1.0e-8
    inverse_square_distance_floor: float = 0.01
    obstacle_influence_cutoff: float = 0.20

    def validate(self) -> None:
        if min(self.target_gain, self.obstacle_gain, self.teammate_gain) < 0.0:
            raise ValueError("potential-field gains cannot be negative")
        if self.epsilon <= 0.0 or self.inverse_square_distance_floor <= 0.0:
            raise ValueError("PFM numerical floors must be positive")
        if self.obstacle_influence_cutoff <= 0.0:
            raise ValueError("obstacle influence cutoff must be positive")


@dataclass(frozen=True)
class AdaptiveGateConfig:
    denominator_lambda: float = 1.0e-6
    beta_ratio: float = 0.1

    def validate(self) -> None:
        if self.denominator_lambda <= 0.0:
            raise ValueError("gate denominator stabilizer must be positive")
        if self.beta_ratio < 0.0:
            raise ValueError("gate threshold cannot be negative")
