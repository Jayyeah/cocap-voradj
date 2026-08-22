"""Independent, checkpointable OU exploration for the three pursuer actors."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OUExplorationConfig:
    agent_count: int = 3
    action_dim: int = 2
    theta: float = 0.15
    sigma_start: float = 0.20
    sigma_end: float = 0.05
    dt: float = 1.0
    decay_steps: int = 360_000

    def validate(self) -> None:
        if self.agent_count != 3 or self.action_dim != 2:
            raise ValueError("paper-v1 exploration requires shape (3,2)")
        if self.theta < 0.0 or self.dt <= 0.0:
            raise ValueError("invalid OU theta/dt")
        if self.sigma_start < 0.0 or self.sigma_end < 0.0:
            raise ValueError("OU sigmas cannot be negative")
        if self.decay_steps <= 0:
            raise ValueError("OU decay_steps must be positive")


class IndependentOUNoise:
    def __init__(self, config: OUExplorationConfig | None = None, *, seed: int = 0):
        self.config = config or OUExplorationConfig()
        self.config.validate()
        self._rng = np.random.default_rng(seed)
        self.state = np.zeros(
            (self.config.agent_count, self.config.action_dim), dtype=np.float64
        )

    def sigma(self, environment_step: int) -> float:
        fraction = np.clip(environment_step / self.config.decay_steps, 0.0, 1.0)
        return float(
            self.config.sigma_start
            + fraction * (self.config.sigma_end - self.config.sigma_start)
        )

    def sample(self, environment_step: int) -> FloatArray:
        sigma = self.sigma(environment_step)
        normal = self._rng.normal(size=self.state.shape)
        self.state += (
            self.config.theta * (-self.state) * self.config.dt
            + sigma * np.sqrt(self.config.dt) * normal
        )
        return self.state.copy()

    def reset_episode(self) -> None:
        self.state.fill(0.0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "state": self.state.copy(),
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["config"] != self.config:
            raise ValueError("OU exploration configuration mismatch")
        values = np.asarray(state["state"], dtype=np.float64)
        if values.shape != self.state.shape:
            raise ValueError("OU exploration state shape mismatch")
        self.state = values.copy()
        self._rng.bit_generator.state = deepcopy(state["rng_state"])
