"""Preregistered actor-only deterministic evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .config import EnvironmentConfig
from .env import MAADPGPursuitEnv
from .maddpg import MADDPGLearner


@dataclass(frozen=True)
class EvaluationEpisode:
    scenario_seed: int
    success: bool
    terminated: bool
    truncated: bool
    steps: int
    collision_boundary: bool
    collision_obstacle: bool
    collision_teammate: bool
    collision_target_hard: bool
    final_minimum_obstacle_clearance: float
    final_maximum_target_distance: float


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: tuple[EvaluationEpisode, ...]
    success_rate: float
    wilson_95_low: float
    wilson_95_high: float
    collision_rate: float
    truncation_rate: float
    conditional_mean_steps_to_success: float | None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid binomial counts")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return centre - half, centre + half


def evaluate_actor_only(
    learner: MADDPGLearner,
    environment_config: EnvironmentConfig,
    scenario_seeds: Iterable[int],
) -> EvaluationSummary:
    """Evaluate direct actor forward passes: no noise, PFM, gate, or Peek."""

    episodes: list[EvaluationEpisode] = []
    for seed_value in scenario_seeds:
        seed = int(seed_value)
        env = MAADPGPursuitEnv(config=environment_config)
        observation, _ = env.reset(seed=seed)
        final_info = None
        terminated = truncated = False
        while not (terminated or truncated):
            action = learner.select_actions(observation)
            observation, _, terminated, truncated, final_info = env.step(action)
        if final_info is None:
            raise AssertionError("evaluation episode produced no step")
        collision = final_info["collision"]
        capture = final_info["capture"]
        episodes.append(
            EvaluationEpisode(
                scenario_seed=seed,
                success=bool(capture.success),
                terminated=terminated,
                truncated=truncated,
                steps=int(final_info["step_count"]),
                collision_boundary=bool(np.any(collision["boundary"])),
                collision_obstacle=bool(np.any(collision["obstacle"])),
                collision_teammate=bool(np.any(collision["teammate"])),
                collision_target_hard=bool(np.any(collision["target_hard"])),
                final_minimum_obstacle_clearance=float(
                    capture.minimum_obstacle_clearance
                ),
                final_maximum_target_distance=float(
                    np.max(capture.target_distances)
                ),
            )
        )
    if not episodes:
        raise ValueError("evaluation requires at least one scenario seed")
    successes = sum(episode.success for episode in episodes)
    collision_count = sum(
        episode.collision_boundary
        or episode.collision_obstacle
        or episode.collision_teammate
        or episode.collision_target_hard
        for episode in episodes
    )
    success_steps = [episode.steps for episode in episodes if episode.success]
    low, high = wilson_interval(successes, len(episodes))
    return EvaluationSummary(
        episodes=tuple(episodes),
        success_rate=successes / len(episodes),
        wilson_95_low=low,
        wilson_95_high=high,
        collision_rate=collision_count / len(episodes),
        truncation_rate=sum(episode.truncated for episode in episodes) / len(episodes),
        conditional_mean_steps_to_success=(
            float(np.mean(success_steps)) if success_steps else None
        ),
    )
