"""Independent three-pursuer MAADPG environment.

The API mirrors the useful part of Gymnasium's reset/step convention without
requiring Gymnasium: observations are ``(3,26)``, rewards are ``(3,)``, and true
termination is separate from horizon truncation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import EnvironmentConfig, default_environment_config
from .dynamics import DynamicsTransition, integrate_usv
from .geometry import obstacle_surface_clearance, pairwise_distances
from .lidar import cast_lidar
from .reward import RewardTerms, compute_reward_terms
from .success import CaptureEvaluation, evaluate_capture
from .target_policy import TargetPolicy, build_target_policy


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Obstacle:
    centre: FloatArray
    radius: float

    def as_pair(self) -> tuple[FloatArray, float]:
        return self.centre, self.radius


@dataclass
class EnvironmentState:
    pursuer_positions: FloatArray
    pursuer_velocities: FloatArray
    pursuer_headings: FloatArray
    target_position: FloatArray
    target_velocity: FloatArray
    target_heading: float
    obstacles: tuple[Obstacle, ...]
    step_count: int

    def copy(self) -> "EnvironmentState":
        return EnvironmentState(
            pursuer_positions=self.pursuer_positions.copy(),
            pursuer_velocities=self.pursuer_velocities.copy(),
            pursuer_headings=self.pursuer_headings.copy(),
            target_position=self.target_position.copy(),
            target_velocity=self.target_velocity.copy(),
            target_heading=float(self.target_heading),
            obstacles=tuple(
                Obstacle(obstacle.centre.copy(), obstacle.radius)
                for obstacle in self.obstacles
            ),
            step_count=int(self.step_count),
        )


@dataclass(frozen=True)
class EnvironmentSnapshot:
    state: EnvironmentState
    scenario_rng_state: dict[str, Any]
    target_policy_state: dict[str, Any]
    seed: int
    episode_done: bool


class MAADPGPursuitEnv:
    """Contract-first point-mass pursuit environment."""

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        target_policy: TargetPolicy | None = None,
    ) -> None:
        self.config = config or default_environment_config()
        self.config.validate()
        self.target_policy = target_policy or build_target_policy(
            self.config.target_policy
        )
        self._rng = np.random.default_rng(0)
        self._seed = 0
        self._episode_done = True
        self.state: EnvironmentState | None = None

    @property
    def obstacle_pairs(self) -> tuple[tuple[FloatArray, float], ...]:
        state = self._require_state()
        return tuple(obstacle.as_pair() for obstacle in state.obstacles)

    def _require_state(self) -> EnvironmentState:
        if self.state is None:
            raise RuntimeError("environment must be reset before use")
        return self.state

    def _sample_obstacles(self) -> tuple[Obstacle, ...]:
        count = int(
            self._rng.integers(
                self.config.obstacle_count_min,
                self.config.obstacle_count_max + 1,
            )
        )
        obstacles: list[Obstacle] = []
        for _ in range(count):
            for _attempt in range(self.config.reset.max_rejection_attempts):
                radius = float(
                    self._rng.uniform(
                        self.config.obstacle_radius_min,
                        self.config.obstacle_radius_max,
                    )
                )
                centre = self._rng.uniform(
                    self.config.workspace.low + radius,
                    self.config.workspace.high - radius,
                    size=2,
                )
                if all(
                    np.linalg.norm(centre - existing.centre)
                    >= radius
                    + existing.radius
                    + self.config.reset.obstacle_inter_clearance
                    for existing in obstacles
                ):
                    obstacles.append(Obstacle(centre.astype(np.float64), radius))
                    break
            else:
                raise RuntimeError("failed to sample a valid obstacle layout")
        return tuple(obstacles)

    def _sample_position(
        self,
        obstacles: tuple[Obstacle, ...],
        existing: list[FloatArray],
    ) -> FloatArray:
        low = self.config.workspace.low + self.config.reset.boundary_margin
        high = self.config.workspace.high - self.config.reset.boundary_margin
        for _attempt in range(self.config.reset.max_rejection_attempts):
            candidate = self._rng.uniform(low, high, size=2).astype(np.float64)
            obstacle_ok = all(
                obstacle_surface_clearance(
                    candidate, obstacle.centre, obstacle.radius
                )
                >= self.config.reset.spawn_obstacle_clearance
                for obstacle in obstacles
            )
            agents_ok = all(
                np.linalg.norm(candidate - other)
                >= self.config.reset.spawn_agent_clearance
                for other in existing
            )
            if obstacle_ok and agents_ok:
                return candidate
        raise RuntimeError("failed to sample a valid agent position")

    @staticmethod
    def _coerce_obstacles(value: Iterable[Any]) -> tuple[Obstacle, ...]:
        result: list[Obstacle] = []
        for raw in value:
            if isinstance(raw, Obstacle):
                obstacle = Obstacle(raw.centre.copy(), float(raw.radius))
            else:
                centre, radius = raw
                obstacle = Obstacle(np.asarray(centre, dtype=np.float64), float(radius))
            if obstacle.centre.shape != (2,) or obstacle.radius <= 0.0:
                raise ValueError("invalid obstacle specification")
            result.append(obstacle)
        return tuple(result)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        self._seed = int(0 if seed is None else seed)
        scenario_sequence, target_sequence = np.random.SeedSequence(self._seed).spawn(2)
        self._rng = np.random.default_rng(scenario_sequence)
        target_seed = int(target_sequence.generate_state(1, dtype=np.uint64)[0])
        self.target_policy.reset(target_seed)
        provided = dict(options or {})
        obstacles = (
            self._coerce_obstacles(provided["obstacles"])
            if "obstacles" in provided
            else self._sample_obstacles()
        )
        existing: list[FloatArray] = []
        if "pursuer_positions" in provided:
            pursuer_positions = np.asarray(
                provided["pursuer_positions"], dtype=np.float64
            )
            if pursuer_positions.shape != (3, 2):
                raise ValueError("pursuer_positions must have shape (3,2)")
            existing.extend(position.copy() for position in pursuer_positions)
        else:
            sampled: list[FloatArray] = []
            for _ in range(self.config.pursuer_count):
                position = self._sample_position(obstacles, existing)
                sampled.append(position)
                existing.append(position)
            pursuer_positions = np.stack(sampled)
        if "target_position" in provided:
            target_position = np.asarray(
                provided["target_position"], dtype=np.float64
            )
            if target_position.shape != (2,):
                raise ValueError("target_position must have shape (2,)")
        else:
            target_position = self._sample_position(obstacles, existing)

        pursuer_velocities = np.asarray(
            provided.get("pursuer_velocities", np.zeros((3, 2))),
            dtype=np.float64,
        )
        target_velocity = np.asarray(
            provided.get("target_velocity", np.zeros(2)), dtype=np.float64
        )
        pursuer_headings = np.asarray(
            provided.get(
                "pursuer_headings",
                np.full(3, self.config.initial_heading, dtype=np.float64),
            ),
            dtype=np.float64,
        )
        target_heading = float(
            provided.get("target_heading", self.config.initial_heading)
        )
        if pursuer_velocities.shape != (3, 2):
            raise ValueError("pursuer_velocities must have shape (3,2)")
        if target_velocity.shape != (2,) or pursuer_headings.shape != (3,):
            raise ValueError("invalid velocity or heading shape")
        self.state = EnvironmentState(
            pursuer_positions=pursuer_positions.copy(),
            pursuer_velocities=pursuer_velocities.copy(),
            pursuer_headings=pursuer_headings.copy(),
            target_position=target_position.copy(),
            target_velocity=target_velocity.copy(),
            target_heading=target_heading,
            obstacles=obstacles,
            step_count=0,
        )
        self._episode_done = False
        observation = self.observation()
        return observation, self._base_info()

    @staticmethod
    def _updated_heading(velocity: FloatArray, previous: float) -> float:
        if float(np.linalg.norm(velocity)) <= 1.0e-12:
            return float(previous)
        return float(math.atan2(velocity[1], velocity[0]))

    def observation(self) -> FloatArray:
        state = self._require_state()
        observations: list[FloatArray] = []
        for index in range(self.config.pursuer_count):
            lidar = cast_lidar(
                state.pursuer_positions[index],
                state.pursuer_headings[index],
                self.obstacle_pairs,
                self.config.workspace,
                self.config.lidar,
            )
            other_positions = np.concatenate(
                [
                    state.pursuer_positions[other] / self.config.workspace.side_length
                    for other in range(self.config.pursuer_count)
                    if other != index
                ]
            )
            offset = state.target_position - state.pursuer_positions[index]
            local = np.concatenate(
                [
                    state.pursuer_positions[index]
                    / self.config.workspace.side_length,
                    state.pursuer_velocities[index]
                    / self.config.pursuer_limits.vmax,
                    other_positions,
                    lidar,
                    np.array(
                        [
                            np.linalg.norm(offset) / self.config.distance_normalizer,
                            math.atan2(offset[1], offset[0]),
                        ]
                    ),
                ]
            )
            if local.shape != (self.config.pursuer_observation_dim,):
                raise AssertionError(f"observation shape drifted to {local.shape}")
            observations.append(local)
        return np.stack(observations).astype(np.float64)

    def _collision_events(
        self,
        transitions: list[DynamicsTransition],
        target_transition: DynamicsTransition,
    ) -> dict[str, Any]:
        state = self._require_state()
        boundary = np.array(
            [transition.boundary_crossing for transition in transitions], dtype=bool
        )
        obstacle = np.zeros(3, dtype=bool)
        for index, position in enumerate(state.pursuer_positions):
            obstacle[index] = any(
                obstacle_surface_clearance(position, item.centre, item.radius)
                < self.config.capture.min_obstacle_clearance
                - self.config.capture.tolerance
                for item in state.obstacles
            )
        teammate = np.zeros(3, dtype=bool)
        distances = pairwise_distances(state.pursuer_positions)
        for left in range(3):
            for right in range(left + 1, 3):
                if distances[left, right] < (
                    self.config.capture.min_pursuer_separation
                    - self.config.capture.tolerance
                ):
                    teammate[left] = teammate[right] = True
        target_distance = np.linalg.norm(
            state.pursuer_positions - state.target_position, axis=1
        )
        target_hard = target_distance < self.config.reward.target_hard_radius
        by_agent = boundary | obstacle | teammate | target_hard
        target_obstacle = any(
            obstacle_surface_clearance(
                state.target_position, item.centre, item.radius
            )
            < 0.0
            for item in state.obstacles
        )
        return {
            "by_agent": by_agent,
            "boundary": boundary,
            "obstacle": obstacle,
            "teammate": teammate,
            "target_hard": target_hard,
            "target_boundary_audit": target_transition.boundary_crossing,
            "target_obstacle_audit": target_obstacle,
        }

    def _minimum_lidar_ranges(self) -> FloatArray:
        state = self._require_state()
        return np.array(
            [
                float(
                    np.min(
                        cast_lidar(
                            state.pursuer_positions[index],
                            state.pursuer_headings[index],
                            self.obstacle_pairs,
                            self.config.workspace,
                            self.config.lidar,
                        )
                    )
                    * self.config.lidar.max_range
                )
                for index in range(3)
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _reward_info(terms: RewardTerms) -> dict[str, Any]:
        return {
            "progress": terms.progress.copy(),
            "safety": terms.safety.copy(),
            "stage": terms.stage.copy(),
            "separation": terms.separation.copy(),
            "target_hit": terms.target_hit.copy(),
            "terminal": terms.terminal.copy(),
            "total": terms.total.copy(),
            "stage_branch": terms.stage_branch,
            "target_fan_area": terms.target_fan_area,
            "target_distance_sum": terms.target_distance_sum,
        }

    def step(
        self, actions: ArrayLike
    ) -> tuple[FloatArray, FloatArray, bool, bool, dict[str, Any]]:
        state = self._require_state()
        if self._episode_done:
            raise RuntimeError("cannot step a completed episode; call reset")
        requested = np.asarray(actions, dtype=np.float64)
        if requested.shape != (3, 2) or not np.all(np.isfinite(requested)):
            raise ValueError("actions must be a finite array with shape (3,2)")
        previous_distances = np.linalg.norm(
            state.pursuer_positions - state.target_position, axis=1
        )
        transitions = [
            integrate_usv(
                state.pursuer_positions[index],
                state.pursuer_velocities[index],
                requested[index],
                self.config.pursuer_limits,
                self.config.workspace,
                self.config.dt,
            )
            for index in range(3)
        ]
        target_action = self.target_policy.action(
            state.target_position,
            state.pursuer_positions,
            self.obstacle_pairs,
            self.config.workspace,
        )
        target_transition = integrate_usv(
            state.target_position,
            state.target_velocity,
            target_action,
            self.config.target_limits,
            self.config.workspace,
            self.config.dt,
        )
        state.pursuer_positions = np.stack(
            [transition.position for transition in transitions]
        )
        state.pursuer_velocities = np.stack(
            [transition.velocity for transition in transitions]
        )
        state.pursuer_headings = np.array(
            [
                self._updated_heading(
                    transition.velocity, state.pursuer_headings[index]
                )
                for index, transition in enumerate(transitions)
            ],
            dtype=np.float64,
        )
        state.target_position = target_transition.position
        state.target_velocity = target_transition.velocity
        state.target_heading = self._updated_heading(
            target_transition.velocity, state.target_heading
        )
        state.step_count += 1

        capture = evaluate_capture(
            state.pursuer_positions,
            state.target_position,
            self.obstacle_pairs,
            self.config.workspace,
            self.config.capture,
        )
        collision = self._collision_events(transitions, target_transition)
        terminated = bool(capture.success or np.any(collision["by_agent"]))
        truncated = bool(
            state.step_count >= self.config.horizon_steps and not terminated
        )
        reward_terms = compute_reward_terms(
            state.pursuer_positions,
            state.pursuer_velocities,
            state.target_position,
            previous_distances,
            self._minimum_lidar_ranges(),
            collision["by_agent"],
            capture.success,
            self.config.pursuer_limits,
            self.config.lidar.max_range,
            self.config.reward,
        )
        self._episode_done = terminated or truncated
        info = self._base_info()
        info.update(
            {
                "requested_action": requested.copy(),
                "executed_action": np.stack(
                    [
                        transition.executed_normalized_action
                        for transition in transitions
                    ]
                ),
                "target_action": target_transition.executed_normalized_action.copy(),
                "capture": capture,
                "capture_clauses": capture.clauses(),
                "collision": collision,
                "reward_terms": self._reward_info(reward_terms),
            }
        )
        return self.observation(), reward_terms.total.copy(), terminated, truncated, info

    def _base_info(self) -> dict[str, Any]:
        state = self._require_state()
        return {
            "spec_version": self.config.spec_version,
            "config_sha256": self.config.canonical_sha256(),
            "scenario_seed": self._seed,
            "target_policy_id": self.target_policy.policy_id,
            "step_count": state.step_count,
        }

    def snapshot(self) -> EnvironmentSnapshot:
        return EnvironmentSnapshot(
            state=self._require_state().copy(),
            scenario_rng_state=deepcopy(self._rng.bit_generator.state),
            target_policy_state=deepcopy(self.target_policy.snapshot()),
            seed=self._seed,
            episode_done=self._episode_done,
        )

    def restore(self, snapshot: EnvironmentSnapshot) -> None:
        self.state = snapshot.state.copy()
        self._rng.bit_generator.state = deepcopy(snapshot.scenario_rng_state)
        self.target_policy.restore(deepcopy(snapshot.target_policy_state))
        self._seed = int(snapshot.seed)
        self._episode_done = bool(snapshot.episode_done)
