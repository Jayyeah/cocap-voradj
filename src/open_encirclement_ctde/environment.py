"""The frozen ``corrected_roundup_v1`` task contract.

This module is a clean adapter/corrected implementation.  The immutable
upstream snapshots remain under ``vendor/upstream`` and are never imported by
this environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
from gymnasium import spaces

from .geometry import angular_metrics, area_containment_residual, point_in_triangle, triangle_area


@dataclass(frozen=True)
class RoundupSpec:
    contract: str = "corrected_roundup_v1"
    num_hunters: int = 3
    arena_length: float = 2.0
    num_obstacles: int = 3
    dt: float = 0.5
    hunter_v_max: float = 0.1
    hunter_a_max: float = 0.04
    target_v_max: float = 0.12
    target_a_max: float = 0.05
    lidar_range: float = 0.2
    num_lasers: int = 16
    capture_radius: float = 0.3
    track_distance_limit: float = 0.75
    max_steps: int = 100
    target_wall_margin: float = 0.25
    target_wall_gain: float = 2.0

    @property
    def obs_dim(self) -> int:
        return 26

    @property
    def action_dim(self) -> int:
        return 2


class CorrectedRoundupEnv:
    """Three learned hunters and one scripted target in a 2-D arena."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        *,
        seed: int = 1,
        spec: RoundupSpec | None = None,
        num_obstacles: int | None = None,
        stationary_target: bool = False,
        fixed_hunter_positions: Iterable[Iterable[float]] | None = None,
        fixed_target_position: Iterable[float] | None = None,
        fixed_obstacles: Iterable[tuple[Iterable[float], float]] | None = None,
    ) -> None:
        base = spec or RoundupSpec()
        if num_obstacles is not None:
            base = RoundupSpec(**{**asdict(base), "num_obstacles": int(num_obstacles)})
        self.spec = base
        self.stationary_target = bool(stationary_target)
        self._initial_seed = int(seed)
        self.rng = np.random.default_rng(self._initial_seed)
        self._fixed_hunter_positions = None if fixed_hunter_positions is None else np.asarray(fixed_hunter_positions, dtype=np.float64)
        self._fixed_target_position = None if fixed_target_position is None else np.asarray(fixed_target_position, dtype=np.float64)
        self._fixed_obstacles = fixed_obstacles
        if self._fixed_hunter_positions is not None and self._fixed_hunter_positions.shape != (3, 2):
            raise ValueError("fixed_hunter_positions must have shape (3, 2)")
        if self._fixed_target_position is not None and self._fixed_target_position.shape != (2,):
            raise ValueError("fixed_target_position must have shape (2,)")

        low = np.full((2,), -self.spec.hunter_a_max, dtype=np.float32)
        high = np.full((2,), self.spec.hunter_a_max, dtype=np.float32)
        self.action_space = [spaces.Box(low=low, high=high, dtype=np.float32) for _ in range(3)]
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.spec.obs_dim,), dtype=np.float32)
            for _ in range(3)
        ]
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(3 * self.spec.obs_dim,), dtype=np.float32)
            for _ in range(3)
        ]
        self.num_agent = self.spec.num_hunters
        self._build_obstacles()
        self.positions = np.zeros((4, 2), dtype=np.float64)
        self.velocities = np.zeros((4, 2), dtype=np.float64)
        self.lasers = np.full((4, self.spec.num_lasers), self.spec.lidar_range, dtype=np.float64)
        self.step_count = 0
        self.episode_index = 0
        self.last_info: dict[str, Any] = {}
        self._trajectory: list[np.ndarray] = []

    def _build_obstacles(self) -> None:
        if self._fixed_obstacles is not None:
            parsed = [(np.asarray(pos, dtype=np.float64), float(radius)) for pos, radius in self._fixed_obstacles]
            self.obstacle_positions = np.stack([p for p, _ in parsed]) if parsed else np.empty((0, 2), dtype=np.float64)
            self.obstacle_radii = np.asarray([r for _, r in parsed], dtype=np.float64)
            return
        n = self.spec.num_obstacles
        self.obstacle_positions = self.rng.uniform(0.45, self.spec.arena_length - 0.55, size=(n, 2))
        # Preserve the upstream RNG draw order: position, angle, radius per obstacle.
        # Angles have no physical effect because obstacles are static.
        if n:
            positions = []
            radii = []
            local = np.random.default_rng(self._initial_seed)
            for _ in range(n):
                positions.append(local.uniform(0.45, self.spec.arena_length - 0.55, size=2))
                local.uniform(0.0, 2.0 * np.pi)
                radii.append(local.uniform(0.1, 0.15))
            self.obstacle_positions = np.asarray(positions, dtype=np.float64)
            self.obstacle_radii = np.asarray(radii, dtype=np.float64)
            self.rng = local
        else:
            self.obstacle_radii = np.empty((0,), dtype=np.float64)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._initial_seed = int(seed)
            self.rng = np.random.default_rng(self._initial_seed)
            self._build_obstacles()
            self.episode_index = 0
        if self._fixed_hunter_positions is None:
            self.positions[:3] = self.rng.uniform(0.1, 0.4, size=(3, 2))
        else:
            self.positions[:3] = self._fixed_hunter_positions
        self.positions[3] = np.array([0.5, 1.75], dtype=np.float64) if self._fixed_target_position is None else self._fixed_target_position
        self.velocities.fill(0.0)
        self.step_count = 0
        self.episode_index += 1
        collisions, collision_types = self._update_lasers_and_collisions(apply_response=True)
        self._trajectory = [self.positions.copy()]
        obs = self._observations()
        info = self._metrics(
            success=False,
            collisions=collisions,
            collision_types=collision_types,
            proposed_actions=np.zeros((3, 2), dtype=np.float64),
            executed_actions=np.zeros((3, 2), dtype=np.float64),
            target_action=np.zeros(2, dtype=np.float64),
            progress=np.zeros(3, dtype=np.float64),
        )
        self.last_info = info
        return obs, info

    def scripted_target_action(self) -> np.ndarray:
        if self.stationary_target:
            return np.zeros(2, dtype=np.float64)
        target = self.positions[3]
        dists = np.linalg.norm(self.positions[:3] - target, axis=1)
        nearest = self.positions[int(np.argmin(dists))]
        flee = target - nearest
        norm = float(np.linalg.norm(flee))
        if norm < 1e-6:
            flee = self.rng.uniform(-1.0, 1.0, size=2)
            norm = float(np.linalg.norm(flee)) + 1e-6
        flee /= norm
        length = self.spec.arena_length
        margin = self.spec.target_wall_margin
        wall_force = np.array(
            [
                max(0.0, 1.0 - target[0] / margin) - max(0.0, 1.0 - (length - target[0]) / margin),
                max(0.0, 1.0 - target[1] / margin) - max(0.0, 1.0 - (length - target[1]) / margin),
            ],
            dtype=np.float64,
        )
        combined = flee + self.spec.target_wall_gain * wall_force
        cnorm = float(np.linalg.norm(combined))
        if cnorm > 1e-6:
            combined /= cnorm
        return np.clip(combined * self.spec.target_a_max, -self.spec.target_a_max, self.spec.target_a_max)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        proposed = np.asarray(actions, dtype=np.float64)
        if proposed.shape != (3, 2):
            raise ValueError(f"expected hunter actions shape (3,2), got {proposed.shape}")
        if not np.isfinite(proposed).all():
            raise ValueError("non-finite hunter action")
        executed = np.clip(proposed, -self.spec.hunter_a_max, self.spec.hunter_a_max)
        target_action = self.scripted_target_action()
        full_actions = np.vstack((executed, target_action))
        previous_distances = np.linalg.norm(self.positions[:3] - self.positions[3], axis=1)

        for idx in range(4):
            self.velocities[idx] += full_actions[idx] * self.spec.dt
            vmax = self.spec.hunter_v_max if idx < 3 else self.spec.target_v_max
            speed = float(np.linalg.norm(self.velocities[idx]))
            if speed > vmax:
                self.velocities[idx] *= vmax / speed
            self.positions[idx] += self.velocities[idx] * self.spec.dt

        collisions, collision_types = self._update_lasers_and_collisions(apply_response=True)
        distances = np.linalg.norm(self.positions[:3] - self.positions[3], axis=1)
        progress = previous_distances - distances
        contained = point_in_triangle(self.positions[3], self.positions[:3])
        success = bool(contained and np.all(distances <= self.spec.capture_radius))
        rewards = self._hunter_rewards(collisions, distances, previous_distances, contained, success)
        self.step_count += 1
        terminated = success
        truncated = bool(not terminated and self.step_count >= self.spec.max_steps)
        self._trajectory.append(self.positions.copy())
        info = self._metrics(
            success=success,
            collisions=collisions,
            collision_types=collision_types,
            proposed_actions=proposed,
            executed_actions=executed,
            target_action=target_action,
            progress=progress,
        )
        info["terminated"] = terminated
        info["truncated"] = truncated
        self.last_info = info
        return self._observations(), rewards.astype(np.float32), terminated, truncated, info

    def _hunter_rewards(
        self,
        collisions: np.ndarray,
        distances: np.ndarray,
        previous_distances: np.ndarray,
        contained: bool,
        success: bool,
    ) -> np.ndarray:
        rewards = np.zeros(3, dtype=np.float64)
        mu1, mu2, mu3, mu4 = 0.7, 0.4, 0.01, 5.0
        for idx in range(3):
            velocity = self.velocities[idx]
            direction = self.positions[3] - self.positions[idx]
            speed = float(np.linalg.norm(velocity))
            distance = float(distances[idx])
            cos_v_d = float(np.dot(velocity, direction) / (speed * distance + 1e-3))
            rewards[idx] += mu1 * abs(2.0 * speed / self.spec.hunter_v_max) * cos_v_d
            if collisions[idx]:
                safe = -10.0
            else:
                safe = (float(np.min(self.lasers[idx])) - self.spec.lidar_range - 0.1) / self.spec.lidar_range
            rewards[idx] += mu2 * safe

        sum_distance = float(np.sum(distances))
        if not contained and sum_distance >= self.spec.track_distance_limit and np.all(distances >= self.spec.capture_radius):
            stage = -sum_distance / float(np.max(distances))
            rewards[:] += mu3 * stage
        elif not contained:
            residual = max(area_containment_residual(self.positions[3], self.positions[:3]), 0.0)
            stage = -(1.0 / 3.0) * np.log(residual + 1.0)
            rewards[:] += mu3 * stage
        elif np.any(distances > self.spec.capture_radius):
            stage = np.exp(float(np.sum(previous_distances) - sum_distance) / (3.0 * self.spec.hunter_v_max))
            rewards[:] += mu3 * stage
        if success:
            rewards[:] += mu4 * 10.0
        return rewards

    def _observations(self) -> np.ndarray:
        observations = []
        target = self.positions[3]
        distance_scale = np.linalg.norm(np.full(2, self.spec.arena_length, dtype=np.float64))
        for idx in range(3):
            own = [
                self.positions[idx, 0] / self.spec.arena_length,
                self.positions[idx, 1] / self.spec.arena_length,
                self.velocities[idx, 0] / self.spec.hunter_v_max,
                self.velocities[idx, 1] / self.spec.hunter_v_max,
            ]
            teammates: list[float] = []
            for other in range(3):
                if other != idx:
                    teammates.extend((self.positions[other, 0] / self.spec.arena_length, self.positions[other, 1] / self.spec.arena_length))
            delta = target - self.positions[idx]
            target_features = [float(np.linalg.norm(delta) / distance_scale), float(np.arctan2(delta[1], delta[0]))]
            obs = np.asarray(own + teammates + self.lasers[idx].tolist() + target_features, dtype=np.float32)
            if obs.shape != (self.spec.obs_dim,):
                raise RuntimeError(f"observation contract drifted: {obs.shape}")
            observations.append(obs)
        result = np.stack(observations)
        if not np.isfinite(result).all():
            raise RuntimeError("environment produced non-finite observation")
        return result

    def _update_lasers_and_collisions(self, *, apply_response: bool) -> tuple[np.ndarray, list[list[str]]]:
        collisions = np.zeros(4, dtype=bool)
        collision_types: list[list[str]] = [[] for _ in range(4)]
        lasers = np.full((4, self.spec.num_lasers), self.spec.lidar_range, dtype=np.float64)
        for idx, pos in enumerate(self.positions):
            outside = bool(np.any(pos < 0.0) or np.any(pos > self.spec.arena_length))
            if outside:
                collisions[idx] = True
                collision_types[idx].append("boundary")
            for obstacle_pos, radius in zip(self.obstacle_positions, self.obstacle_radii):
                if float(np.linalg.norm(pos - obstacle_pos)) < float(radius):
                    collisions[idx] = True
                    collision_types[idx].append("obstacle")
                for ray_idx, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, self.spec.num_lasers, endpoint=False)):
                    lasers[idx, ray_idx] = min(
                        lasers[idx, ray_idx],
                        self._ray_circle_distance(pos, angle, obstacle_pos, float(radius), self.spec.lidar_range),
                    )
            for ray_idx, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, self.spec.num_lasers, endpoint=False)):
                lasers[idx, ray_idx] = min(
                    lasers[idx, ray_idx],
                    self._ray_wall_distance(pos, angle, self.spec.arena_length, self.spec.lidar_range),
                )
            if collisions[idx]:
                lasers[idx].fill(0.0)
                if apply_response:
                    self.velocities[idx].fill(0.0)
        self.lasers = lasers
        return collisions, collision_types

    @staticmethod
    def _ray_circle_distance(start: np.ndarray, angle: float, center: np.ndarray, radius: float, maximum: float) -> float:
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        offset = start - center
        b = 2.0 * float(np.dot(offset, direction))
        c = float(np.dot(offset, offset) - radius * radius)
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return maximum
        root = np.sqrt(disc)
        for distance in ((-b - root) / 2.0, (-b + root) / 2.0):
            if 0.0 <= distance <= maximum:
                return float(distance)
        return maximum

    @staticmethod
    def _ray_wall_distance(start: np.ndarray, angle: float, bound: float, maximum: float) -> float:
        result = maximum
        cos_theta, sin_theta = float(np.cos(angle)), float(np.sin(angle))
        if sin_theta > 0:
            result = min(result, abs((bound - start[1]) / sin_theta))
        elif sin_theta < 0:
            result = min(result, abs(start[1] / -sin_theta))
        if cos_theta > 0:
            result = min(result, abs((bound - start[0]) / cos_theta))
        elif cos_theta < 0:
            result = min(result, abs(start[0] / -cos_theta))
        return float(result)

    def _metrics(
        self,
        *,
        success: bool,
        collisions: np.ndarray,
        collision_types: list[list[str]],
        proposed_actions: np.ndarray,
        executed_actions: np.ndarray,
        target_action: np.ndarray,
        progress: np.ndarray,
    ) -> dict[str, Any]:
        distances = np.linalg.norm(self.positions[:3] - self.positions[3], axis=1)
        largest_gap, pairwise = angular_metrics(self.positions[:3], self.positions[3])
        contained = point_in_triangle(self.positions[3], self.positions[:3])
        saturation = np.isclose(np.abs(executed_actions), self.spec.hunter_a_max, atol=1e-8)
        return {
            "contract": self.spec.contract,
            "episode_index": self.episode_index,
            "step": self.step_count,
            "success": bool(success),
            "hull_contained": bool(contained),
            "all_within_capture_radius": bool(np.all(distances <= self.spec.capture_radius)),
            "distances": distances.tolist(),
            "min_target_distance": float(np.min(distances)),
            "mean_target_distance": float(np.mean(distances)),
            "progress": np.asarray(progress, dtype=np.float64).tolist(),
            "largest_angular_gap": largest_gap,
            "pairwise_angle_separation": pairwise,
            "triangle_area": triangle_area(self.positions[:3]),
            "collisions": np.asarray(collisions, dtype=bool).tolist(),
            "collision_types": collision_types,
            "collision": bool(np.any(collisions[:3])),
            "proposed_actions": np.asarray(proposed_actions, dtype=np.float64).tolist(),
            "executed_actions": np.asarray(executed_actions, dtype=np.float64).tolist(),
            "target_action": np.asarray(target_action, dtype=np.float64).tolist(),
            "action_saturation_fraction": float(np.mean(saturation)),
            "hunter_speeds": np.linalg.norm(self.velocities[:3], axis=1).tolist(),
            "target_speed": float(np.linalg.norm(self.velocities[3])),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "spec": asdict(self.spec),
            "initial_seed": self._initial_seed,
            "rng_state": self.rng.bit_generator.state,
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "lasers": self.lasers.copy(),
            "obstacle_positions": self.obstacle_positions.copy(),
            "obstacle_radii": self.obstacle_radii.copy(),
            "step_count": self.step_count,
            "episode_index": self.episode_index,
            "stationary_target": self.stationary_target,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["spec"] != asdict(self.spec):
            raise ValueError("environment spec mismatch during resume")
        self._initial_seed = int(state["initial_seed"])
        self.rng.bit_generator.state = state["rng_state"]
        self.positions = np.asarray(state["positions"], dtype=np.float64).copy()
        self.velocities = np.asarray(state["velocities"], dtype=np.float64).copy()
        self.lasers = np.asarray(state["lasers"], dtype=np.float64).copy()
        self.obstacle_positions = np.asarray(state["obstacle_positions"], dtype=np.float64).copy()
        self.obstacle_radii = np.asarray(state["obstacle_radii"], dtype=np.float64).copy()
        self.step_count = int(state["step_count"])
        self.episode_index = int(state["episode_index"])
        self.stationary_target = bool(state["stationary_target"])

    def render(self) -> np.ndarray:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.set_xlim(0, self.spec.arena_length)
        ax.set_ylim(0, self.spec.arena_length)
        ax.set_aspect("equal")
        for pos, radius in zip(self.obstacle_positions, self.obstacle_radii):
            ax.add_patch(plt.Circle(pos, radius, color="#555b66", alpha=0.65))
        if self._trajectory:
            trajectory = np.asarray(self._trajectory)
            colors = ("#1f77b4", "#2ca02c", "#9467bd", "#d62728")
            for idx in range(4):
                ax.plot(trajectory[:, idx, 0], trajectory[:, idx, 1], color=colors[idx], alpha=0.5)
        ax.scatter(self.positions[:3, 0], self.positions[:3, 1], c=("#1f77b4", "#2ca02c", "#9467bd"), s=70)
        ax.scatter(self.positions[3, 0], self.positions[3, 1], c="#d62728", marker="*", s=140)
        ax.add_patch(plt.Circle(self.positions[3], self.spec.capture_radius, fill=False, linestyle=":", color="#d62728"))
        ax.set_title(f"{self.spec.contract} step={self.step_count}")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return frame
