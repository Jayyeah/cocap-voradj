
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cocap_voradj.config import ConfigManager
from cocap_voradj.dynamics.evader import Evader
from cocap_voradj.dynamics.pursuer import Pursuer
from cocap_voradj.dynamics.continuous_action import (
    AccelerationActionAdapter,
    AccelerationAngularVelocityActionAdapter,
)

TWO_PI = 2.0 * np.pi


class Obstacle:
    def __init__(self, x: float, y: float, r: float):
        self.x = float(x)
        self.y = float(y)
        self.r = float(r)


@dataclass
class StepResult:
    observations: List[Optional[Dict[str, np.ndarray]]]
    rewards: np.ndarray
    dones: List[bool]
    infos: List[Dict[str, Any]]


class CoCapEnv:
    """CoCap marine environment with capture and coverage semantics.

    This class owns the robot dynamics, APF-compatible evader observations,
    CoCap observation model, rewards, hard-boundary handling, coverage, and
    capture semantics.
    """

    def __init__(self, config: Dict[str, Any], task: str, seed: int = 0):
        if task not in {"coverage", "encirclement", "voradj"}:
            raise ValueError("task must be coverage, encirclement, or voradj")
        self.config = config
        self.task = task
        self.rng = np.random.RandomState(seed)
        self.seed = int(seed)
        self.env_cfg = config.get("env", {})
        pursuer_cfg = config.get("pursuer", {}) or {}
        raw_action_mode = config.get(
            "action_mode",
            self.env_cfg.get("action_mode", pursuer_cfg.get("action_mode", "unicycle_discrete")),
        )
        raw_action_mode = str(raw_action_mode).strip().lower()
        if raw_action_mode in {"unicycle", "unicycle_discrete"}:
            self.action_mode = "unicycle_discrete"
        elif raw_action_mode in {
            "acceleration_2d_world",
            "continuous_acceleration_2d_world",
        }:
            self.action_mode = "acceleration_2d_world"
        elif raw_action_mode in {
            "acceleration_2d_body",
            "continuous_acceleration_2d_body",
            # Read old isolated configs without preserving their old
            # velocity-command semantics.
            "velocity_2d_body",
            "continuous_velocity_2d_body",
        }:
            self.action_mode = "acceleration_2d_body"
        elif raw_action_mode in {
            "acceleration_angular_velocity_body",
            "continuous_acceleration_angular_velocity_body",
            "continuous_aw",
            "aw",
        }:
            self.action_mode = "acceleration_angular_velocity_body"
        else:
            raise ValueError(f"unsupported pursuer action_mode: {raw_action_mode}")
        self.continuous_world_action = self.action_mode == "acceleration_2d_world"
        self.continuous_action = self.action_mode == "acceleration_2d_body" or self.continuous_world_action
        self.continuous_aw_action = self.action_mode == "acceleration_angular_velocity_body"
        self.continuous_control = self.continuous_action or self.continuous_aw_action
        self.v_max = float(config.get("v_max", pursuer_cfg.get("max_speed", 3.0)))
        if not np.isfinite(self.v_max) or self.v_max <= 0.0:
            raise ValueError("v_max must be finite and positive")
        self.acceleration_adapter: Optional[AccelerationActionAdapter] = None
        self.action_adapter = None
        if self.continuous_action:
            decision_dt = float(
                config.get(
                    "decision_dt",
                    self.env_cfg.get("decision_dt", float(0.05 * 10)),
                )
            )
            action_cfg = config.get("action", {}) or {}
            a_max = float(
                action_cfg.get(
                    "a_max",
                    config.get("a_max", pursuer_cfg.get("a_max", 0.8)),
                )
            )
            self.acceleration_adapter = AccelerationActionAdapter(
                a_max=a_max,
                decision_dt=decision_dt,
            )
            self.action_adapter = self.acceleration_adapter
        elif self.continuous_aw_action:
            decision_dt = float(
                config.get(
                    "decision_dt",
                    self.env_cfg.get("decision_dt", float(0.05 * 10)),
                )
            )
            self.action_adapter = AccelerationAngularVelocityActionAdapter(
                a_max=float(config.get("a_max", pursuer_cfg.get("a_max", 0.4))),
                w_max=float(config.get("w_max", pursuer_cfg.get("w_max", np.pi / 6))),
                decision_dt=decision_dt,
            )
        self.per_cfg = config.get("perception", {})
        self.reward_cfg = config.get("reward", {})
        self.width = float(self.env_cfg.get("width", 55.0))
        self.height = float(self.env_cfg.get("height", self.width))
        self.inner_x_min = float(self.env_cfg.get("inner_x_min", 0.0))
        self.inner_x_max = float(self.env_cfg.get("inner_x_max", self.width))
        self.inner_y_min = float(self.env_cfg.get("inner_y_min", 0.0))
        self.inner_y_max = float(self.env_cfg.get("inner_y_max", self.height))
        self.num_pursuers = int(self.env_cfg.get("num_pursuers", 4))
        self.num_evaders = int(self.env_cfg.get("num_evaders", 0 if task == "coverage" else 1))
        self.num_obstacles = int(self.env_cfg.get("num_obstacles", 1))
        self.episode_max_length = int(self.env_cfg.get("episode_max_length", 3000))
        self.enforce_hard_boundary = bool(self.env_cfg.get("enforce_hard_boundary", True))
        self.boundary_collision_death = bool(self.env_cfg.get("boundary_collision_death", True))
        self.boundary_penalty = float(self.env_cfg.get("boundary_penalty", -5.0))
        self.distribution_converged_grace_steps = int(
            self.reward_cfg.get(
                "distribution_converged_grace_steps",
                config.get("iqn", {}).get("distribution_converged_grace_steps", 0),
            )
        )
        self.pursuers: List[Pursuer] = []
        self.evaders: List[Evader] = []
        self.obstacles: List[Obstacle] = []
        self.episode_step = 0
        self.total_steps = 0
        self.distribution_hold_steps = 0
        self.last_distribution_metrics: Dict[str, Any] = {}
        self._geometry_first_achieved = False
        self.last_capture_events: List[Dict[str, Any]] = []
        self.last_reward_terms: Dict[str, float] = {}
        self.last_boundary_penalty_count = 0
        self.last_boundary_penalty_sum = 0.0
        self.last_boundary_proximity_penalty_count = 0
        self.last_boundary_proximity_penalty_sum = 0.0
        self.last_emergency_proximity_penalty_count = 0
        self.last_emergency_proximity_penalty_sum = 0.0
        self.last_out_of_bounds_pursuers: List[int] = []
        self.last_out_of_bounds_evaders: List[int] = []
        self.episode_out_of_bounds_step_count = 0
        self.episode_out_of_bounds_pursuer_steps = 0
        self.episode_out_of_bounds_evader_steps = 0
        self.episode_out_of_bounds_pursuer_ids: List[int] = []
        self.episode_out_of_bounds_evader_ids: List[int] = []
        self.archived_evader_trajectories: List[List[List[float]]] = []
        dynamics_cfg = self.config.get("dynamics", {}) or {}
        default_collision_semantics = (
            "synchronized_swept_v1"
            if bool(dynamics_cfg.get("collision_check_each_substep", False))
            else "legacy_end_step"
        )
        raw_collision_semantics = str(
            self.env_cfg.get("collision_semantics", default_collision_semantics)
        ).strip().lower()
        collision_aliases = {
            "legacy": "legacy_end_step",
            "end_step": "legacy_end_step",
            "legacy_end_step": "legacy_end_step",
            "swept": "synchronized_swept_v1",
            "swept_v1": "synchronized_swept_v1",
            "synchronized_swept_v1": "synchronized_swept_v1",
        }
        if raw_collision_semantics not in collision_aliases:
            raise ValueError(
                "env.collision_semantics must be legacy_end_step or synchronized_swept_v1"
            )
        self.collision_semantics = collision_aliases[raw_collision_semantics]
        self._collision_step_active = False
        self.last_collision_events: List[Dict[str, Any]] = []
        self.episode_collision_events: List[Dict[str, Any]] = []
        self.episode_collision_type_counts: Dict[str, int] = {}

    @property
    def action_size(self) -> int:
        if self.continuous_control:
            return 2
        if self.pursuers:
            return len(self.pursuers[0].action_list)
        return 9

    def _rand_pos(self, margin: float) -> np.ndarray:
        return self.rng.uniform([margin, margin], [self.width - margin, self.height - margin]).astype(float)

    def _clamp_pos(self, pos: np.ndarray, padding: float = 2.0) -> np.ndarray:
        low = np.array([padding, padding], dtype=float)
        high = np.array([self.width - padding, self.height - padding], dtype=float)
        return np.clip(np.asarray(pos, dtype=float), low, high)

    def _sample_cluster_positions(self, center: np.ndarray, count: int, radius: float, min_sep: float) -> List[np.ndarray]:
        for _batch_attempt in range(300):
            positions: List[np.ndarray] = []
            for _ in range(count):
                r = radius * np.sqrt(self.rng.uniform(0.0, 1.0))
                theta = self.rng.uniform(0.0, TWO_PI)
                candidate = self._clamp_pos(
                    np.asarray(center, dtype=float) + r * np.array([np.cos(theta), np.sin(theta)]),
                    padding=2.0,
                )
                if all(np.linalg.norm(candidate - pos) >= min_sep for pos in positions):
                    positions.append(candidate)
                else:
                    break
            if len(positions) == count:
                return positions
        if count <= 1:
            return [self._clamp_pos(center, padding=2.0)]
        base_angle = self.rng.uniform(0.0, TWO_PI)
        min_ring_radius = min_sep / max(2.0 * np.sin(np.pi / count), 1e-6)
        ring_radius = min(radius, max(0.75 * radius, 1.05 * min_ring_radius))
        return [
            self._clamp_pos(
                np.asarray(center, dtype=float)
                + ring_radius * np.array([
                    np.cos(base_angle + TWO_PI * idx / count),
                    np.sin(base_angle + TWO_PI * idx / count),
                ]),
                padding=2.0,
            )
            for idx in range(count)
        ]

    def _valid_position(self, pos: np.ndarray, radius: float, existing: List[Tuple[np.ndarray, float]], min_sep: float) -> bool:
        if pos[0] - radius <= 0 or pos[0] + radius >= self.width or pos[1] - radius <= 0 or pos[1] + radius >= self.height:
            return False
        for center, other_r in existing:
            if np.linalg.norm(pos - center) < radius + other_r + min_sep:
                return False
        return True

    def _reset_robot(self, robot, pos: np.ndarray, theta: Optional[float] = None) -> None:
        robot.start = np.asarray(pos, dtype=float)
        if theta is None and self.continuous_control and getattr(robot, "robot_type", None) == "pursuer":
            yaw_cfg = self.config.get("yaw", {}) or self.env_cfg.get("yaw", {}) or {}
            yaw_init = str(yaw_cfg.get("init", "aligned_world_axis")).strip().lower()
            if yaw_init in {"aligned_world_axis", "world_axis", "zero"}:
                theta = 0.0
        robot.init_theta = float(theta if theta is not None else self.rng.uniform(0.0, TWO_PI))
        robot.init_speed = float(self.env_cfg.get("init_speed", 0.0))
        robot.collision = False
        robot.deactivated = False
        robot.boundary_collision = False
        robot.collision_types = set()
        robot.last_action_diagnostics = {}
        if hasattr(robot, "captured_evaderId_list"):
            robot.captured_evaderId_list.clear()
            robot.is_current_target_captured = False
            robot.is_pursuing = False
        robot.reset_state(np.zeros(2, dtype=float))

    def reset(self, initial_pursuer_positions: Optional[List[List[float]]] = None, initial_pursuer_active: Optional[List[bool]] = None):
        self.episode_step = 0
        self.distribution_hold_steps = 0
        self.last_distribution_metrics = {}
        self._geometry_first_achieved = False
        self.last_reward_terms = {}
        self.last_boundary_penalty_count = 0
        self.last_boundary_penalty_sum = 0.0
        self.last_boundary_proximity_penalty_count = 0
        self.last_boundary_proximity_penalty_sum = 0.0
        self.last_emergency_proximity_penalty_count = 0
        self.last_emergency_proximity_penalty_sum = 0.0
        self.last_out_of_bounds_pursuers = []
        self.last_out_of_bounds_evaders = []
        self.episode_out_of_bounds_step_count = 0
        self.episode_out_of_bounds_pursuer_steps = 0
        self.episode_out_of_bounds_evader_steps = 0
        self.episode_out_of_bounds_pursuer_ids = []
        self.episode_out_of_bounds_evader_ids = []
        self.last_capture_events = []
        self.last_collision_events = []
        self.episode_collision_events = []
        self.episode_collision_type_counts = {}
        self._stationary_capture_counters: Dict[str, int] = {}
        self.archived_evader_trajectories.clear()
        self.pursuers = [Pursuer(i) for i in range(self.num_pursuers)]
        if self.continuous_control:
            # In the explicit acceleration contract, v_max is an environment
            # physical limit. Make the root canonical value authoritative over
            # duplicated legacy pursuer.max_speed config fields.
            for pursuer in self.pursuers:
                pursuer.max_speed = self.v_max
            dynamics_cfg = self.config.get("dynamics", {}) or {}
            if "linear_drag_coefficient" in dynamics_cfg:
                drag = float(dynamics_cfg["linear_drag_coefficient"])
                for pursuer in self.pursuers:
                    pursuer.coefficient_water_resistance = drag
        self.evaders = [Evader(i) for i in range(self.num_evaders)]
        for robot in [*self.pursuers, *self.evaders]:
            robot.perception.range = float(self.per_cfg.get("range", 20.0))
            robot.perception.angle = float(self.per_cfg.get("angle", TWO_PI))
        for pursuer in self.pursuers:
            pursuer.perception.max_obstacle_num = int(self.per_cfg.get("max_obstacle_num", 5))
            pursuer.perception.max_pursuer_num = int(self.per_cfg.get("max_pursuer_num", 12))
            pursuer.perception.max_evader_num = int(self.per_cfg.get("max_evader_num", 8))
        # Keep APF evader observations byte-for-byte compatible with APF's
        # 59-dimensional APF input: self4 + 5 static*3 + (5 pursuers + 5 evaders)*4.
        for evader in self.evaders:
            evader.perception.max_obstacle_num = 5
            evader.perception.max_pursuer_num = 5
            evader.perception.max_evader_num = 5
        self.obstacles = []

        existing: List[Tuple[np.ndarray, float]] = []
        obs_range = self.env_cfg.get("obs_r_range", [2.0, 3.0])
        for _ in range(self.num_obstacles):
            placed = False
            for _attempt in range(2000):
                radius = float(self.rng.uniform(float(obs_range[0]), float(obs_range[1])))
                pos = self._rand_pos(max(5.0, radius + 2.0))
                if self._valid_position(pos, radius, existing, 4.0):
                    self.obstacles.append(Obstacle(pos[0], pos[1], radius))
                    existing.append((pos, radius))
                    placed = True
                    break
            if not placed:
                radius = float(self.rng.uniform(float(obs_range[0]), float(obs_range[1])))
                pos = self._rand_pos(3.0)
                self.obstacles.append(Obstacle(pos[0], pos[1], radius))
                existing.append((pos, radius))

        if initial_pursuer_positions is not None:
            positions = [np.asarray(p, dtype=float) for p in initial_pursuer_positions[: self.num_pursuers]]
        else:
            positions = []
            spawn_mode = str(self.env_cfg.get("pursuer_spawn_mode", "map_random"))
            if spawn_mode in {"inner_random_cluster", "post_capture_cluster"}:
                cluster_radius = float(self.env_cfg.get("spawn_cluster_radius", 10.0))
                min_sep = float(self.env_cfg.get("pursuer_spawn_min_sep", 5.0))
                center_margin = max(float(self.env_cfg.get("spawn_edge_margin", 10.0)), cluster_radius + min_sep)
                center = self._rand_pos(center_margin)
                positions = self._sample_cluster_positions(center, self.num_pursuers, cluster_radius, min_sep)
        while len(positions) < self.num_pursuers:
            positions.append(self._rand_pos(float(self.env_cfg.get("spawn_edge_margin", 10.0))))

        if initial_pursuer_active is None:
            active_mask = [True] * self.num_pursuers
        else:
            active_mask = [bool(value) for value in initial_pursuer_active[: self.num_pursuers]]
            while len(active_mask) < self.num_pursuers:
                active_mask.append(False)

        existing_for_pursuers = existing.copy()
        active_positions: List[np.ndarray] = []
        min_sep = float(self.env_cfg.get("pursuer_spawn_min_sep", 5.0))
        for i in range(self.num_pursuers):
            pos = np.asarray(positions[i], dtype=float)
            if active_mask[i] and not self._valid_position(pos, self.pursuers[i].r, existing_for_pursuers, min_sep):
                placed = False
                for _attempt in range(4000):
                    pos = self._rand_pos(float(self.env_cfg.get("spawn_edge_margin", 4.0)))
                    if self._valid_position(pos, self.pursuers[i].r, existing_for_pursuers, min_sep):
                        placed = True
                        break
                if not placed:
                    for _attempt in range(4000):
                        pos = self._rand_pos(float(self.env_cfg.get("spawn_edge_margin", 4.0)))
                        if self._valid_position(pos, self.pursuers[i].r, existing_for_pursuers, max(0.5, min_sep * 0.3)):
                            break
            self._reset_robot(self.pursuers[i], pos)
            if active_mask[i]:
                active_positions.append(pos)
                existing_for_pursuers.append((pos, self.pursuers[i].r))
            else:
                self.pursuers[i].deactivated = True

        existing_all = existing_for_pursuers.copy()
        evader_min_sep = float(self.env_cfg.get("evader_spawn_min_sep", 8.0))
        min_pe = float(self.env_cfg.get("min_pursuer_evader_init_dis", 13.0))
        for evader in self.evaders:
            placed = False
            for _attempt in range(8000):
                pos = self._rand_pos(float(self.env_cfg.get("spawn_edge_margin", 10.0)))
                if self._valid_position(pos, evader.r, existing_all, evader_min_sep):
                    if all(np.linalg.norm(pos - p) >= min_pe for p in active_positions):
                        placed = True
                        break
            if not placed:
                relaxed_pe = max(3.0, min_pe * 0.5)
                for _attempt in range(8000):
                    pos = self._rand_pos(float(self.env_cfg.get("spawn_edge_margin", 10.0)))
                    if self._valid_position(pos, evader.r, existing_all, max(1.0, evader_min_sep * 0.5)):
                        if all(np.linalg.norm(pos - p) >= relaxed_pe for p in active_positions):
                            placed = True
                            break
            self._reset_robot(evader, pos)
            existing_all.append((pos, evader.r))

        return self.get_observations()

    def _position(self, robot) -> np.ndarray:
        return np.array([float(robot.x), float(robot.y)], dtype=float)

    def _touch_boundary(self, robot) -> bool:
        r = float(getattr(robot, "r", 0.0))
        return bool(robot.x - r <= 0.0 or robot.x + r >= self.width or robot.y - r <= 0.0 or robot.y + r >= self.height)

    def _center_out_of_bounds(self, robot) -> bool:
        return bool(robot.x < 0.0 or robot.x > self.width or robot.y < 0.0 or robot.y > self.height)

    def _begin_collision_step(self) -> None:
        """Start a synchronous trace interval for collision resolution."""
        self._collision_step_active = True
        self.last_collision_events = []
        for robot in [*self.pursuers, *self.evaders]:
            robot._collision_was_active = bool(not robot.deactivated)
            robot._collision_trace = [self._position(robot)]

    def _record_collision_substep(self, robot) -> None:
        if not self._collision_step_active:
            return
        trace = getattr(robot, "_collision_trace", None)
        if trace is None:
            robot._collision_trace = [self._position(robot)]
        else:
            trace.append(self._position(robot))

    def _end_collision_step(self) -> None:
        self._collision_step_active = False

    @staticmethod
    def _segment_contact_fraction(
        relative_start: np.ndarray,
        relative_end: np.ndarray,
        contact_radius: float,
        tolerance: float = 1e-9,
    ) -> Optional[float]:
        """Return the first circular contact fraction on a relative segment."""
        start = np.asarray(relative_start, dtype=float)
        end = np.asarray(relative_end, dtype=float)
        radius = float(contact_radius) + float(tolerance)
        c = float(np.dot(start, start) - radius * radius)
        if c <= 0.0:
            return 0.0
        delta = end - start
        a = float(np.dot(delta, delta))
        if a <= np.finfo(float).eps:
            return None
        b = 2.0 * float(np.dot(start, delta))
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return None
        root = float(np.sqrt(max(discriminant, 0.0)))
        for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
            if -tolerance <= value <= 1.0 + tolerance:
                return float(np.clip(value, 0.0, 1.0))
        return None

    def _trace_points(self, robot) -> List[np.ndarray]:
        raw = getattr(robot, "_collision_trace", None)
        if not raw:
            return [self._position(robot)]
        return [np.asarray(point, dtype=float) for point in raw]

    def _trace_pair_contact(self, first, second) -> Optional[float]:
        first_trace = self._trace_points(first)
        second_trace = self._trace_points(second)
        first_stationary = len(first_trace) == 1 and bool(
            getattr(first, "_collision_was_active", not first.deactivated)
        )
        second_stationary = len(second_trace) == 1 and bool(
            getattr(second, "_collision_was_active", not second.deactivated)
        )
        if len(first_trace) == 1 and len(second_trace) == 1:
            relative = first_trace[0] - second_trace[0]
            return self._segment_contact_fraction(
                relative,
                relative,
                float(first.r) + float(second.r),
            )
        if first_stationary:
            intervals = len(second_trace) - 1
        elif second_stationary:
            intervals = len(first_trace) - 1
        else:
            intervals = min(len(first_trace), len(second_trace)) - 1
        for index in range(max(intervals, 0)):
            first_start = first_trace[0] if first_stationary else first_trace[index]
            first_end = first_trace[0] if first_stationary else first_trace[index + 1]
            second_start = second_trace[0] if second_stationary else second_trace[index]
            second_end = second_trace[0] if second_stationary else second_trace[index + 1]
            fraction = self._segment_contact_fraction(
                first_start - second_start,
                first_end - second_end,
                float(first.r) + float(second.r),
            )
            if fraction is not None:
                return float(index) + fraction
        return None

    def _trace_obstacle_contact(self, robot, obstacle: Obstacle) -> Optional[float]:
        trace = self._trace_points(robot)
        center = np.asarray([obstacle.x, obstacle.y], dtype=float)
        radius = float(robot.r) + float(obstacle.r)
        if len(trace) == 1:
            relative = trace[0] - center
            return self._segment_contact_fraction(relative, relative, radius)
        for index in range(len(trace) - 1):
            fraction = self._segment_contact_fraction(
                trace[index] - center,
                trace[index + 1] - center,
                radius,
            )
            if fraction is not None:
                return float(index) + fraction
        return None

    def _register_collision_type(self, robot, collision_type: str) -> None:
        types = set(getattr(robot, "collision_types", set()) or set())
        types.add(str(collision_type))
        robot.collision_types = types
        robot.collision = True

    def _record_collision_event(
        self,
        collision_type: str,
        *,
        pursuer_ids: Optional[List[int]] = None,
        evader_ids: Optional[List[int]] = None,
        obstacle_id: Optional[int] = None,
        contact_substep: Optional[float] = None,
    ) -> None:
        event = {
            "type": str(collision_type),
            "step": int(self.episode_step),
            "pursuer_ids": sorted(int(value) for value in (pursuer_ids or [])),
            "evader_ids": sorted(int(value) for value in (evader_ids or [])),
            "obstacle_id": None if obstacle_id is None else int(obstacle_id),
            "contact_substep": (
                None if contact_substep is None else float(contact_substep)
            ),
        }
        if event in self.last_collision_events:
            return
        self.last_collision_events.append(event)
        self.episode_collision_events.append(dict(event))
        key = str(collision_type)
        self.episode_collision_type_counts[key] = int(
            self.episode_collision_type_counts.get(key, 0)
        ) + 1

    def _record_soft_boundary_out_of_bounds(self) -> None:
        if self.enforce_hard_boundary or self.boundary_collision_death:
            self.last_out_of_bounds_pursuers = []
            self.last_out_of_bounds_evaders = []
            return
        pursuer_ids = [int(p.id) for p in self.pursuers if not p.deactivated and self._center_out_of_bounds(p)]
        evader_ids = [int(e.id) for e in self.evaders if not e.deactivated and self._center_out_of_bounds(e)]
        self.last_out_of_bounds_pursuers = pursuer_ids
        self.last_out_of_bounds_evaders = evader_ids
        if pursuer_ids or evader_ids:
            self.episode_out_of_bounds_step_count += 1
            self.episode_out_of_bounds_pursuer_steps += len(pursuer_ids)
            self.episode_out_of_bounds_evader_steps += len(evader_ids)
            self.episode_out_of_bounds_pursuer_ids = sorted(set(self.episode_out_of_bounds_pursuer_ids).union(pursuer_ids))
            self.episode_out_of_bounds_evader_ids = sorted(set(self.episode_out_of_bounds_evader_ids).union(evader_ids))

    def _clip_and_kill_boundary(self, robot) -> None:
        if not (self.enforce_hard_boundary or self.boundary_collision_death):
            return
        zone_cfg = self.config.get("zone_demo", {}) or {}
        if (
            bool(zone_cfg.get("enabled", False))
            and bool(zone_cfg.get("evader_ignore_boundary_death", True))
            and isinstance(robot, Evader)
        ):
            return
        if not self._touch_boundary(robot):
            return
        r = float(getattr(robot, "r", 0.0))
        robot.x = float(np.clip(robot.x, r, self.width - r))
        robot.y = float(np.clip(robot.y, r, self.height - r))
        robot.velocity = np.zeros(2, dtype=float)
        robot.speed = 0.0
        self._register_collision_type(robot, "boundary")
        robot.boundary_collision = True
        if isinstance(robot, Pursuer):
            self._record_collision_event("boundary", pursuer_ids=[int(robot.id)])
        else:
            self._record_collision_event("boundary", evader_ids=[int(robot.id)])
        if self.boundary_collision_death:
            robot.deactivated = True

    def _boundary_penalties(self) -> np.ndarray:
        penalties = np.zeros(len(self.pursuers), dtype=float)
        if self.enforce_hard_boundary or self.boundary_collision_death:
            return penalties
        for i, pursuer in enumerate(self.pursuers):
            if self._center_out_of_bounds(pursuer):
                penalties[i] += self.boundary_penalty
        return penalties

    def _boundary_clearance(self, robot) -> float:
        r = float(getattr(robot, "r", 0.0))
        return float(min(robot.x - r, self.width - r - robot.x, robot.y - r, self.height - r - robot.y))

    def _proximity_penalty_value(
        self,
        clearance: float,
        warning_distance: float,
        base_penalty: float,
        mode: str,
        critical_distance: float,
        penalty_cap: Optional[float],
    ) -> float:
        if warning_distance <= 0.0 or clearance >= warning_distance:
            return 0.0
        normalized_mode = str(mode or "flat").strip().lower()
        if normalized_mode not in {"graded", "quadratic", "distance"}:
            return float(base_penalty)
        cap = abs(float(base_penalty)) if penalty_cap is None else abs(float(penalty_cap))
        denom = max(float(warning_distance) - float(critical_distance), 1e-9)
        frac = float(np.clip((float(warning_distance) - float(clearance)) / denom, 0.0, 1.0))
        return -cap * frac * frac

    def _emergency_proximity_penalties(self) -> np.ndarray:
        penalties = np.zeros(len(self.pursuers), dtype=float)
        warning_distance = float(
            self.reward_cfg.get(
                "emergency_warning_distance",
                self.reward_cfg.get("d_safe", 4.0),
            )
        )
        base_penalty = float(self.env_cfg.get("emergency_penalty", -5.0))
        mode = str(self.reward_cfg.get("emergency_penalty_mode", self.env_cfg.get("emergency_penalty_mode", "flat")))
        critical_distance = float(self.reward_cfg.get("emergency_critical_distance", self.env_cfg.get("emergency_critical_distance", 0.5)))
        cap_cfg = self.reward_cfg.get("emergency_penalty_cap", self.env_cfg.get("emergency_penalty_cap", None))
        penalty_cap = None if cap_cfg is None else float(cap_cfg)
        for i, pursuer in enumerate(self.pursuers):
            if pursuer.deactivated or pursuer.collision:
                continue
            penalties[i] = self._proximity_penalty_value(
                clearance=self._minimum_clearance(i),
                warning_distance=warning_distance,
                base_penalty=base_penalty,
                mode=mode,
                critical_distance=critical_distance,
                penalty_cap=penalty_cap,
            )
        self.last_emergency_proximity_penalty_count = int(np.count_nonzero(penalties))
        self.last_emergency_proximity_penalty_sum = float(np.sum(penalties))
        return penalties

    def _boundary_proximity_penalties(self) -> np.ndarray:
        penalties = np.zeros(len(self.pursuers), dtype=float)
        enabled = bool(
            self.env_cfg.get(
                "boundary_proximity_penalty_enabled",
                self.reward_cfg.get("boundary_proximity_penalty_enabled", False),
            )
        )
        if not enabled:
            return penalties
        distance = float(
            self.env_cfg.get(
                "boundary_proximity_distance",
                self.reward_cfg.get("boundary_proximity_distance", self.reward_cfg.get("d_safe", 4.0)),
            )
        )
        if distance <= 0.0:
            return penalties
        penalty = float(
            self.env_cfg.get(
                "boundary_proximity_penalty",
                self.reward_cfg.get("boundary_proximity_penalty", self.env_cfg.get("emergency_penalty", -5.0)),
            )
        )
        mode = str(self.reward_cfg.get("boundary_proximity_penalty_mode", self.env_cfg.get("boundary_proximity_penalty_mode", "flat")))
        critical_distance = float(
            self.env_cfg.get(
                "boundary_proximity_critical_distance",
                self.reward_cfg.get("boundary_proximity_critical_distance", 0.5),
            )
        )
        cap_cfg = self.env_cfg.get("boundary_proximity_penalty_cap", self.reward_cfg.get("boundary_proximity_penalty_cap", None))
        penalty_cap = None if cap_cfg is None else float(cap_cfg)
        for i, pursuer in enumerate(self.pursuers):
            if pursuer.deactivated or pursuer.collision:
                continue
            penalties[i] += self._proximity_penalty_value(
                clearance=self._boundary_clearance(pursuer),
                warning_distance=distance,
                base_penalty=penalty,
                mode=mode,
                critical_distance=critical_distance,
                penalty_cap=penalty_cap,
            )
        return penalties

    def _move_robot(self, robot, action: Any) -> None:
        if robot.deactivated or action is None:
            return
        if self.continuous_aw_action and getattr(robot, "robot_type", None) == "pursuer":
            if self.action_adapter is None:
                raise RuntimeError("continuous (a,w) adapter is not initialized")
            command = self.action_adapter.validate(action)
            speed_before = float(robot.speed)
            previous_diagnostics = getattr(robot, "last_action_diagnostics", {}) or {}
            previous_acceleration = float(previous_diagnostics.get("actual_acceleration", 0.0))

            def substep_checks() -> None:
                self._clip_and_kill_boundary(robot)
                self._record_collision_substep(robot)

            speed_limited = robot.update_state_acceleration_angular_velocity_body(
                command,
                np.zeros(2, dtype=float),
                substep_callback=substep_checks,
            )
            decision_dt = float(robot.dt * max(int(robot.N), 1))
            execution_interrupted = bool(robot.deactivated)
            if execution_interrupted:
                actual_acceleration = 0.0
                jerk = 0.0
            else:
                actual_acceleration = float(
                    abs(float(robot.speed) - speed_before) / max(decision_dt, np.finfo(float).eps)
                )
                jerk = float(
                    abs(actual_acceleration - previous_acceleration)
                    / max(decision_dt, np.finfo(float).eps)
                )
            robot.action_history.append(command.astype(float).tolist())
            robot.last_action_diagnostics = {
                "commanded_acceleration": abs(float(command[0])),
                "validated_acceleration": abs(float(command[0])),
                "commanded_angular_velocity": float(command[1]),
                "validated_angular_velocity": float(command[1]),
                "action_rejected": False,
                "validation_delta": 0.0,
                "speed_before": speed_before,
                "speed_after": float(robot.speed),
                "speed_limited": bool(speed_limited),
                "execution_interrupted": execution_interrupted,
                "actual_acceleration": actual_acceleration,
                "jerk": jerk,
                "validation_rate": 0.0,
            }
            robot.trajectory.append([robot.x, robot.y, robot.theta, robot.speed, robot.velocity[0], robot.velocity[1]])
            return
        if self.continuous_world_action and getattr(robot, "robot_type", None) == "pursuer":
            if self.acceleration_adapter is None:
                raise RuntimeError("continuous acceleration adapter is not initialized")
            acceleration_world = self.acceleration_adapter.validate(action)
            speed_before = float(robot.speed)
            previous_world = np.asarray(robot.velocity, dtype=float)
            previous_diagnostics = getattr(robot, "last_action_diagnostics", {}) or {}
            previous_acceleration = float(previous_diagnostics.get("actual_acceleration", 0.0))

            def substep_checks() -> None:
                self._clip_and_kill_boundary(robot)
                self._record_collision_substep(robot)
                if (
                    self.collision_semantics == "legacy_end_step"
                    and not robot.deactivated
                ):
                    self._refresh_collisions()

            speed_limited = robot.update_state_acceleration_world(
                acceleration_world,
                np.zeros(2, dtype=float),
                substep_callback=substep_checks,
            )
            decision_dt = float(robot.dt * max(int(robot.N), 1))
            execution_interrupted = bool(robot.deactivated)
            if execution_interrupted:
                actual_acceleration = 0.0
                jerk = 0.0
            else:
                actual_delta = np.asarray(robot.velocity, dtype=float) - previous_world
                actual_acceleration = float(np.linalg.norm(actual_delta) / max(decision_dt, np.finfo(float).eps))
                jerk = float(abs(actual_acceleration - previous_acceleration) / max(decision_dt, np.finfo(float).eps))
            robot.action_history.append(acceleration_world.astype(float).tolist())
            robot.last_action_diagnostics = {
                "commanded_acceleration": float(np.linalg.norm(acceleration_world)),
                "validated_acceleration": float(np.linalg.norm(acceleration_world)),
                "action_rejected": False,
                "validation_delta": 0.0,
                "speed_before": speed_before,
                "speed_after": float(robot.speed),
                "speed_limited": bool(speed_limited),
                "execution_interrupted": execution_interrupted,
                "actual_acceleration": actual_acceleration,
                "jerk": jerk,
                "validation_rate": 0.0,
            }
            robot.trajectory.append([robot.x, robot.y, robot.theta, robot.speed, robot.velocity[0], robot.velocity[1]])
            return
        elif self.continuous_action and getattr(robot, "robot_type", None) == "pursuer":
            if self.acceleration_adapter is None:
                raise RuntimeError("continuous acceleration adapter is not initialized")
            acceleration_body = self.acceleration_adapter.validate(action)
            speed_before = float(robot.speed)
            previous_world = np.asarray(robot.velocity, dtype=float)
            previous_diagnostics = getattr(robot, "last_action_diagnostics", {}) or {}
            previous_acceleration = float(previous_diagnostics.get("actual_acceleration", 0.0))

            def substep_checks() -> None:
                self._clip_and_kill_boundary(robot)
                self._record_collision_substep(robot)
                if (
                    self.collision_semantics == "legacy_end_step"
                    and not robot.deactivated
                ):
                    self._refresh_collisions()

            speed_limited = robot.update_state_acceleration_body(
                acceleration_body,
                np.zeros(2, dtype=float),
                substep_callback=substep_checks,
            )
            yaw_cfg = self.config.get("yaw", {}) or self.env_cfg.get("yaw", {}) or {}
            yaw_mode = str(yaw_cfg.get("mode", "hold")).strip().lower()
            if yaw_mode in {"velocity_heading", "velocity"} and not robot.deactivated:
                speed = float(robot.speed)
                epsilon = float(yaw_cfg.get("speed_epsilon", 0.05))
                if speed > epsilon:
                    robot.theta = float(np.arctan2(robot.velocity[1], robot.velocity[0])) % TWO_PI
            decision_dt = float(robot.dt * max(int(robot.N), 1))
            execution_interrupted = bool(robot.deactivated)
            if execution_interrupted:
                # Collision/boundary termination may reset velocity to zero. That
                # terminal state change is not a policy acceleration command.
                actual_acceleration = 0.0
                jerk = 0.0
            else:
                actual_delta = np.asarray(robot.velocity, dtype=float) - previous_world
                actual_acceleration = float(np.linalg.norm(actual_delta) / max(decision_dt, np.finfo(float).eps))
                jerk = float(abs(actual_acceleration - previous_acceleration) / max(decision_dt, np.finfo(float).eps))
            robot.action_history.append(acceleration_body.astype(float).tolist())
            robot.last_action_diagnostics = {
                "commanded_acceleration": float(np.linalg.norm(acceleration_body)),
                "validated_acceleration": float(np.linalg.norm(acceleration_body)),
                "action_rejected": False,
                "validation_delta": 0.0,
                "speed_before": speed_before,
                "speed_after": float(robot.speed),
                "speed_limited": bool(speed_limited),
                "execution_interrupted": execution_interrupted,
                "actual_acceleration": actual_acceleration,
                "jerk": jerk,
                "validation_rate": 0.0,
            }
            robot.trajectory.append([robot.x, robot.y, robot.theta, robot.speed, robot.velocity[0], robot.velocity[1]])
            return
        robot.action_history.append(int(action))
        for _ in range(robot.N):
            robot.update_state(int(action), np.zeros(2, dtype=float))
            self._clip_and_kill_boundary(robot)
            self._record_collision_substep(robot)
            if robot.deactivated:
                break
        robot.trajectory.append([robot.x, robot.y, robot.theta, robot.speed, robot.velocity[0], robot.velocity[1]])

    def _detect(self, pursuer: Pursuer, x: float, y: float, r: float) -> bool:
        if bool(self.per_cfg.get("global_evader_visibility", False)):
            return True
        try:
            return bool(pursuer.check_detection(x, y, r))
        except Exception:
            return float(np.linalg.norm(self._position(pursuer) - np.array([x, y]))) <= float(self.per_cfg.get("range", 20.0)) + r

    def _robot_frame(self, robot, vec: np.ndarray, is_vector: bool) -> np.ndarray:
        return robot.project_to_robot_frame(np.asarray(vec, dtype=float), is_vector=is_vector)

    def _local_coverage_features(self, idx: int) -> List[float]:
        pursuer = self.pursuers[idx]
        sense = max(float(self.per_cfg.get("range", 20.0)), 1e-6)
        offsets_1d = np.linspace(-sense, sense, int(self.per_cfg.get("local_coverage_grid_size", 9)))
        ox, oy = np.meshgrid(offsets_1d, offsets_1d)
        offsets = np.stack([ox.ravel(), oy.ravel()], axis=1)
        offsets = offsets[np.linalg.norm(offsets, axis=1) <= sense]
        if offsets.size == 0:
            return [0.0, 0.0, 0.0]
        self_pos = self._position(pursuer)
        visible = [idx]
        for j, other in enumerate(self.pursuers):
            if j == idx or other.deactivated:
                continue
            if pursuer.check_detection(other.x, other.y, other.r):
                visible.append(j)
        if len(visible) == 1:
            return [0.0, 0.0, 0.0]
        sample_points = self_pos + offsets
        visible_positions = np.asarray([self._position(self.pursuers[j]) for j in visible])
        dist_sq = np.sum((sample_points[:, None, :] - visible_positions[None, :, :]) ** 2, axis=2)
        owners = np.argmin(dist_sq, axis=1)
        owned = sample_points[owners == 0]
        if len(owned) == 0:
            center_vec = np.zeros(2)
            owned_fraction = 0.0
        else:
            center_vec = np.mean(owned, axis=0) - self_pos
            owned_fraction = float(len(owned) / max(len(sample_points), 1))
        center_r = self._robot_frame(pursuer, center_vec, True) / sense
        expected = 1.0 / max(len(visible), 1)
        area_error = float(np.clip(owned_fraction / max(expected, 1e-9) - 1.0, -1.0, 1.0))
        return [float(center_r[0]), float(center_r[1]), area_error]

    def _pack_agent_obs(self, idx: int) -> Optional[Dict[str, np.ndarray]]:
        pursuer = self.pursuers[idx]
        if pursuer.deactivated:
            return None
        max_p = int(self.per_cfg.get("max_pursuer_num", 12))
        max_e = int(self.per_cfg.get("max_evader_num", 8))
        max_o = int(self.per_cfg.get("max_obstacle_num", 5))
        abs_vel = self._robot_frame(pursuer, pursuer.velocity, True)
        if self.obstacles:
            dists = [np.linalg.norm(self._position(pursuer) - np.array([o.x, o.y])) for o in self.obstacles]
            min_obs = min([d for d in dists if d <= float(self.per_cfg.get("range", 20.0))] or [float(self.per_cfg.get("range", 20.0))])
        else:
            min_obs = float(self.per_cfg.get("range", 20.0))
        boundary_clearance = min(pursuer.x, pursuer.y, self.width - pursuer.x, self.height - pursuer.y)
        self_feat = list(abs_vel) + [float(min_obs), float(boundary_clearance)] + self._local_coverage_features(idx)
        pursuer_feats: List[List[float]] = []
        for j, other in enumerate(self.pursuers):
            if j == idx or other.deactivated:
                continue
            if not pursuer.check_detection(other.x, other.y, other.r):
                continue
            pos_r = self._robot_frame(pursuer, self._position(other), False)
            vel_r = self._robot_frame(pursuer, other.velocity, True)
            dist = float(np.linalg.norm(pos_r))
            ang = float(np.arctan2(pos_r[1], pos_r[0]))
            pursuer_feats.append([pos_r[0], pos_r[1], vel_r[0], vel_r[1], dist, ang, 0.0])
        evader_feats: List[List[float]] = []
        for evader in self.evaders:
            if evader.deactivated:
                continue
            if not self._detect(pursuer, evader.x, evader.y, evader.r):
                continue
            pos_r = self._robot_frame(pursuer, self._position(evader), False)
            vel_r = self._robot_frame(pursuer, evader.velocity, True)
            dist = float(np.linalg.norm(pos_r))
            ang = float(np.arctan2(pos_r[1], pos_r[0]))
            heading = float(np.arctan2(vel_r[1], vel_r[0])) if np.linalg.norm(vel_r) > 1e-9 else 0.0
            evader_feats.append([pos_r[0], pos_r[1], vel_r[0], vel_r[1], dist, ang, heading])
        obstacle_feats: List[List[float]] = []
        for obs in self.obstacles:
            if not pursuer.check_detection(obs.x, obs.y, obs.r):
                continue
            pos_r = self._robot_frame(pursuer, np.array([obs.x, obs.y]), False)
            dist = float(np.linalg.norm(pos_r))
            ang = float(np.arctan2(pos_r[1], pos_r[0]))
            obstacle_feats.append([pos_r[0], pos_r[1], obs.r, dist, ang])
        masks = [True]
        masks += [True] * min(len(pursuer_feats), max_p) + [False] * max(0, max_p - len(pursuer_feats))
        masks += [True] * min(len(evader_feats), max_e) + [False] * max(0, max_e - len(evader_feats))
        masks += [True] * min(len(obstacle_feats), max_o) + [False] * max(0, max_o - len(obstacle_feats))
        types = [0] + [1] * max_p + [2] * max_e + [3] * max_o
        return {
            "self": np.asarray(self_feat[:7], dtype=np.float32),
            "pursuers": self._pad(pursuer_feats, max_p, 7),
            "evaders": self._pad(evader_feats, max_e, 7),
            "obstacles": self._pad(obstacle_feats, max_o, 5),
            "masks": np.asarray(masks, dtype=bool),
            "types": np.asarray(types, dtype=np.int64),
        }

    @staticmethod
    def _pad(values: List[List[float]], count: int, dim: int) -> np.ndarray:
        arr = np.zeros((count, dim), dtype=np.float32)
        for i, value in enumerate(values[:count]):
            arr[i, :dim] = np.asarray(value[:dim], dtype=np.float32)
        return arr

    def get_observations(self) -> List[Optional[Dict[str, np.ndarray]]]:
        return [self._pack_agent_obs(i) for i in range(len(self.pursuers))]

    def get_evader_observations_for_apf(self) -> List[Optional[List[float]]]:
        observations = []
        for evader in self.evaders:
            if evader.deactivated:
                observations.append(None)
            else:
                obs, _ = evader.perception_output(self.obstacles, self.pursuers, self.evaders, True)
                observations.append(obs)
        return observations

    def _minimum_clearance(self, i: int) -> float:
        p = self.pursuers[i]
        pos = self._position(p)
        clearances = []
        for j, other in enumerate(self.pursuers):
            if i != j and not other.deactivated:
                clearances.append(np.linalg.norm(pos - self._position(other)) - p.r - other.r)
        for evader in self.evaders:
            if not evader.deactivated:
                clearances.append(np.linalg.norm(pos - self._position(evader)) - p.r - evader.r)
        for obs in self.obstacles:
            clearances.append(np.linalg.norm(pos - np.array([obs.x, obs.y])) - p.r - obs.r)
        return float(min(clearances) if clearances else np.inf)

    def _refresh_collisions_legacy(self) -> None:
        """Historical sequential endpoint detector, retained for paired audits."""
        for i, p in enumerate(self.pursuers):
            if p.deactivated:
                continue
            peer_ids = [
                int(other.id)
                for j, other in enumerate(self.pursuers)
                if i != j
                and not other.deactivated
                and np.linalg.norm(self._position(p) - self._position(other))
                < float(p.r) + float(other.r)
            ]
            evader_ids = [
                int(evader.id)
                for evader in self.evaders
                if not evader.deactivated
                and np.linalg.norm(self._position(p) - self._position(evader))
                < float(p.r) + float(evader.r)
            ]
            obstacle_ids = [
                int(getattr(obstacle, "id", obstacle_index))
                for obstacle_index, obstacle in enumerate(self.obstacles)
                if np.linalg.norm(
                    self._position(p) - np.array([obstacle.x, obstacle.y], dtype=float)
                )
                < float(p.r) + float(obstacle.r)
            ]
            if peer_ids:
                self._register_collision_type(p, "agent_agent")
                for peer_id in peer_ids:
                    self._record_collision_event(
                        "agent_agent", pursuer_ids=[int(p.id), peer_id]
                    )
            if evader_ids:
                self._register_collision_type(p, "evader_contact")
                for evader_id in evader_ids:
                    self._record_collision_event(
                        "evader_contact",
                        pursuer_ids=[int(p.id)],
                        evader_ids=[evader_id],
                    )
            if obstacle_ids:
                self._register_collision_type(p, "obstacle")
                for obstacle_id in obstacle_ids:
                    self._record_collision_event(
                        "obstacle",
                        pursuer_ids=[int(p.id)],
                        obstacle_id=obstacle_id,
                    )
            if peer_ids or evader_ids or obstacle_ids:
                p.deactivated = True
        for e in self.evaders:
            if e.deactivated:
                continue
            for obstacle_index, obs in enumerate(self.obstacles):
                if np.linalg.norm(self._position(e) - np.array([obs.x, obs.y])) < float(e.r) + float(obs.r):
                    self._register_collision_type(e, "obstacle")
                    self._record_collision_event(
                        "obstacle",
                        evader_ids=[int(e.id)],
                        obstacle_id=int(getattr(obs, "id", obstacle_index)),
                    )
                    e.deactivated = True
            self._clip_and_kill_boundary(e)

    def _refresh_collisions_synchronized_swept(self) -> None:
        """Detect on immutable synchronized traces, then deactivate atomically."""
        pursuer_types: Dict[int, set[str]] = {}
        evader_types: Dict[int, set[str]] = {}

        def was_active(robot) -> bool:
            return bool(getattr(robot, "_collision_was_active", not robot.deactivated))

        def mark_pursuer(index: int, collision_type: str) -> None:
            pursuer_types.setdefault(int(index), set()).add(str(collision_type))

        def mark_evader(index: int, collision_type: str) -> None:
            evader_types.setdefault(int(index), set()).add(str(collision_type))

        for i, first in enumerate(self.pursuers):
            if not was_active(first):
                continue
            for j in range(i + 1, len(self.pursuers)):
                second = self.pursuers[j]
                if not was_active(second):
                    continue
                contact = self._trace_pair_contact(first, second)
                if contact is None:
                    continue
                mark_pursuer(i, "agent_agent")
                mark_pursuer(j, "agent_agent")
                self._record_collision_event(
                    "agent_agent",
                    pursuer_ids=[int(first.id), int(second.id)],
                    contact_substep=contact,
                )

        for i, pursuer in enumerate(self.pursuers):
            if not was_active(pursuer):
                continue
            for j, evader in enumerate(self.evaders):
                if not was_active(evader):
                    continue
                contact = self._trace_pair_contact(pursuer, evader)
                if contact is None:
                    continue
                # Preserve the historical task contract: evader contact removes
                # the pursuer, but does not itself remove the active target.
                mark_pursuer(i, "evader_contact")
                self._record_collision_event(
                    "evader_contact",
                    pursuer_ids=[int(pursuer.id)],
                    evader_ids=[int(evader.id)],
                    contact_substep=contact,
                )
            for obstacle_index, obstacle in enumerate(self.obstacles):
                contact = self._trace_obstacle_contact(pursuer, obstacle)
                if contact is None:
                    continue
                mark_pursuer(i, "obstacle")
                self._record_collision_event(
                    "obstacle",
                    pursuer_ids=[int(pursuer.id)],
                    obstacle_id=int(getattr(obstacle, "id", obstacle_index)),
                    contact_substep=contact,
                )

        for i, evader in enumerate(self.evaders):
            if not was_active(evader):
                continue
            for obstacle_index, obstacle in enumerate(self.obstacles):
                contact = self._trace_obstacle_contact(evader, obstacle)
                if contact is None:
                    continue
                mark_evader(i, "obstacle")
                self._record_collision_event(
                    "obstacle",
                    evader_ids=[int(evader.id)],
                    obstacle_id=int(getattr(obstacle, "id", obstacle_index)),
                    contact_substep=contact,
                )

        for index, collision_types in pursuer_types.items():
            pursuer = self.pursuers[index]
            for collision_type in sorted(collision_types):
                self._register_collision_type(pursuer, collision_type)
            pursuer.deactivated = True
        for index, collision_types in evader_types.items():
            evader = self.evaders[index]
            for collision_type in sorted(collision_types):
                self._register_collision_type(evader, collision_type)
            evader.deactivated = True

    def _refresh_collisions(self) -> None:
        if self.collision_semantics == "synchronized_swept_v1":
            self._refresh_collisions_synchronized_swept()
        else:
            self._refresh_collisions_legacy()

    def _collision_episode_fields(self) -> Dict[str, Any]:
        pursuer_collision = bool(any(p.collision for p in self.pursuers))
        evader_collision = bool(any(e.collision for e in self.evaders))
        agent_agent_ids = sorted({
            int(value)
            for event in self.episode_collision_events
            if event.get("type") == "agent_agent"
            for value in event.get("pursuer_ids", [])
        })
        return {
            "collision_semantics": str(self.collision_semantics),
            "collision_event": bool(pursuer_collision or evader_collision),
            "pursuer_collision_event": pursuer_collision,
            "evader_collision_event": evader_collision,
            "collision_type_counts": {
                str(key): int(value)
                for key, value in sorted(self.episode_collision_type_counts.items())
            },
            "collision_events": [dict(event) for event in self.episode_collision_events],
            "last_collision_events": [dict(event) for event in self.last_collision_events],
            "agent_agent_collision_event": bool(
                self.episode_collision_type_counts.get("agent_agent", 0)
            ),
            "agent_agent_collision_count": int(
                self.episode_collision_type_counts.get("agent_agent", 0)
            ),
            "agent_agent_collision_pursuers": agent_agent_ids,
            "obstacle_collision_event": bool(
                self.episode_collision_type_counts.get("obstacle", 0)
            ),
            "evader_contact_collision_event": bool(
                self.episode_collision_type_counts.get("evader_contact", 0)
            ),
            "pursuer_collision_types": {
                str(int(p.id)): sorted(str(value) for value in getattr(p, "collision_types", set()))
                for p in self.pursuers
                if getattr(p, "collision_types", set())
            },
            "evader_collision_types": {
                str(int(e.id)): sorted(str(value) for value in getattr(e, "collision_types", set()))
                for e in self.evaders
                if getattr(e, "collision_types", set())
            },
        }

    def _loose_capture_events(self) -> List[Dict[str, Any]]:
        events = []
        k_required = int(self.reward_cfg.get("k_required", 3))
        capture_distance = float(self.env_cfg.get("capture_distance", self.reward_cfg.get("r_e", 8.0)))
        max_gap_threshold = float(self.reward_cfg.get("max_angle_gap", np.pi))
        max_ratio = float(self.reward_cfg.get("max_angle_ratio", 3.0))
        stationary_enabled = bool(self.reward_cfg.get("capture_stationary_enabled", False))
        v_evader_static = float(self.reward_cfg.get("capture_stationary_speed_threshold", 0.2))
        stationary_hold = int(self.reward_cfg.get("capture_stationary_hold_steps", 10))
        stationary_min_pursuers = int(self.reward_cfg.get("capture_stationary_min_pursuers", 2))
        for j, evader in enumerate(self.evaders):
            if evader.deactivated:
                continue
            participants, angles, distances = [], [], []
            epos = self._position(evader)
            for i, p in enumerate(self.pursuers):
                if p.deactivated:
                    continue
                vec = self._position(p) - epos
                d = float(np.linalg.norm(vec))
                if d <= capture_distance:
                    participants.append(i)
                    angles.append(float(np.arctan2(vec[1], vec[0]) % TWO_PI))
                    distances.append(d)
            # Loose multi-angle capture
            if len(participants) >= k_required:
                sorted_angles = sorted(angles)
                gaps = [(sorted_angles[(m + 1) % len(sorted_angles)] - sorted_angles[m]) % TWO_PI for m in range(len(sorted_angles))]
                max_gap = float(max(gaps))
                min_gap = float(min(gaps))
                if max_gap <= max_gap_threshold and max_gap <= min_gap * max_ratio:
                    events.append({"evader_id": j, "participants": participants, "angles": gaps, "distances": distances, "capture_type": "loose"})
            # 支线3: stationary capture (wall/corner/obstacle)
            if stationary_enabled and not events:
                evader_static = bool(float(evader.speed) <= v_evader_static)
                stationary_condition = bool(evader_static and len(participants) >= stationary_min_pursuers)
                key = f"evader_{j}"
                if stationary_condition:
                    self._stationary_capture_counters[key] = self._stationary_capture_counters.get(key, 0) + 1
                else:
                    self._stationary_capture_counters[key] = 0
                if self._stationary_capture_counters.get(key, 0) >= stationary_hold:
                    events.append({"evader_id": j, "participants": participants, "angles": [], "distances": distances, "capture_type": "stationary"})
        return events

    def _coverage_potentials(self, positions: np.ndarray) -> np.ndarray:
        active = [i for i, p in enumerate(self.pursuers) if not p.deactivated]
        values = np.zeros(len(self.pursuers), dtype=float)
        if not active:
            return values
        grid_n = int(self.reward_cfg.get("coverage_grid_size", 30))
        xs = np.linspace(self.inner_x_min, self.inner_x_max, grid_n)
        ys = np.linspace(self.inner_y_min, self.inner_y_max, grid_n)
        pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        apos = positions[active]
        dist_sq = np.sum((pts[:, None, :] - apos[None, :, :]) ** 2, axis=2)
        owners = np.argmin(dist_sq, axis=1)
        fractions = np.bincount(owners, minlength=len(active)).astype(float) / max(len(pts), 1)
        expected = 1.0 / max(len(active), 1)
        diag = np.hypot(self.inner_x_max - self.inner_x_min, self.inner_y_max - self.inner_y_min)
        for local, idx in enumerate(active):
            owned = pts[owners == local]
            center_error = 1.0 if len(owned) == 0 else float(np.linalg.norm(apos[local] - owned.mean(axis=0)) / max(diag, 1e-6))
            area_error = min(abs(fractions[local] / max(expected, 1e-9) - 1.0), float(self.reward_cfg.get("coverage_area_error_clip", 1.0)))
            values[idx] = float(self.reward_cfg.get("coverage_center_weight", 1.0)) * center_error + float(self.reward_cfg.get("coverage_area_weight", 0.5)) * area_error
        return values

    def check_distribution_converged(self, update_hold: bool = False, strict: bool = False) -> bool:
        active = [i for i, p in enumerate(self.pursuers) if not p.deactivated]
        positions = np.asarray([self._position(self.pursuers[i]) for i in active], dtype=float) if active else np.zeros((0, 2))
        if len(active) < 2:
            converged = False
            area_cv = np.inf
            center_ok_ratio = 0.0
            inside_ratio = 0.0
            geometry_strict = False
            speed_strict = False
            max_speed_val = float("inf")
            mean_speed_val = float("inf")
        else:
            grid_n = int(self.reward_cfg.get("distribution_voronoi_grid_size", 40))
            xs = np.linspace(self.inner_x_min, self.inner_x_max, grid_n)
            ys = np.linspace(self.inner_y_min, self.inner_y_max, grid_n)
            pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
            dist_sq = np.sum((pts[:, None, :] - positions[None, :, :]) ** 2, axis=2)
            owners = np.argmin(dist_sq, axis=1)
            counts = np.bincount(owners, minlength=len(active)).astype(float)
            areas = counts / max(np.sum(counts), 1.0)
            area_cv = float(np.std(areas) / max(np.mean(areas), 1e-9))
            diag = np.hypot(self.inner_x_max - self.inner_x_min, self.inner_y_max - self.inner_y_min)
            center_dists = []
            for local, pos in enumerate(positions):
                owned = pts[owners == local]
                center_dists.append(np.inf if len(owned) == 0 else float(np.linalg.norm(pos - owned.mean(axis=0)) / max(diag, 1e-6)))
            center_threshold = float(self.reward_cfg.get("strict_center_distance", 0.08) if strict else self.reward_cfg.get("distribution_voronoi_center_distance_threshold", 0.12))
            center_ok_ratio_threshold = float(self.reward_cfg.get("strict_center_ok_ratio", 0.75) if strict else self.reward_cfg.get("distribution_voronoi_center_ok_ratio_threshold", 0.5))
            area_threshold = float(self.reward_cfg.get("strict_area_cv", 0.30) if strict else self.reward_cfg.get("distribution_area_cv_threshold", 0.40))
            inside_required = float(self.reward_cfg.get("strict_inside_ratio", 0.99) if strict else self.reward_cfg.get("distribution_inside_ratio_threshold", 0.95))
            center_ok_ratio = float(np.mean(np.asarray(center_dists) <= center_threshold))
            inside_ratio = float(np.mean([(self.inner_x_min <= p[0] <= self.inner_x_max and self.inner_y_min <= p[1] <= self.inner_y_max) for p in positions]))
            max_speed_val = float(max(self.pursuers[i].speed for i in active)) if active else float("inf")
            mean_speed_val = float(np.mean([self.pursuers[i].speed for i in active])) if active else float("inf")
            require_all_active_geometry = bool(self.reward_cfg.get("coverage_geometry_requires_all_active", True))
            all_active_for_geometry = (len(active) == len(self.pursuers)) if require_all_active_geometry else True
            geometry_strict = bool(area_cv <= area_threshold and center_ok_ratio >= center_ok_ratio_threshold and inside_ratio >= inside_required and all_active_for_geometry and not self.evaders)
            if strict and bool(self.reward_cfg.get("coverage_strict_static_enabled", False)):
                speed_threshold = float(self.reward_cfg.get("strict_speed_threshold", 0.15))
                speed_strict = bool(max_speed_val <= speed_threshold)
                converged = bool(geometry_strict and speed_strict)
            else:
                speed_strict = bool(max_speed_val <= float(self.reward_cfg.get("strict_speed_threshold", 0.15))) if strict else False
                converged = geometry_strict
        if update_hold:
            self.distribution_hold_steps = self.distribution_hold_steps + 1 if converged else 0
        hold_required = int(self.reward_cfg.get("strict_hold_steps", 20) if strict else self.reward_cfg.get("distribution_success_hold_steps", 1))
        success = bool(converged and self.distribution_hold_steps >= hold_required)
        strict_static_hold_counter = int(self.distribution_hold_steps) if strict else 0
        self.last_distribution_metrics = {"area_cv": float(area_cv), "center_ok_ratio": float(center_ok_ratio), "inside_ratio": float(inside_ratio), "converged_now": bool(converged), "hold_steps": int(self.distribution_hold_steps), "success_after_hold": success, "max_speed": max_speed_val, "mean_speed": mean_speed_val, "geometry_strict": bool(geometry_strict), "speed_strict": bool(speed_strict), "strict_static_hold_counter": strict_static_hold_counter, "strict_static_success": bool(success and strict)}
        return success

    def _mean_shift_reward(self, i: int, target_id: int, before_p: np.ndarray, before_e: np.ndarray, after_p: np.ndarray) -> float:
        epos = before_e[target_id]
        ppos = before_p[i]
        r_e = float(self.reward_cfg.get("r_e", 8.0))
        r_in = max(0.0, r_e - float(self.reward_cfg.get("mean_shift_inner_margin", 2.0)))
        r_out = r_e + float(self.reward_cfg.get("mean_shift_outer_margin", 2.0))
        cell = float(self.reward_cfg.get("mean_shift_cell_size", 1.5))
        sense = float(self.reward_cfg.get("mean_shift_sense_radius", 35.0))
        xs = np.arange(max(0.5, epos[0] - r_out), min(self.width - 0.5, epos[0] + r_out) + 0.5 * cell, cell)
        ys = np.arange(max(0.5, epos[1] - r_out), min(self.height - 0.5, epos[1] + r_out) + 0.5 * cell, cell)
        weighted, weights = [], []
        for x in xs:
            for y in ys:
                c = np.array([x, y], dtype=float)
                radius = np.linalg.norm(c - epos)
                if radius < r_in or radius > r_out or np.linalg.norm(c - ppos) > sense:
                    continue
                if any(np.linalg.norm(c - np.array([o.x, o.y])) <= o.r + float(self.reward_cfg.get("mean_shift_obstacle_margin", 2.0)) for o in self.obstacles):
                    continue
                occupied = False
                for j, p in enumerate(self.pursuers):
                    if j != i and not p.deactivated and np.linalg.norm(c - after_p[j]) <= float(self.reward_cfg.get("mean_shift_occupancy_radius", 4.0)):
                        occupied = True
                        break
                if occupied:
                    continue
                local_w = 0.5 * (1.0 + np.cos(np.pi * np.clip(np.linalg.norm(c - ppos) / max(sense, 1e-6), 0.0, 1.0)))
                radial_w = np.exp(-((radius - r_e) ** 2) / (2.0 * float(self.reward_cfg.get("mean_shift_radial_sigma", 2.0)) ** 2))
                weighted.append(c)
                weights.append(local_w * radial_w)
        if not weights:
            return 0.0
        raw = np.average(np.asarray(weighted), axis=0, weights=np.asarray(weights))
        direction = raw - epos
        norm = np.linalg.norm(direction)
        if norm <= 1e-6:
            direction = epos - ppos
            norm = max(np.linalg.norm(direction), 1e-6)
        target = epos + r_e * direction / norm
        before_d = np.linalg.norm(before_p[i] - target)
        after_d = np.linalg.norm(after_p[i] - target)
        return float(np.clip(before_d - after_d, -float(self.reward_cfg.get("mean_shift_progress_clip", 3.0)), float(self.reward_cfg.get("mean_shift_progress_clip", 3.0))))

    def _capture_reward_mode(self) -> str:
        return str(self.reward_cfg.get("capture_reward_mode", "legacy")).strip().lower()

    def _ring_importance_ms_enabled(self) -> bool:
        return self._capture_reward_mode() in {"ring_importance_ms_v0", "ring_ms_v0", "cr_ms_v0"}

    def _ring_ms_radii(self, i: int, target_id: int) -> Tuple[float, float, float]:
        pursuer_radius = float(getattr(self.pursuers[i], "r", 0.0)) if 0 <= i < len(self.pursuers) else 0.0
        evader_radius = float(getattr(self.evaders[target_id], "r", 0.0)) if 0 <= target_id < len(self.evaders) else pursuer_radius
        default_inner_surface = float(self.reward_cfg.get("d_safe", 4.0)) + float(self.reward_cfg.get("ring_ms_inner_extra_margin", 0.5))
        inner_surface = float(self.reward_cfg.get("ring_ms_inner_surface_radius", default_inner_surface))
        inner_center = float(self.reward_cfg.get("ring_ms_inner_center_radius", inner_surface + pursuer_radius + evader_radius))
        preferred = float(self.reward_cfg.get("ring_ms_preferred_center_radius", self.reward_cfg.get("r_e", 8.0)))
        outer = float(self.reward_cfg.get("ring_ms_outer_center_radius", preferred + float(self.reward_cfg.get("ring_ms_outer_margin", 2.5))))
        if outer <= inner_center:
            outer = inner_center + max(1.0, float(self.reward_cfg.get("ring_ms_cell_size", 1.5)))
        preferred = float(np.clip(preferred, inner_center + 1e-6, outer - 1e-6))
        return inner_center, preferred, outer

    def _ring_ms_angle_weight(self, candidate: np.ndarray, evader_pos: np.ndarray, evader_velocity: np.ndarray) -> float:
        velocity = np.asarray(evader_velocity, dtype=float)
        speed = float(np.linalg.norm(velocity))
        static_threshold = float(self.reward_cfg.get("ring_ms_velocity_static_threshold", 0.30))
        full_threshold = float(self.reward_cfg.get("ring_ms_velocity_full_threshold", 1.00))
        if speed <= static_threshold or full_threshold <= static_threshold:
            return 1.0
        rel = np.asarray(candidate, dtype=float) - np.asarray(evader_pos, dtype=float)
        dist = float(np.linalg.norm(rel))
        if dist <= 1e-9:
            return 1.0
        gate = float(np.clip((speed - static_threshold) / max(full_threshold - static_threshold, 1e-9), 0.0, 1.0))
        cos_theta = float(np.dot(rel / dist, velocity / max(speed, 1e-9)))
        alpha = float(self.reward_cfg.get("ring_ms_angle_alpha", 0.25))
        w_min = float(self.reward_cfg.get("ring_ms_angle_weight_min", 0.75))
        w_max = float(self.reward_cfg.get("ring_ms_angle_weight_max", 1.25))
        return float(np.clip(1.0 + alpha * gate * cos_theta, w_min, w_max))

    def _ring_ms_point_blocked_by_obstacle(self, point: np.ndarray) -> bool:
        margin = float(self.reward_cfg.get("ring_ms_obstacle_margin", self.reward_cfg.get("mean_shift_obstacle_margin", 2.0)))
        return any(np.linalg.norm(point - np.array([o.x, o.y], dtype=float)) <= o.r + margin for o in self.obstacles)

    def _ring_ms_point_in_bounds(self, point: np.ndarray) -> bool:
        margin = float(self.reward_cfg.get("ring_ms_boundary_margin", self.reward_cfg.get("mean_shift_boundary_margin", 0.5)))
        return bool(margin <= point[0] <= self.width - margin and margin <= point[1] <= self.height - margin)

    @staticmethod
    def _angle_delta(a: float, b: float) -> float:
        return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)

    def _ring_ms_candidate_occupied(
        self,
        candidate: np.ndarray,
        evader_pos: np.ndarray,
        candidate_radius: float,
        pursuer_index: int,
        before_p: np.ndarray,
        preferred_radius: float,
    ) -> bool:
        mode = str(self.reward_cfg.get("ring_ms_occupancy_mode", "cartesian_disk")).strip().lower()
        if mode in {"phase", "phase_sector", "angular"}:
            width = float(self.reward_cfg.get("ring_ms_phase_occupancy_width", np.deg2rad(30.0)))
            if width > np.pi:
                width = float(np.deg2rad(width))
            radial_margin = float(self.reward_cfg.get("ring_ms_phase_occupancy_radial_margin", 4.0))
            candidate_phase = float(np.arctan2(candidate[1] - evader_pos[1], candidate[0] - evader_pos[0]))
            for j, pursuer in enumerate(self.pursuers):
                if j == pursuer_index or pursuer.deactivated:
                    continue
                rel = before_p[j] - evader_pos
                radius = float(np.linalg.norm(rel))
                if abs(radius - preferred_radius) > radial_margin:
                    continue
                phase = float(np.arctan2(rel[1], rel[0]))
                if abs(self._angle_delta(candidate_phase, phase)) <= width:
                    return True
            return False
        occupancy_radius = float(self.reward_cfg.get("ring_ms_occupancy_radius", self.reward_cfg.get("mean_shift_occupancy_radius", 4.0)))
        for j, pursuer in enumerate(self.pursuers):
            if j != pursuer_index and not pursuer.deactivated and np.linalg.norm(candidate - before_p[j]) <= occupancy_radius:
                return True
        return False

    def _ring_ms_target(
        self,
        i: int,
        target_id: int,
        before_p: np.ndarray,
        before_e: np.ndarray,
        before_evader_velocities: np.ndarray,
    ) -> Optional[np.ndarray]:
        epos = np.asarray(before_e[target_id], dtype=float)
        ppos = np.asarray(before_p[i], dtype=float)
        inner_radius, preferred_radius, outer_radius = self._ring_ms_radii(i, target_id)
        cell = float(self.reward_cfg.get("ring_ms_cell_size", self.reward_cfg.get("mean_shift_cell_size", 1.5)))
        radial_sigma = max(float(self.reward_cfg.get("ring_ms_radial_sigma", 2.0)), 1e-6)
        xs = np.arange(max(0.5, epos[0] - outer_radius), min(self.width - 0.5, epos[0] + outer_radius) + 0.5 * cell, cell)
        ys = np.arange(max(0.5, epos[1] - outer_radius), min(self.height - 0.5, epos[1] + outer_radius) + 0.5 * cell, cell)
        velocity = np.asarray(before_evader_velocities[target_id], dtype=float) if len(before_evader_velocities) > target_id else np.zeros(2, dtype=float)
        candidates: List[np.ndarray] = []
        weights: List[float] = []
        radii: List[float] = []
        for x in xs:
            for y in ys:
                candidate = np.array([x, y], dtype=float)
                radius = float(np.linalg.norm(candidate - epos))
                if radius < inner_radius or radius > outer_radius:
                    continue
                if not self._ring_ms_point_in_bounds(candidate) or self._ring_ms_point_blocked_by_obstacle(candidate):
                    continue
                if self._ring_ms_candidate_occupied(candidate, epos, radius, i, before_p, preferred_radius):
                    continue
                radial_w = float(np.exp(-((radius - preferred_radius) ** 2) / (2.0 * radial_sigma ** 2)))
                angle_w = self._ring_ms_angle_weight(candidate, epos, velocity)
                weight = radial_w * angle_w
                if weight <= 0.0:
                    continue
                candidates.append(candidate)
                weights.append(weight)
                radii.append(radius)
        if not weights:
            return None
        candidate_arr = np.asarray(candidates, dtype=float)
        weight_arr = np.asarray(weights, dtype=float)
        raw = np.average(candidate_arr, axis=0, weights=weight_arr)
        direction = raw - epos
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            direction = epos - ppos
            norm = max(float(np.linalg.norm(direction)), 1e-6)
        target = epos + preferred_radius * direction / norm
        if self._ring_ms_point_in_bounds(target) and not self._ring_ms_point_blocked_by_obstacle(target):
            return target.astype(float)
        beta = float(self.reward_cfg.get("ring_ms_fallback_radial_penalty", 0.10))
        scores = weight_arr - beta * np.abs(np.asarray(radii, dtype=float) - preferred_radius)
        return candidate_arr[int(np.argmax(scores))].astype(float)

    def _ring_importance_ms_reward(
        self,
        i: int,
        target_id: int,
        before_p: np.ndarray,
        before_e: np.ndarray,
        before_evader_velocities: np.ndarray,
        after_p: np.ndarray,
    ) -> float:
        target = self._ring_ms_target(i, target_id, before_p, before_e, before_evader_velocities)
        if target is None:
            return 0.0
        before_d = float(np.linalg.norm(before_p[i] - target))
        after_d = float(np.linalg.norm(after_p[i] - target))
        clip = float(self.reward_cfg.get("ring_ms_progress_clip", self.reward_cfg.get("mean_shift_progress_clip", 3.0)))
        return float(np.clip(before_d - after_d, -clip, clip))

    def _front_reward(self, i: int, target_id: int, after_p: np.ndarray, after_e: np.ndarray) -> float:
        evader = self.evaders[target_id]
        vel = np.asarray(evader.velocity, dtype=float)
        speed = np.linalg.norm(vel)
        if speed <= float(self.reward_cfg.get("front_min_evader_speed", 1e-3)):
            return 0.0
        heading = vel / max(speed, 1e-6)
        rel = after_p[i] - after_e[target_id]
        dist = np.linalg.norm(rel)
        if dist <= 1e-6:
            return 0.0
        direction = rel / dist
        front_score = float(np.dot(direction, heading))
        if front_score <= 0.0:
            return 0.0
        r_e = float(self.reward_cfg.get("r_e", 8.0))
        ring_score = np.exp(-((dist - r_e) ** 2) / max(float(self.reward_cfg.get("front_radial_sigma", 4.0)) ** 2, 1e-6))
        gate = 1.0 / (1.0 + np.exp(-float(self.reward_cfg.get("front_k", 8.0)) * (front_score - np.cos(float(self.reward_cfg.get("front_theta", np.pi / 3.0))))))
        return float(ring_score * gate * front_score)

    def _coverage_training_uses_strict(self) -> bool:
        mode = str(self.reward_cfg.get("coverage_training_success_mode", "loose")).strip().lower()
        if mode not in {"loose", "strict"}:
            raise ValueError("reward.coverage_training_success_mode must be 'loose' or 'strict'")
        return mode == "strict"

    def step(self, pursuer_actions: List[Optional[int]], evader_actions: Optional[List[Optional[int]]] = None) -> StepResult:
        if evader_actions is None:
            evader_actions = [None] * len(self.evaders)
        before_p = np.asarray([self._position(p) for p in self.pursuers], dtype=float)
        before_e = np.asarray([self._position(e) for e in self.evaders], dtype=float) if self.evaders else np.zeros((0, 2))
        before_cov = self._coverage_potentials(before_p)
        self._begin_collision_step()
        try:
            for p, action in zip(self.pursuers, pursuer_actions):
                self._move_robot(p, action)
            for e, action in zip(self.evaders, evader_actions):
                self._move_robot(e, action)
            self._refresh_collisions()
        finally:
            self._end_collision_step()
        self._record_soft_boundary_out_of_bounds()
        after_p = np.asarray([self._position(p) for p in self.pursuers], dtype=float)
        after_e = np.asarray([self._position(e) for e in self.evaders], dtype=float) if self.evaders else np.zeros((0, 2))
        rewards = np.zeros(len(self.pursuers), dtype=float)
        infos = [{"state": "normal"} for _ in self.pursuers]
        captured_events: List[Dict[str, Any]] = []
        if self.task == "coverage":
            coverage_success_bonus = 0.0
            coverage_hold_bonus = 0.0
            after_cov = self._coverage_potentials(after_p)
            rewards += np.clip(before_cov - after_cov, -float(self.reward_cfg.get("coverage_phi_clip", 0.05)), float(self.reward_cfg.get("coverage_phi_clip", 0.05)))
            rewards -= float(self.reward_cfg.get("coverage_phi_coeff", 0.15)) * after_cov
            rewards += float(self.env_cfg.get("timestep_penalty", -1.0))
            training_strict = self._coverage_training_uses_strict()
            training_success = self.check_distribution_converged(update_hold=True, strict=training_strict)
            converged_now = bool(self.last_distribution_metrics.get("converged_now", False))
            geometry_strict = bool(self.last_distribution_metrics.get("geometry_strict", False))
            speed_strict = bool(self.last_distribution_metrics.get("speed_strict", False))
            hold_required = int(self.reward_cfg.get("strict_hold_steps", 20) if training_strict else self.reward_cfg.get("distribution_success_hold_steps", 1))
            base_hold = float(self.reward_cfg.get("coverage_hold_reward", 0.0))
            base_success = float(self.reward_cfg.get("coverage_success_reward", 120.0))
            # ---- hold / success reward ----
            if training_strict and bool(self.reward_cfg.get("coverage_static_speed_penalty_enabled", False)):
                # 支线2b revised: two-phase reward
                # Phase 1: geometry only → full hold per step
                # Phase 2: geometry + speed both met → 3x hold per step
                if converged_now:
                    if not self._geometry_first_achieved:
                        coverage_success_bonus = base_success
                        self._geometry_first_achieved = True
                    coverage_hold_bonus = base_hold * 3.0
                elif geometry_strict and not speed_strict:
                    if not self._geometry_first_achieved:
                        coverage_success_bonus = base_success
                        self._geometry_first_achieved = True
                    coverage_hold_bonus = base_hold
                else:
                    coverage_hold_bonus = 0.0
                    coverage_success_bonus = 0.0
            else:
                # Original logic (speed penalty disabled)
                if converged_now:
                    coverage_hold_bonus = base_hold
                if training_success and int(self.distribution_hold_steps) == hold_required:
                    coverage_success_bonus = base_success
                elif converged_now and bool(self.reward_cfg.get("coverage_success_reward_repeat", False)):
                    coverage_success_bonus = float(self.reward_cfg.get("coverage_success_repeat_reward", 0.0))
            # 支线2a: speed penalty (quadratic on excess)
            speed_penalty_gate = False
            speed_penalty_sum = 0.0
            if training_strict and bool(self.reward_cfg.get("coverage_static_speed_penalty_enabled", False)):
                gate_area = float(self.reward_cfg.get("coverage_static_gate_area_cv", 0.35))
                gate_inside = float(self.reward_cfg.get("coverage_static_gate_inside_ratio", 0.99))
                gate_center = float(self.reward_cfg.get("coverage_static_gate_center_ok_ratio", 0.90))
                cv_now = float(self.last_distribution_metrics.get("area_cv", float("inf")))
                inside_now = float(self.last_distribution_metrics.get("inside_ratio", 0.0))
                center_now = float(self.last_distribution_metrics.get("center_ok_ratio", 0.0))
                speed_penalty_gate = bool(cv_now <= gate_area and inside_now >= gate_inside and center_now >= gate_center)
                if speed_penalty_gate:
                    penalty_omega = float(self.reward_cfg.get("coverage_static_speed_penalty_omega", 0.20))
                    penalty_cap = float(self.reward_cfg.get("coverage_static_speed_penalty_cap", 1.00))
                    speed_t = float(self.reward_cfg.get("strict_speed_threshold", 0.15))
                    for i, p in enumerate(self.pursuers):
                        if p.deactivated:
                            continue
                        speed_excess = max(0.0, float(p.speed) - speed_t)
                        penalty = -penalty_omega * (speed_excess / speed_t) ** 2
                        penalty = max(penalty, -penalty_cap)
                        rewards[i] += penalty
                        speed_penalty_sum += penalty
            if coverage_success_bonus or coverage_hold_bonus:
                rewards += coverage_success_bonus + coverage_hold_bonus
        else:
            active_targets = [j for j, e in enumerate(self.evaders) if not e.deactivated]
            related = float(self.env_cfg.get("related_distance", 18.0))
            for target_id in active_targets:
                dists = np.asarray([np.inf if p.deactivated else np.linalg.norm(after_p[i] - after_e[target_id]) for i, p in enumerate(self.pursuers)])
                mask = dists <= related
                count = int(mask.sum())
                if count >= 3:
                    rewards += float(self.env_cfg.get("evenly_distributed_reward", 5.0)) * max(0.0, 1.0 - 0.3 * (count - 3)) * mask.astype(float)
            for i, p in enumerate(self.pursuers):
                if p.deactivated or not active_targets:
                    continue
                rewards[i] += float(self.env_cfg.get("timestep_penalty", -1.0))
                target_id = min(active_targets, key=lambda j: np.linalg.norm(after_p[i] - after_e[j]))
                d_before = np.linalg.norm(before_p[i] - before_e[target_id])
                d_after = np.linalg.norm(after_p[i] - after_e[target_id])
                rewards[i] += float(self.reward_cfg.get("omega_approach", 1.0)) * np.clip(d_before - d_after, -float(self.reward_cfg.get("c_d", 3.0)), float(self.reward_cfg.get("c_d", 3.0)))
                rewards[i] += float(self.reward_cfg.get("omega_mean_shift", 2.0)) * self._mean_shift_reward(i, target_id, before_p, before_e, after_p)
                rewards[i] += float(self.reward_cfg.get("omega_front", 0.5)) * self._front_reward(i, target_id, after_p, after_e)
                emergency_penalty = float(self._emergency_proximity_penalties()[i])
                if emergency_penalty:
                    rewards[i] += emergency_penalty
            captured_events = self._loose_capture_events()
            if captured_events:
                for event in captured_events:
                    self.evaders[event["evader_id"]].deactivated = True
                    if event.get("capture_type") == "stationary":
                        factor = 1.0
                    else:
                        factor = (TWO_PI / max(len(event["participants"]), 1)) * np.exp(-float(np.std(event["angles"])))
                    for i in event["participants"]:
                        rewards[i] += float(self.env_cfg.get("goal_reward", 120.0)) * factor
            self.last_capture_events = captured_events
        boundary_penalties = self._boundary_penalties()
        self.last_boundary_penalty_count = int(np.count_nonzero(boundary_penalties))
        self.last_boundary_penalty_sum = float(np.sum(boundary_penalties))
        rewards += boundary_penalties
        boundary_proximity_penalties = self._boundary_proximity_penalties()
        self.last_boundary_proximity_penalty_count = int(np.count_nonzero(boundary_proximity_penalties))
        self.last_boundary_proximity_penalty_sum = float(np.sum(boundary_proximity_penalties))
        rewards += boundary_proximity_penalties
        for i, p in enumerate(self.pursuers):
            if p.collision:
                rewards[i] += float(self.env_cfg.get("collision_penalty", -80.0))
                infos[i] = {"state": "deactivated after collision" if p.deactivated else "collision"}
        all_captured = bool(self.evaders and all(e.deactivated and not e.collision for e in self.evaders))
        evader_lost = bool(self.evaders and all(e.deactivated for e in self.evaders) and not all_captured)
        self.episode_step += 1
        self.total_steps += 1
        timeout = self.episode_step >= self.episode_max_length
        coverage_done = False
        if self.task == "coverage" and bool(self.last_distribution_metrics.get("success_after_hold", False)):
            hold_required = int(self.reward_cfg.get("strict_hold_steps", 20) if self._coverage_training_uses_strict() else self.reward_cfg.get("distribution_success_hold_steps", 1))
            coverage_done = int(self.distribution_hold_steps) >= hold_required + int(self.distribution_converged_grace_steps)
        dones = []
        for i, p in enumerate(self.pursuers):
            done = bool(p.deactivated or timeout or coverage_done or all_captured or evader_lost or sum(not q.deactivated for q in self.pursuers) < 3)
            if timeout and infos[i]["state"] == "normal":
                infos[i] = {"state": "too long episode"}
            elif coverage_done and infos[i]["state"] == "normal":
                infos[i] = {"state": "distribution converged"}
            elif all_captured and infos[i]["state"] == "normal":
                infos[i] = {"state": "all targets captured"}
            elif evader_lost and infos[i]["state"] == "normal":
                infos[i] = {"state": "evader collision"}
            dones.append(done)
        if self.continuous_control:
            for i, pursuer in enumerate(self.pursuers):
                infos[i]["action_diagnostics"] = dict(getattr(pursuer, "last_action_diagnostics", {}) or {})
        self.last_reward_terms = {
            "reward_mean": float(np.mean(rewards)),
            "capture_count": float(len(captured_events)),
            "coverage_success": float(coverage_done),
            "coverage_converged_now": float(self.last_distribution_metrics.get("converged_now", False)) if self.task == "coverage" else 0.0,
            "coverage_hold_steps": float(self.distribution_hold_steps) if self.task == "coverage" else 0.0,
            "coverage_done_grace_steps": float(self.distribution_converged_grace_steps) if self.task == "coverage" else 0.0,
            "coverage_training_success_mode_strict": float(self._coverage_training_uses_strict()) if self.task == "coverage" else 0.0,
            "coverage_success_bonus": float(coverage_success_bonus) if self.task == "coverage" else 0.0,
            "coverage_hold_bonus": float(coverage_hold_bonus) if self.task == "coverage" else 0.0,
            "coverage_max_speed": float(self.last_distribution_metrics.get("max_speed", 0.0)) if self.task == "coverage" else 0.0,
            "coverage_mean_speed": float(self.last_distribution_metrics.get("mean_speed", 0.0)) if self.task == "coverage" else 0.0,
            "coverage_geometry_strict": float(geometry_strict) if self.task == "coverage" else 0.0,
            "coverage_speed_strict": float(speed_strict) if self.task == "coverage" else 0.0,
            "coverage_strict_static_hold_counter": float(self.last_distribution_metrics.get("strict_static_hold_counter", 0)) if self.task == "coverage" else 0.0,
            "coverage_static_speed_penalty_gate": float(speed_penalty_gate) if self.task == "coverage" else 0.0,
            "coverage_static_speed_penalty_sum": float(speed_penalty_sum) if self.task == "coverage" else 0.0,
            "capture_stationary_event": float(any(e.get("capture_type") == "stationary" for e in captured_events)),
            "capture_stationary_counter": float(max(self._stationary_capture_counters.values()) if self._stationary_capture_counters else 0.0),
            "boundary_penalty_count": float(self.last_boundary_penalty_count),
            "boundary_penalty_sum": float(self.last_boundary_penalty_sum),
            "boundary_proximity_penalty_count": float(self.last_boundary_proximity_penalty_count),
            "boundary_proximity_penalty_sum": float(self.last_boundary_proximity_penalty_sum),
            "emergency_proximity_penalty_count": float(self.last_emergency_proximity_penalty_count),
            "emergency_proximity_penalty_sum": float(self.last_emergency_proximity_penalty_sum),
            "soft_boundary_out_of_bounds_pursuer_count": float(len(self.last_out_of_bounds_pursuers)),
            "soft_boundary_out_of_bounds_evader_count": float(len(self.last_out_of_bounds_evaders)),
        }
        return StepResult(self.get_observations(), rewards, dones, infos)

    def episode_record(self, task: Optional[str] = None) -> Dict[str, Any]:
        captured = bool(self.evaders and all(e.deactivated and not e.collision for e in self.evaders))
        collision_fields = self._collision_episode_fields()
        collision_free = not bool(collision_fields["collision_event"])
        all_active = all(not p.deactivated for p in self.pursuers)
        loose_metrics = dict(self.last_distribution_metrics) if self.task == "coverage" else {}
        loose = bool(loose_metrics.get("success_after_hold", False)) if self.task == "coverage" else False
        strict = False
        strict_metrics: Dict[str, Any] = {}
        training_success = False
        if self.task == "coverage":
            old_hold = int(self.distribution_hold_steps)
            old_metrics = dict(self.last_distribution_metrics)
            loose = self.check_distribution_converged(update_hold=False, strict=False)
            loose_metrics = dict(self.last_distribution_metrics)
            strict = self.check_distribution_converged(update_hold=False, strict=True)
            strict_metrics = dict(self.last_distribution_metrics)
            self.distribution_hold_steps = old_hold
            self.last_distribution_metrics = old_metrics
            training_success = strict if self._coverage_training_uses_strict() else loose
        coverage_requires_collision_free = bool(self.reward_cfg.get("coverage_success_requires_collision_free", True)) if self.task == "coverage" else True
        coverage_requires_all_active = bool(self.reward_cfg.get("coverage_success_requires_all_active", True)) if self.task == "coverage" else True
        coverage_survival_ok = bool((collision_free or not coverage_requires_collision_free) and (all_active or not coverage_requires_all_active))
        return {
            "task": task or self.task,
            "length": int(self.episode_step),
            "captured": captured,
            "fully_capture": bool(captured and collision_free and all_active),
            "collision_free": collision_free,
            **collision_fields,
            "all_pursuers_active": all_active,
            "active_pursuers": int(sum(not p.deactivated for p in self.pursuers)),
            "coverage_success_requires_collision_free": bool(coverage_requires_collision_free) if self.task == "coverage" else None,
            "coverage_success_requires_all_active": bool(coverage_requires_all_active) if self.task == "coverage" else None,
            "boundary_collision_event": bool(any(getattr(p, "boundary_collision", False) for p in self.pursuers)),
            "boundary_collision_pursuers": [int(p.id) for p in self.pursuers if getattr(p, "boundary_collision", False)],
            "soft_boundary_out_of_bounds_event": bool(self.episode_out_of_bounds_step_count > 0),
            "soft_boundary_out_of_bounds_step_count": int(self.episode_out_of_bounds_step_count),
            "soft_boundary_out_of_bounds_pursuer_steps": int(self.episode_out_of_bounds_pursuer_steps),
            "soft_boundary_out_of_bounds_evader_steps": int(self.episode_out_of_bounds_evader_steps),
            "soft_boundary_out_of_bounds_pursuer_ids": list(self.episode_out_of_bounds_pursuer_ids),
            "soft_boundary_out_of_bounds_evader_ids": list(self.episode_out_of_bounds_evader_ids),
            "soft_boundary_current_out_of_bounds_pursuers": list(self.last_out_of_bounds_pursuers),
            "soft_boundary_current_out_of_bounds_evaders": list(self.last_out_of_bounds_evaders),
            "coverage_loose_success": bool(loose and coverage_survival_ok),
            "coverage_strict_success": bool(strict and coverage_survival_ok),
            "coverage_training_success": bool(training_success and coverage_survival_ok),
            "coverage_training_success_mode": "strict" if self.task == "coverage" and self._coverage_training_uses_strict() else "loose",
            "distribution_metrics": loose_metrics,
            "coverage_strict_distribution_metrics": strict_metrics,
            "reward_terms": dict(self.last_reward_terms),
            "pursuer_positions": [[float(p.x), float(p.y)] for p in self.pursuers if not p.deactivated],
            "pursuer_all_positions": [[float(p.x), float(p.y)] for p in self.pursuers],
            "pursuer_active_mask": [bool(not p.deactivated) for p in self.pursuers],
        }
