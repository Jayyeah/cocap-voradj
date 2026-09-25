"""Initialization-only A3 curriculum for canonical Final Capture.

This module deliberately owns reset geometry only.  It does not alter reward,
observations, actions, dynamics, collision semantics, or episode lifecycle.
The curriculum is opt-in through ``a3_initialization_curriculum.enabled``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


STAGES = ("I0", "I1", "I2", "I3")
SCHEMA = "a3-initialization-only-v1"


def _as_range(value: Sequence[float], field: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must be a two-element range")
    low, high = float(value[0]), float(value[1])
    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        raise ValueError(f"{field} must be finite with low <= high")
    return low, high


def _close_range(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == 2 and all(np.isclose(float(a), float(b)) for a, b in zip(left, right))


class A3InitializationCurriculum:
    """Validate and apply one registered A3 initialization stage."""

    _EXPECTED = {
        "I0": {"mode": "ring", "distance_ranges": [[12.0, 13.0]] * 4, "angle_jitter_deg": 5.0, "visibility_mode": "all_direct"},
        "I1": {"mode": "ring", "distance_ranges": [[13.0, 16.0]] * 4, "angle_jitter_deg": 7.5, "visibility_mode": "all_direct"},
        "I2": {
            "mode": "ring",
            "distance_ranges": [[17.5, 19.0], [17.5, 19.0], [23.0, 26.0], [23.0, 26.0]],
            "angle_jitter_deg": 12.5,
            "visibility_mode": "two_direct_two_hidden",
        },
        "I3": {"mode": "map_random", "angle_jitter_deg": 0.0, "visibility_mode": "unconstrained"},
    }

    def __init__(self, env: Any):
        self.env = env
        raw = env.config.get("a3_initialization_curriculum", {}) or {}
        if not isinstance(raw, dict):
            raise ValueError("a3_initialization_curriculum must be a mapping")
        self.config = raw
        self.enabled = bool(raw.get("enabled", False))
        self.stage = str(raw.get("stage", "I3")).strip().upper()
        if not self.enabled:
            return
        if raw.get("schema") != SCHEMA:
            raise ValueError(f"a3_initialization_curriculum.schema must be {SCHEMA}")
        if self.stage not in STAGES:
            raise ValueError(f"a3_initialization_curriculum.stage must be one of {STAGES}")
        # The mixed Final trainer also constructs a VorAdj coverage scene with
        # zero evaders.  That scene must retain its ordinary initialization;
        # A3 is a capture-task reset curriculum only.
        if int(env.env_cfg.get("num_evaders", 0)) == 0:
            self._skip_no_evader_task = True
            return
        self._validate_final_contract()
        self._validate_stage_table()

    def _validate_final_contract(self) -> None:
        env_cfg = self.env.env_cfg
        reward = self.env.reward_cfg
        voradj = self.env.config.get("voradj", {}) or {}
        if int(env_cfg.get("num_pursuers", 0)) != 4 or int(env_cfg.get("num_evaders", 0)) != 1:
            raise ValueError("A3 initialization-only curriculum requires canonical 4 pursuers / 1 evader")
        if int(env_cfg.get("num_obstacles", 0)) != 1:
            raise ValueError("A3 initialization-only curriculum requires one obstacle")
        if not np.isclose(float(env_cfg.get("width", 0.0)), 120.0) or not np.isclose(float(env_cfg.get("height", 0.0)), 120.0):
            raise ValueError("A3 initialization-only curriculum requires a 120x120 map")
        if str(env_cfg.get("collision_semantics", "")).strip().lower() != "synchronized_swept_v1":
            raise ValueError("A3 requires synchronized_swept_v1 collision semantics")
        if int(env_cfg.get("episode_max_length", 0)) != 3000:
            raise ValueError("A3 requires a 3000-step Final episode horizon")
        if not np.isclose(float(env_cfg.get("init_speed", -1.0)), 0.0):
            raise ValueError("A3 reset target speed must remain canonical zero")
        if bool(self.env.per_cfg.get("global_evader_visibility", False)):
            raise ValueError("A3 cannot enable global evader visibility")
        required_reward = {
            "capture_reward_mode": "ring_importance_ms_v0",
            "coverage_ce_reward_scale": 10.0,
            "coverage_ce_pbrs_enabled": True,
            "post_capture_coverage_window_steps": 500,
            "r_e": 8.0,
            "k_required": 3,
        }
        for key, expected in required_reward.items():
            actual = reward.get(key)
            if isinstance(expected, float):
                if not np.isclose(float(actual), expected):
                    raise ValueError(f"A3 Final reward contract drift at reward.{key}: {actual!r}")
            elif actual != expected:
                raise ValueError(f"A3 Final reward contract drift at reward.{key}: {actual!r}")
        if not bool(voradj.get("support_reward_blend_enabled", False)):
            raise ValueError("A3 requires Final support reward blend to remain enabled")
        if str(voradj.get("support_reward_capture_target_mode", "")) != "neighbor_visible":
            raise ValueError("A3 requires neighbor-visible support semantics")
        if str(voradj.get("support_reward_capture_component_mode", "")) != "approach_only":
            raise ValueError("A3 requires approach-only support semantics")
        if bool(voradj.get("capture_episode_ends_on_capture", False)):
            raise ValueError("A3 cannot end an episode at capture")

    def _validate_stage_table(self) -> None:
        stages = self.config.get("stages")
        if not isinstance(stages, dict):
            raise ValueError("a3_initialization_curriculum.stages must be a mapping")
        for stage in STAGES:
            if stage not in stages or not isinstance(stages[stage], dict):
                raise ValueError(f"missing registered A3 stage {stage}")
            expected = self._EXPECTED[stage]
            actual = stages[stage]
            if str(actual.get("mode", "")).strip().lower() != expected["mode"]:
                raise ValueError(f"A3 stage {stage} mode is not registered")
            if not np.isclose(float(actual.get("angle_jitter_deg", -1.0)), expected["angle_jitter_deg"]):
                raise ValueError(f"A3 stage {stage} angle jitter is not registered")
            if str(actual.get("visibility_mode", "")) != expected["visibility_mode"]:
                raise ValueError(f"A3 stage {stage} visibility mode is not registered")
            if stage != "I3":
                ranges = actual.get("distance_ranges")
                if not isinstance(ranges, list) or len(ranges) != 4:
                    raise ValueError(f"A3 stage {stage} must contain four distance ranges")
                for observed, registered in zip(ranges, expected["distance_ranges"]):
                    if not _close_range(observed, registered):
                        raise ValueError(f"A3 stage {stage} distance range is not registered")
        target_speed = _as_range(self.config.get("target_speed_range", [0.0, 0.0]), "target_speed_range")
        if not (np.isclose(target_speed[0], 0.0) and np.isclose(target_speed[1], 0.0)):
            raise ValueError("A3 target_speed_range must remain [0.0, 0.0]")

    def apply(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "stage": None, "source": "environment_default"}
        if not self.env.evaders:
            return {"enabled": True, "stage": self.stage, "source": "skipped_no_evader_task"}
        if self.stage != "I3":
            target, pursuers, assignment = self._sample_ring_layout()
            self._reset_layout(target, pursuers)
        else:
            assignment = list(range(len(self.env.pursuers)))
        return self._metadata(assignment)

    def _sample_ring_layout(self) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
        spec = self.config["stages"][self.stage]
        base_ranges = [_as_range(value, f"{self.stage}.distance_ranges") for value in spec["distance_ranges"]]
        if spec["visibility_mode"] == "two_direct_two_hidden":
            assignment = [int(value) for value in self.env.rng.permutation(4)]
        else:
            assignment = list(range(4))
        ranges = [base_ranges[index] for index in assignment]
        max_distance = max(high for _low, high in ranges)
        edge_margin = float(self.config.get("spawn_edge_margin", self.env.env_cfg.get("spawn_edge_margin", 12.0)))
        d_safe = float(self.config.get("obstacle_clearance", self.env.reward_cfg.get("d_safe", 4.0)))
        max_radius = max([float(getattr(p, "r", 1.0)) for p in self.env.pursuers] + [float(getattr(self.env.evaders[0], "r", 1.0))])
        center_low = edge_margin + max_distance + max_radius
        center_high = np.asarray([self.env.width, self.env.height], dtype=float) - center_low
        if np.any(center_high <= center_low):
            raise ValueError("A3 ring stage cannot fit inside the map with its edge margin")
        jitter = math.radians(float(spec["angle_jitter_deg"]))
        min_pursuer_distance = float(self.config.get("pursuer_min_center_distance", 15.0))
        for _attempt in range(int(self.config.get("max_layout_attempts", 10000))):
            target = self.env.rng.uniform(center_low, center_high).astype(float)
            phase = float(self.env.rng.uniform(0.0, 2.0 * np.pi))
            angles = [phase + (2.0 * np.pi * i / 4.0) + float(self.env.rng.uniform(-jitter, jitter)) for i in range(4)]
            distances = [float(self.env.rng.uniform(low, high)) for low, high in ranges]
            pursuers = [target + distance * np.asarray([np.cos(angle), np.sin(angle)], dtype=float) for distance, angle in zip(distances, angles)]
            if self._layout_valid(target, pursuers, edge_margin, d_safe, min_pursuer_distance):
                return target, pursuers, assignment
        raise RuntimeError(f"unable to sample a valid A3 {self.stage} initialization layout")

    def _layout_valid(
        self,
        target: np.ndarray,
        pursuers: List[np.ndarray],
        edge_margin: float,
        obstacle_clearance: float,
        min_pursuer_distance: float,
    ) -> bool:
        target_robot = self.env.evaders[0]
        entities = [(target, float(getattr(target_robot, "r", 1.0)))]
        entities += [(pos, float(getattr(robot, "r", 1.0))) for pos, robot in zip(pursuers, self.env.pursuers)]
        for pos, radius in entities:
            if float(pos[0] - radius) < edge_margin or float(pos[0] + radius) > self.env.width - edge_margin:
                return False
            if float(pos[1] - radius) < edge_margin or float(pos[1] + radius) > self.env.height - edge_margin:
                return False
            for obstacle in self.env.obstacles:
                obstacle_pos = np.asarray([obstacle.x, obstacle.y], dtype=float)
                if float(np.linalg.norm(pos - obstacle_pos)) < radius + float(obstacle.r) + obstacle_clearance:
                    return False
        for i, left in enumerate(pursuers):
            for right in pursuers[i + 1 :]:
                if float(np.linalg.norm(left - right)) < min_pursuer_distance:
                    return False
        if any(float(np.linalg.norm(pos - target)) <= 8.0 for pos in pursuers):
            return False
        return True

    def _reset_layout(self, target: np.ndarray, pursuers: List[np.ndarray]) -> None:
        self.env._reset_robot(self.env.evaders[0], target)
        for robot, position in zip(self.env.pursuers, pursuers):
            self.env._reset_robot(robot, position)

    def _metadata(self, assignment: List[int]) -> Dict[str, Any]:
        target = np.asarray([self.env.evaders[0].x, self.env.evaders[0].y], dtype=float)
        pursuer_positions = [np.asarray([robot.x, robot.y], dtype=float) for robot in self.env.pursuers]
        distances = [float(np.linalg.norm(position - target)) for position in pursuer_positions]
        bearings = [float(np.arctan2(*(position - target)[::-1]) % (2.0 * np.pi)) for position in pursuer_positions]
        direct_count = sum(distance <= float(self.env.per_cfg.get("range", 20.0)) for distance in distances)
        pairwise = [float(np.linalg.norm(left - right)) for i, left in enumerate(pursuer_positions) for right in pursuer_positions[i + 1 :]]
        obstacle_clearance = min(
            [
                float(np.linalg.norm(position - np.asarray([obstacle.x, obstacle.y], dtype=float))) - float(getattr(robot, "r", 1.0)) - float(obstacle.r)
                for position, robot in [(target, self.env.evaders[0]), *zip(pursuer_positions, self.env.pursuers)]
                for obstacle in self.env.obstacles
            ]
            or [float("inf")]
        )
        return {
            "enabled": True,
            "schema": SCHEMA,
            "stage": self.stage,
            "source": "a3_ring_sampler" if self.stage != "I3" else "canonical_map_random",
            "pursuer_positions": [position.tolist() for position in pursuer_positions],
            "target_position": target.tolist(),
            "pursuer_target_distances": distances,
            "pursuer_target_bearings_rad": bearings,
            "direct_visible_count_by_center_range": int(direct_count),
            "pursuer_pairwise_min_center_distance": min(pairwise) if pairwise else None,
            "obstacle_min_surface_clearance": obstacle_clearance,
            "pursuer_assignment": list(assignment),
            "target_speed": float(self.env.evaders[0].speed),
            "initial_capture_events": len(getattr(self.env, "last_capture_events", [])),
            "initial_collision_events": len(getattr(self.env, "last_collision_events", [])),
        }
