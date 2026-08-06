
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from cocap_voradj.envs.coverage_ce import (
    build_masked_voronoi_map,
    ce_transition_reward,
    normalized_control_cost,
)

from cocap_voradj.envs.base import CoCapEnv, StepResult, TWO_PI


SiteKey = Tuple[str, int]


class VorAdjEnv(CoCapEnv):
    """Voronoi-adjacency mixed coverage/capture environment.

    This branch keeps APF dynamics and APF evader control, but replaces pursuer
    fixed-radius entity observation with a global Voronoi ownership map. It is
    intentionally isolated from the legacy CoCapEnv coverage/encirclement paths.
    """

    def __init__(self, config: Dict[str, Any], seed: int = 0):
        super().__init__(config, task="voradj", seed=seed)
        self.coverage_hold_reward_claim_steps = 0
        self.coverage_hold_reward_steps_by_phase = {
            "pre_capture": 0,
            "post_capture": 0,
            "pure_coverage": 0,
        }
        self.post_capture_started = False
        self.post_capture_step = 0
        self.post_capture_coverage_success = False
        self.post_capture_coverage_step: Optional[int] = None
        self.post_capture_grace_remaining = 0
        self.capture_snapshot: Optional[Dict[str, Any]] = None
        self.coverage_geometric_success = False
        self.coverage_settled_success = False
        self.coverage_settle_timeout = False
        self.settle_hold_steps = 0
        self.coverage_settle_elapsed_steps = 0
        self.coverage_geometric_latch_mean_speed: Optional[float] = None
        self.coverage_geometric_latch_max_speed: Optional[float] = None
        self.coverage_geometric_to_settled_steps: Optional[int] = None
        self.coverage_mean_score = 0.0
        self.coverage_max_score = 0.0
        self.coverage_speed_score = 0.0
        self.coverage_motion_gate = 0.0
        self.coverage_motion_score = 0.0
        self.coverage_motion_max_accel = 0.0
        self.coverage_motion_max_turn = 0.0
        self.coverage_motion_success_now = False
        self.last_voradj_metrics: Dict[str, Any] = {}
        self.last_task_labels: List[str] = []
        self.last_raw_task_labels: List[str] = []
        voradj_cfg = self.config.get("voradj", {}) or {}
        self._pursuing_release_delay_steps = max(0, int(voradj_cfg.get("is_pursuing_release_delay_steps", 0)))
        self._center_sqrt_n_normalization_enabled = bool(voradj_cfg.get("center_sqrt_n_normalization_enabled", False))
        self._pursuing_release_counters = [0] * len(self.pursuers)
        self._pursuing_flags_initialized = False
        self._voronoi_cache_version = 0
        self._voronoi_cache_version_tag = -1
        self._voronoi_cache: Dict[str, Dict[str, Any]] = {}
        self.zone_evader_entered_inner: List[bool] = []
        self.zone_evader_exited_after_entry: List[bool] = []
        self.zone_evader_targets: List[Optional[np.ndarray]] = []
        self.zone_evader_target_reached: List[bool] = []
        self.zone_pursuer_left_inner_event = False

    def reset(self, *args, **kwargs):
        self.coverage_hold_reward_claim_steps = 0
        self.coverage_hold_reward_steps_by_phase = {
            "pre_capture": 0,
            "post_capture": 0,
            "pure_coverage": 0,
        }
        self.post_capture_started = False
        self.post_capture_step = 0
        self.post_capture_coverage_success = False
        self.post_capture_coverage_step = None
        self.post_capture_grace_remaining = 0
        self.capture_snapshot = None
        self.coverage_geometric_success = False
        self.coverage_settled_success = False
        self.coverage_settle_timeout = False
        self.settle_hold_steps = 0
        self.coverage_settle_elapsed_steps = 0
        self.coverage_geometric_latch_mean_speed = None
        self.coverage_geometric_latch_max_speed = None
        self.coverage_geometric_to_settled_steps = None
        self.coverage_mean_score = 0.0
        self.coverage_max_score = 0.0
        self.coverage_speed_score = 0.0
        self.coverage_motion_gate = 0.0
        self.coverage_motion_score = 0.0
        self.coverage_motion_max_accel = 0.0
        self.coverage_motion_max_turn = 0.0
        self.coverage_motion_success_now = False
        self.last_voradj_metrics = {}
        self.last_task_labels = []
        self.last_raw_task_labels = []
        self.zone_evader_entered_inner = []
        self.zone_evader_exited_after_entry = []
        self.zone_evader_targets = []
        self.zone_evader_target_reached = []
        self.zone_pursuer_left_inner_event = False
        self._pursuing_release_counters = [0] * len(self.pursuers)
        self._pursuing_flags_initialized = False
        self._invalidate_voronoi_cache()
        super().reset(*args, **kwargs)
        self._zone_place_evaders_after_reset()
        self._zone_reset_episode_state()
        self._zone_update_metrics()
        self._invalidate_voronoi_cache()
        self._pursuing_release_counters = [0] * len(self.pursuers)
        data = self._capture_voronoi_map()
        self.last_task_labels = self._task_labels_from_map(data, update_effective=True)
        return self.get_observations()

    def _invalidate_voronoi_cache(self) -> None:
        self._voronoi_cache_version += 1
        self._voronoi_cache_version_tag = -1
        self._voronoi_cache = {}

    def _store_current_voronoi_map(
        self,
        data: Dict[str, Any],
        obstacle_mode: Optional[str] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        mode = str(
            obstacle_mode
            if obstacle_mode is not None
            else data.get("voronoi_obstacle_mode", self._voronoi_obstacle_mode())
        )
        self._voronoi_cache[str(cache_key if cache_key is not None else mode)] = data
        self._voronoi_cache_version_tag = self._voronoi_cache_version
        return data

    def _bounds(self) -> Tuple[float, float, float, float]:
        return (
            float(self.env_cfg.get("x_boundary_left", 0.0)),
            float(self.env_cfg.get("x_boundary_right", self.width)),
            float(self.env_cfg.get("y_boundary_bottom", 0.0)),
            float(self.env_cfg.get("y_boundary_top", self.height)),
        )

    def _ce_coverage_enabled(self) -> bool:
        return str(self.reward_cfg.get("coverage_objective_version", "legacy")) == "centroid_energy_v0"

    def _voronoi_obstacle_mode(self) -> str:
        return str((self.config.get("voradj", {}) or {}).get("voronoi_obstacle_mode", "legacy_assign"))

    def _capture_voronoi_obstacle_mode(self) -> str:
        cfg = self.config.get("voradj", {}) or {}
        return str(
            cfg.get(
                "capture_adjacency_obstacle_mode",
                self._voronoi_obstacle_mode(),
            )
        )

    def _perception_topology_version(self) -> str:
        return str(
            (self.config.get("voradj", {}) or {}).get(
                "perception_topology_version",
                "legacy_voradj",
            )
        ).strip().lower()

    def _vct_ls_enabled(self) -> bool:
        return self._perception_topology_version() in {
            "friendly_voronoi_comm_v0",
            "vct_ls_v0",
            "vct-ls-v0",
            "vct_ls",
        }

    def _vct_ls_apply_release_delay(self) -> bool:
        cfg = self.config.get("voradj", {}) or {}
        return bool(cfg.get("vct_ls_apply_release_delay", False))

    def _vct_ls_support_reward_blend_enabled(self) -> bool:
        cfg = self.config.get("voradj", {}) or {}
        return bool(
            self._vct_ls_enabled()
            and cfg.get(
                "support_reward_blend_enabled",
                cfg.get("vct_ls_support_reward_blend_enabled", False),
            )
        )

    def _vct_ls_support_reward_weights(self) -> Tuple[float, float]:
        cfg = self.config.get("voradj", {}) or {}
        capture_weight = float(
            cfg.get(
                "support_reward_capture_weight",
                cfg.get("vct_ls_support_reward_capture_weight", 0.5),
            )
        )
        coverage_weight = float(
            cfg.get(
                "support_reward_coverage_weight",
                cfg.get("vct_ls_support_reward_coverage_weight", 0.5),
            )
        )
        return capture_weight, coverage_weight

    def _support_pursuing_friend_ids(
        self,
        idx: int,
        labels: List[str],
        data: Dict[str, Any],
    ) -> List[int]:
        if idx < 0 or idx >= len(labels) or labels[idx] != "coverage":
            return []
        neighbor_ids = [
            int(nk[1])
            for nk in data.get("adjacency", {}).get(("pursuer", idx), set())
            if nk[0] == "pursuer" and 0 <= int(nk[1]) < len(labels) and labels[int(nk[1])] == "capture"
        ]
        return sorted(set(neighbor_ids))

    def _support_reward_capture_target_mode(self) -> str:
        cfg = self.config.get("voradj", {}) or {}
        return str(
            cfg.get(
                "support_reward_capture_target_mode",
                cfg.get("vct_ls_support_reward_capture_target_mode", "neighbor_visible"),
            )
        ).strip().lower()

    def _support_reward_capture_component_mode(self) -> str:
        """Return the dense capture component used only by support agents.

        ``capture_task`` preserves the original support-blend behavior. The
        ``approach_only`` option is intentionally separate from the direct
        pursuer's capture reward mode: a direct detector may keep CR-MS while a
        one-hop support agent receives only the legacy distance-progress term.
        """
        cfg = self.config.get("voradj", {}) or {}
        return str(
            cfg.get(
                "support_reward_capture_component_mode",
                cfg.get("vct_ls_support_reward_capture_component_mode", "capture_task"),
            )
        ).strip().lower()

    def _vct_ls_voronoi_obstacle_mode(self) -> str:
        cfg = self.config.get("voradj", {}) or {}
        return str(
            cfg.get(
                "friendly_comm_obstacle_mode",
                cfg.get("vct_ls_voronoi_obstacle_mode", cfg.get("voronoi_obstacle_mode", "legacy_assign")),
            )
        )

    def _vct_ls_sensing_radius(self, entity_type: str) -> float:
        cfg = self.config.get("voradj", {}) or {}
        if entity_type == "enemy":
            value = cfg.get("enemy_sensing_radius", cfg.get("vct_ls_enemy_sensing_radius", self.per_cfg.get("range", 20.0)))
        elif entity_type == "obstacle":
            value = cfg.get("obstacle_sensing_radius", cfg.get("vct_ls_obstacle_sensing_radius", self.per_cfg.get("range", 20.0)))
        else:
            value = self.per_cfg.get("range", 20.0)
        return max(float(value), 0.0)

    def _vct_ls_use_surface_distance(self) -> bool:
        cfg = self.config.get("voradj", {}) or {}
        return bool(cfg.get("local_sensing_uses_surface_distance", True))

    def _vct_ls_surface_clearance(
        self,
        observer_pos: np.ndarray,
        observer_radius: float,
        target_pos: np.ndarray,
        target_radius: float,
    ) -> float:
        center_distance = float(np.linalg.norm(np.asarray(observer_pos, dtype=float) - np.asarray(target_pos, dtype=float)))
        if not self._vct_ls_use_surface_distance():
            return center_distance
        return center_distance - float(observer_radius) - float(target_radius)

    def _vct_ls_direct_enemy_ids_for_pursuer(
        self,
        pursuer_idx: int,
        pursuer_positions: Optional[np.ndarray] = None,
        evader_positions: Optional[np.ndarray] = None,
    ) -> List[int]:
        if pursuer_idx < 0 or pursuer_idx >= len(self.pursuers):
            return []
        pursuer = self.pursuers[pursuer_idx]
        if pursuer.deactivated:
            return []
        if pursuer_positions is None:
            pursuer_positions = np.asarray([self._position(p) for p in self.pursuers], dtype=float)
        if evader_positions is None:
            evader_positions = np.asarray([self._position(e) for e in self.evaders], dtype=float) if self.evaders else np.zeros((0, 2), dtype=float)
        if bool(self.per_cfg.get("global_evader_visibility", False)):
            return [j for j, evader in enumerate(self.evaders) if not evader.deactivated]
        radius = self._vct_ls_sensing_radius("enemy")
        ids: List[int] = []
        ppos = np.asarray(pursuer_positions[pursuer_idx], dtype=float)
        for j, evader in enumerate(self.evaders):
            if evader.deactivated:
                continue
            epos = np.asarray(evader_positions[j], dtype=float)
            clearance = self._vct_ls_surface_clearance(ppos, float(getattr(pursuer, "r", 0.0)), epos, float(getattr(evader, "r", 0.0)))
            if clearance <= radius:
                ids.append(j)
        return ids

    def _vct_ls_direct_detect_counts(
        self,
        pursuer_positions: Optional[np.ndarray] = None,
        evader_positions: Optional[np.ndarray] = None,
    ) -> Dict[int, int]:
        counts = {j: 0 for j, evader in enumerate(self.evaders) if not evader.deactivated}
        for i, pursuer in enumerate(self.pursuers):
            if pursuer.deactivated:
                continue
            for j in self._vct_ls_direct_enemy_ids_for_pursuer(i, pursuer_positions, evader_positions):
                counts[j] = counts.get(j, 0) + 1
        return counts

    def _vct_ls_obstacle_visible_for_pursuer(self, pursuer_idx: int, obstacle_idx: int) -> Tuple[bool, float]:
        pursuer = self.pursuers[pursuer_idx]
        obstacle = self.obstacles[obstacle_idx]
        clearance = self._vct_ls_surface_clearance(
            self._position(pursuer),
            float(getattr(pursuer, "r", 0.0)),
            np.asarray([obstacle.x, obstacle.y], dtype=float),
            float(obstacle.r),
        )
        return bool(clearance <= self._vct_ls_sensing_radius("obstacle")), clearance

    def _masked_voronoi_grid_margin(self, grid_n: int) -> float:
        cfg = self.config.get("voradj", {}) or {}
        explicit = cfg.get("voronoi_obstacle_grid_margin")
        if explicit is not None:
            return max(float(explicit), 0.0)
        xl, xr, yb, yt = self._bounds()
        dx = (xr - xl) / max(grid_n - 1, 1)
        dy = (yt - yb) / max(grid_n - 1, 1)
        return 0.5 * float(np.hypot(dx, dy))

    def _map_diagonal(self) -> float:
        xl, xr, yb, yt = self._bounds()
        return max(float(np.hypot(xr - xl, yt - yb)), 1e-6)

    def _ce_center_distance_and_cost(
        self, position: np.ndarray, centroid: np.ndarray, participant_count: int
    ) -> Tuple[float, float]:
        distance = float(np.linalg.norm(np.asarray(position, dtype=float) - np.asarray(centroid, dtype=float)))
        normalized = float(np.sqrt(max(participant_count, 1)) * distance / self._map_diagonal())
        return normalized, normalized * normalized

    def _ce_terminal_pbrs_correction(self, center_cost: float) -> float:
        if not bool(self.reward_cfg.get("coverage_ce_pbrs_enabled", True)):
            return 0.0
        scale = float(self.reward_cfg.get("coverage_ce_reward_scale", 10.0))
        gamma = float(
            (self.config.get("discount", {}) or {}).get(
                "gamma",
                (self.config.get("iqn", {}) or {}).get("gamma", 0.99),
            )
        )
        kappa = float(self.reward_cfg.get("coverage_ce_pbrs_kappa", 1.0))
        return scale * gamma * kappa * float(center_cost)

    def _ce_pbrs_reset_mode(self) -> str:
        return str(
            self.reward_cfg.get(
                "coverage_ce_pbrs_reset_mode",
                "success_only_legacy",
            )
        ).strip().lower()

    def _zone_config(self) -> Dict[str, Any]:
        return self.config.get("zone_demo", {}) or {}

    def _zone_enabled(self) -> bool:
        return bool(self._zone_config().get("enabled", False))

    def _zone_inner_bounds(self) -> Tuple[float, float, float, float]:
        cfg = self._zone_config()
        raw = cfg.get("inner_bounds")
        if raw is not None:
            return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        return self._bounds()

    def _zone_outer_bounds(self) -> Tuple[float, float, float, float]:
        cfg = self._zone_config()
        raw = cfg.get("outer_bounds")
        if raw is not None:
            return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        xl, xr, yb, yt = self._zone_inner_bounds()
        margin = float(cfg.get("outer_margin", 20.0))
        return (xl - margin, xr + margin, yb - margin, yt + margin)

    @staticmethod
    def _point_in_rect(point: np.ndarray, bounds: Tuple[float, float, float, float], pad: float = 0.0) -> bool:
        xl, xr, yb, yt = bounds
        return bool((xl - pad) <= float(point[0]) <= (xr + pad) and (yb - pad) <= float(point[1]) <= (yt + pad))

    def _zone_point_in_inner(self, point: np.ndarray, pad: float = 0.0) -> bool:
        return self._point_in_rect(np.asarray(point, dtype=float), self._zone_inner_bounds(), pad=pad)

    def _zone_sample_outer_ring_position(self, radius: float = 0.0) -> np.ndarray:
        xl, xr, yb, yt = self._zone_inner_bounds()
        oxl, oxr, oyb, oyt = self._zone_outer_bounds()
        bands = [
            (oxl + radius, xl - radius, oyb + radius, oyt - radius),
            (xr + radius, oxr - radius, oyb + radius, oyt - radius),
            (xl + radius, xr - radius, oyb + radius, yb - radius),
            (xl + radius, xr - radius, yt + radius, oyt - radius),
        ]
        valid = [band for band in bands if band[0] <= band[1] and band[2] <= band[3]]
        if not valid:
            return self.rng.uniform([oxl, oyb], [oxr, oyt]).astype(float)
        band = valid[int(self.rng.randint(0, len(valid)))]
        return self.rng.uniform([band[0], band[2]], [band[1], band[3]]).astype(float)

    def _zone_sample_inner_goal(self) -> np.ndarray:
        xl, xr, yb, yt = self._zone_inner_bounds()
        center = np.asarray([(xl + xr) * 0.5, (yb + yt) * 0.5], dtype=float)
        radius = max(float(self._zone_config().get("evader_goal_radius", 6.0)), 0.0)
        theta = float(self.rng.uniform(0.0, TWO_PI))
        rr = radius * float(np.sqrt(self.rng.uniform(0.0, 1.0)))
        return center + rr * np.asarray([np.cos(theta), np.sin(theta)], dtype=float)

    def _zone_place_evaders_after_reset(self) -> None:
        if not self._zone_enabled():
            return
        mode = str(self._zone_config().get("evader_spawn_mode", self.env_cfg.get("evader_spawn_mode", "map_random"))).strip().lower()
        if mode not in {"outer_ring_random", "zone_outer_ring_random"}:
            return
        center = np.asarray([(self._zone_inner_bounds()[0] + self._zone_inner_bounds()[1]) * 0.5, (self._zone_inner_bounds()[2] + self._zone_inner_bounds()[3]) * 0.5], dtype=float)
        occupied = [(self._position(p), float(getattr(p, "r", 1.0))) for p in self.pursuers if not p.deactivated]
        occupied += [(np.asarray([o.x, o.y], dtype=float), float(o.r)) for o in self.obstacles]
        min_sep = float(self.env_cfg.get("evader_spawn_min_sep", 8.0))
        for evader in self.evaders:
            pos = self._zone_sample_outer_ring_position(float(getattr(evader, "r", 1.0)) + 0.5)
            for _attempt in range(4000):
                candidate = self._zone_sample_outer_ring_position(float(getattr(evader, "r", 1.0)) + 0.5)
                if all(float(np.linalg.norm(candidate - other)) >= min_sep + er + rr for other, rr in occupied for er in [float(getattr(evader, "r", 1.0))]):
                    pos = candidate
                    break
            theta = float(np.arctan2(center[1] - pos[1], center[0] - pos[0]))
            self._reset_robot(evader, pos, theta=theta)
            occupied.append((pos, float(getattr(evader, "r", 1.0))))

    def _zone_reset_episode_state(self) -> None:
        if not self._zone_enabled():
            return
        self.zone_evader_entered_inner = [False] * len(self.evaders)
        self.zone_evader_exited_after_entry = [False] * len(self.evaders)
        self.zone_evader_target_reached = [False] * len(self.evaders)
        goal_enabled = bool(self._zone_config().get("evader_goal_enabled", True))
        self.zone_evader_targets = [self._zone_sample_inner_goal() if goal_enabled else None for _ in self.evaders]

    def _zone_maybe_update_evader_target(self, idx: int) -> None:
        if not self._zone_enabled() or idx >= len(self.evaders) or idx >= len(self.zone_evader_targets):
            return
        target = self.zone_evader_targets[idx]
        if target is None:
            return
        reached_radius = float(self._zone_config().get("evader_goal_reached_radius", 4.0))
        if float(np.linalg.norm(self._position(self.evaders[idx]) - np.asarray(target, dtype=float))) > reached_radius:
            return
        self.zone_evader_target_reached[idx] = True
        mode = str(self._zone_config().get("evader_goal_after_reach", "pure_apf")).strip().lower()
        self.zone_evader_targets[idx] = None if mode == "pure_apf" else self._zone_sample_outer_ring_position(1.0)

    def configure_evader_apf_agents(self, apf_agents: List[Any]) -> None:
        if not self._zone_enabled():
            for agent in apf_agents:
                if hasattr(agent, "clear_goal"):
                    agent.clear_goal()
                if hasattr(agent, "set_boundary_enabled"):
                    agent.set_boundary_enabled(True)
            return
        weight = float(self._zone_config().get("evader_goal_weight", self.config.get("apf", {}).get("goal_attraction_k", 0.0)))
        disable_boundary_all = bool(self._zone_config().get("evader_disable_boundary_apf", True))
        disable_boundary_outside = bool(self._zone_config().get("evader_disable_boundary_apf_outside_inner", False))
        for idx, agent in enumerate(apf_agents):
            if idx >= len(self.evaders) or self.evaders[idx].deactivated:
                if hasattr(agent, "clear_goal"):
                    agent.clear_goal()
                if hasattr(agent, "set_boundary_enabled"):
                    agent.set_boundary_enabled(True)
                continue
            self._zone_maybe_update_evader_target(idx)
            target = self.zone_evader_targets[idx] if idx < len(self.zone_evader_targets) else None
            if hasattr(agent, "set_goal"):
                agent.set_goal(target, weight if target is not None else None)
            if hasattr(agent, "set_boundary_enabled"):
                outside_inner = not self._zone_point_in_inner(self._position(self.evaders[idx]))
                agent.set_boundary_enabled(not (disable_boundary_all or (disable_boundary_outside and outside_inner)))

    def _zone_update_metrics(self) -> None:
        if not self._zone_enabled():
            return
        if len(self.zone_evader_entered_inner) != len(self.evaders):
            self._zone_reset_episode_state()
        for idx, evader in enumerate(self.evaders):
            pos = self._position(evader)
            inside = self._zone_point_in_inner(pos)
            if inside:
                self.zone_evader_entered_inner[idx] = True
            elif self.zone_evader_entered_inner[idx] and not evader.deactivated:
                self.zone_evader_exited_after_entry[idx] = True
        if any(not self._zone_point_in_inner(self._position(p)) for p in self.pursuers if not p.deactivated):
            self.zone_pursuer_left_inner_event = True

    def _zone_extra_evader_ids_for_pursuer(self, pursuer_idx: int, data: Optional[Dict[str, Any]] = None) -> List[int]:
        if not self._zone_enabled() or pursuer_idx < 0 or pursuer_idx >= len(self.pursuers):
            return []
        radius = float(self._zone_config().get("outer_evader_detection_radius", self.per_cfg.get("range", 20.0)))
        if radius <= 0.0 or self.pursuers[pursuer_idx].deactivated:
            return []
        ppos = self._position(self.pursuers[pursuer_idx])
        ids: List[int] = []
        for idx, evader in enumerate(self.evaders):
            if evader.deactivated:
                continue
            epos = self._position(evader)
            if self._zone_point_in_inner(epos):
                continue
            if float(np.linalg.norm(epos - ppos)) <= radius + float(getattr(evader, "r", 0.0)):
                ids.append(idx)
        return ids

    def _zone_metrics(self) -> Dict[str, Any]:
        if not self._zone_enabled():
            return {"zone_demo_enabled": False}
        evader_inside = [bool((not e.deactivated) and self._zone_point_in_inner(self._position(e))) for e in self.evaders]
        pursuer_inside = [bool(p.deactivated or self._zone_point_in_inner(self._position(p))) for p in self.pursuers]
        targets = [None if target is None else [float(target[0]), float(target[1])] for target in self.zone_evader_targets]
        target_reached_event = bool(any(self.zone_evader_target_reached))
        exit_after_entry_event = bool(any(self.zone_evader_exited_after_entry))
        breach_definition = str(self._zone_config().get("breach_definition", "target_reached")).strip().lower()
        if breach_definition == "exit_after_entry":
            breach_event = exit_after_entry_event
        elif breach_definition in {"target_reached_or_exit", "target_or_exit", "either"}:
            breach_event = bool(target_reached_event or exit_after_entry_event)
        else:
            breach_event = target_reached_event
        return {
            "zone_demo_enabled": True,
            "zone_breach_definition": breach_definition,
            "zone_inner_bounds": [float(v) for v in self._zone_inner_bounds()],
            "zone_outer_bounds": [float(v) for v in self._zone_outer_bounds()],
            "zone_evader_inside_inner": evader_inside,
            "zone_evader_entered_inner": [bool(v) for v in self.zone_evader_entered_inner],
            "zone_evader_exited_after_entry": [bool(v) for v in self.zone_evader_exited_after_entry],
            "zone_exit_after_entry_event": exit_after_entry_event,
            "zone_any_evader_entered_inner": bool(any(self.zone_evader_entered_inner)),
            "zone_breach_event": breach_event,
            "zone_all_pursuers_inside_inner": bool(all(pursuer_inside)),
            "zone_pursuer_left_inner_event": bool(self.zone_pursuer_left_inner_event),
            "zone_evader_targets": targets,
            "zone_evader_target_reached": [bool(v) for v in self.zone_evader_target_reached],
            "zone_target_reached_event": target_reached_event,
            "zone_outer_evader_detection_radius": float(self._zone_config().get("outer_evader_detection_radius", self.per_cfg.get("range", 20.0))),
            "zone_evader_disable_boundary_apf": bool(self._zone_config().get("evader_disable_boundary_apf", True)),
        }

    def _distance_scale(self) -> float:
        if not bool(self.per_cfg.get("normalize_distances", False)):
            return 1.0
        xl, xr, yb, yt = self._bounds()
        return max(float(np.hypot(xr - xl, yt - yb)), 1e-6)

    def _center_sqrt_n_scale(self, data: Optional[Dict[str, Any]] = None) -> float:
        if not self._center_sqrt_n_normalization_enabled:
            return 1.0
        if data is not None:
            n_active = sum(1 for key in data.get("keys", []) if key[0] == "pursuer")
        else:
            n_active = sum(1 for p in self.pursuers if not p.deactivated)
        return float(np.sqrt(max(n_active, 1)))

    def _nearest_boundary_vector(self, pursuer) -> Tuple[np.ndarray, bool]:
        xl, xr, yb, yt = self._bounds()
        position = self._position(pursuer)
        inside = bool(xl <= pursuer.x <= xr and yb <= pursuer.y <= yt)
        if inside:
            boundary_points = [
                np.asarray([xl, pursuer.y], dtype=float),
                np.asarray([xr, pursuer.y], dtype=float),
                np.asarray([pursuer.x, yb], dtype=float),
                np.asarray([pursuer.x, yt], dtype=float),
            ]
            nearest = min(boundary_points, key=lambda point: float(np.linalg.norm(point - position)))
        else:
            nearest = np.asarray(
                [np.clip(pursuer.x, xl, xr), np.clip(pursuer.y, yb, yt)],
                dtype=float,
            )
        return self._robot_frame(pursuer, nearest - position, True), not inside

    def _active_sites(
        self,
        pursuer_positions: Optional[np.ndarray] = None,
        evader_positions: Optional[np.ndarray] = None,
        include_evaders: bool = True,
    ) -> Tuple[List[SiteKey], np.ndarray]:
        keys: List[SiteKey] = []
        pts: List[np.ndarray] = []
        if pursuer_positions is None:
            pursuer_positions = np.asarray([self._position(p) for p in self.pursuers], dtype=float)
        if evader_positions is None:
            evader_positions = np.asarray([self._position(e) for e in self.evaders], dtype=float) if self.evaders else np.zeros((0, 2), dtype=float)
        for i, p in enumerate(self.pursuers):
            if not p.deactivated:
                keys.append(("pursuer", i))
                pts.append(np.asarray(pursuer_positions[i], dtype=float))
        if include_evaders:
            for j, e in enumerate(self.evaders):
                if not e.deactivated:
                    if self._zone_enabled() and not self._zone_point_in_inner(np.asarray(evader_positions[j], dtype=float)):
                        continue
                    keys.append(("evader", j))
                    pts.append(np.asarray(evader_positions[j], dtype=float))
        if not pts:
            return keys, np.zeros((0, 2), dtype=float)
        return keys, np.asarray(pts, dtype=float)

    def _voronoi_map(
        self,
        pursuer_positions: Optional[np.ndarray] = None,
        evader_positions: Optional[np.ndarray] = None,
        obstacle_mode: Optional[str] = None,
        include_evaders: bool = True,
    ) -> Dict[str, Any]:
        use_state_cache = pursuer_positions is None and evader_positions is None
        mode = str(
            self._voronoi_obstacle_mode()
            if obstacle_mode is None
            else obstacle_mode
        )
        site_scope = "all_active_entities" if include_evaders else "pursuer_only"
        cache_key = mode if include_evaders else f"{mode}|{site_scope}"
        if (
            use_state_cache
            and self._voronoi_cache_version_tag == self._voronoi_cache_version
            and cache_key in self._voronoi_cache
        ):
            return self._voronoi_cache[cache_key]
        keys, sites = self._active_sites(pursuer_positions, evader_positions, include_evaders=include_evaders)
        grid_n = int(self.reward_cfg.get("voradj_grid_size", self.reward_cfg.get("distribution_voronoi_grid_size", 60)))
        grid_n = max(grid_n, 8)
        xl, xr, yb, yt = self._bounds()
        xs = np.linspace(xl, xr, grid_n)
        ys = np.linspace(yb, yt, grid_n)
        mesh_x, mesh_y = np.meshgrid(xs, ys)
        grid_points = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)
        if mode == "free_mask_projected":
            obstacles = [(float(o.x), float(o.y), float(o.r)) for o in self.obstacles]
            active_radii = [float(p.r) for p in self.pursuers if not p.deactivated]
            robot_radius = max(active_radii) if active_radii else 0.0
            result = build_masked_voronoi_map(
                keys=keys,
                sites=sites,
                grid_points=grid_points,
                grid_n=grid_n,
                obstacles=obstacles,
                robot_radius=robot_radius,
                grid_margin=self._masked_voronoi_grid_margin(grid_n),
            )
            result = dict(result)
            result["site_scope"] = site_scope
            result["include_evaders_as_sites"] = bool(include_evaders)
            return self._store_current_voronoi_map(result, mode, cache_key=cache_key) if use_state_cache else result
        if len(keys) == 0:
            owners = np.zeros(len(grid_points), dtype=np.int64)
            owner_grid = owners.reshape(grid_n, grid_n)
            result = {
                "keys": keys,
                "sites": sites,
                "points": grid_points,
                "owners": owners,
                "owner_grid": owner_grid,
                "counts": {},
                "centroids": {},
                "adjacency": {},
                "shared_counts": {},
                "grid_n": grid_n,
                "voronoi_obstacle_mode": mode,
                "site_scope": site_scope,
                "include_evaders_as_sites": bool(include_evaders),
            }
            return self._store_current_voronoi_map(result, mode, cache_key=cache_key) if use_state_cache else result
        dist_sq = np.sum((grid_points[:, None, :] - sites[None, :, :]) ** 2, axis=2)
        owners = np.argmin(dist_sq, axis=1).astype(np.int64)
        owner_grid = owners.reshape(grid_n, grid_n)
        counts: Dict[SiteKey, int] = {}
        centroids: Dict[SiteKey, np.ndarray] = {}
        for site_id, key in enumerate(keys):
            owned = grid_points[owners == site_id]
            counts[key] = int(len(owned))
            centroids[key] = owned.mean(axis=0) if len(owned) else sites[site_id].copy()
        adjacency: Dict[SiteKey, set[SiteKey]] = {key: set() for key in keys}
        shared_counts: Dict[Tuple[SiteKey, SiteKey], int] = {}

        def add_edges(a: np.ndarray, b: np.ndarray) -> None:
            diff = a != b
            for sid_a, sid_b in zip(a[diff].ravel(), b[diff].ravel()):
                key_a = keys[int(sid_a)]
                key_b = keys[int(sid_b)]
                if key_a == key_b:
                    continue
                adjacency[key_a].add(key_b)
                adjacency[key_b].add(key_a)
                pair = tuple(sorted((key_a, key_b)))  # type: ignore[arg-type]
                shared_counts[pair] = shared_counts.get(pair, 0) + 1

        add_edges(owner_grid[:, :-1], owner_grid[:, 1:])
        add_edges(owner_grid[:-1, :], owner_grid[1:, :])
        result = {
            "keys": keys,
            "sites": sites,
            "points": grid_points,
            "owners": owners,
            "owner_grid": owner_grid,
            "counts": counts,
            "centroids": centroids,
            "adjacency": adjacency,
            "shared_counts": shared_counts,
            "grid_n": grid_n,
            "voronoi_obstacle_mode": mode,
            "site_scope": site_scope,
            "include_evaders_as_sites": bool(include_evaders),
        }
        return self._store_current_voronoi_map(result, mode, cache_key=cache_key) if use_state_cache else result

    def _capture_voronoi_map(
        self,
        pursuer_positions: Optional[np.ndarray] = None,
        evader_positions: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if self._vct_ls_enabled():
            return self._voronoi_map(
                pursuer_positions,
                evader_positions,
                obstacle_mode=self._vct_ls_voronoi_obstacle_mode(),
                include_evaders=False,
            )
        return self._voronoi_map(
            pursuer_positions,
            evader_positions,
            obstacle_mode=self._capture_voronoi_obstacle_mode(),
        )

    def _coverage_voronoi_map(
        self,
        pursuer_positions: Optional[np.ndarray] = None,
        evader_positions: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        return self._voronoi_map(
            pursuer_positions,
            evader_positions,
            obstacle_mode=self._vct_ls_voronoi_obstacle_mode() if self._vct_ls_enabled() else self._voronoi_obstacle_mode(),
            include_evaders=not self._vct_ls_enabled(),
        )

    def _shared_count(self, data: Dict[str, Any], a: SiteKey, b: SiteKey) -> int:
        pair = tuple(sorted((a, b)))  # type: ignore[arg-type]
        return int(data.get("shared_counts", {}).get(pair, 0))

    def _has_enemy_neighbor(self, data: Dict[str, Any], key: SiteKey) -> bool:
        if self._vct_ls_enabled():
            return bool(key[0] == "pursuer" and self._vct_ls_direct_enemy_ids_for_pursuer(int(key[1])))
        if key[0] == "pursuer" and self._zone_extra_evader_ids_for_pursuer(int(key[1]), data):
            return True
        return any(nk[0] == "evader" for nk in data.get("adjacency", {}).get(key, set()))

    def _effective_is_pursuing(self, idx: int, raw_is_pursuing: Optional[bool] = None) -> bool:
        if raw_is_pursuing is None:
            data = self._capture_voronoi_map()
            raw_is_pursuing = self._has_enemy_neighbor(data, ("pursuer", idx))
        if self._vct_ls_enabled() and not self._vct_ls_apply_release_delay():
            return bool(raw_is_pursuing)
        if self._pursuing_release_delay_steps <= 0 or not self._pursuing_flags_initialized:
            return bool(raw_is_pursuing)
        return bool(getattr(self.pursuers[idx], "is_pursuing", bool(raw_is_pursuing)))

    def _update_effective_pursuing_flags(self, raw_labels: List[str]) -> None:
        if len(self._pursuing_release_counters) != len(self.pursuers):
            self._pursuing_release_counters = [0] * len(self.pursuers)
        delay = 0 if (self._vct_ls_enabled() and not self._vct_ls_apply_release_delay()) else int(self._pursuing_release_delay_steps)
        for i, p in enumerate(self.pursuers):
            raw_capture = bool(i < len(raw_labels) and raw_labels[i] == "capture")
            if p.deactivated:
                p.is_pursuing = False
                self._pursuing_release_counters[i] = 0
            elif delay <= 0:
                p.is_pursuing = raw_capture
                self._pursuing_release_counters[i] = 0
            elif raw_capture:
                p.is_pursuing = True
                self._pursuing_release_counters[i] = delay
            elif bool(getattr(p, "is_pursuing", False)) and self._pursuing_release_counters[i] > 1:
                p.is_pursuing = True
                self._pursuing_release_counters[i] -= 1
            elif bool(getattr(p, "is_pursuing", False)) and self._pursuing_release_counters[i] == 1:
                p.is_pursuing = False
                self._pursuing_release_counters[i] = 0
            else:
                p.is_pursuing = False
                self._pursuing_release_counters[i] = 0
        self._pursuing_flags_initialized = True

    def _clear_release_delay_after_single_enemy_capture(
        self,
        data: Dict[str, Any],
        captured_evader_ids: set[int],
    ) -> List[int]:
        if self._pursuing_release_delay_steps <= 0 or not captured_evader_ids:
            return []
        cleared: List[int] = []
        if self.evaders and all(evader.deactivated for evader in self.evaders):
            for i, pursuer in enumerate(self.pursuers):
                if pursuer.deactivated:
                    continue
                pursuer.is_pursuing = False
                self._pursuing_release_counters[i] = 0
                cleared.append(i)
            return cleared
        adjacency_by_key = data.get("adjacency", {})
        for i, pursuer in enumerate(self.pursuers):
            if pursuer.deactivated:
                continue
            adjacency = adjacency_by_key.get(("pursuer", i), set())
            enemy_neighbors = [int(nk[1]) for nk in adjacency if nk[0] == "evader"]
            if len(enemy_neighbors) == 1 and enemy_neighbors[0] in captured_evader_ids:
                pursuer.is_pursuing = False
                self._pursuing_release_counters[i] = 0
                cleared.append(i)
        return cleared

    def _raw_task_labels_from_map(self, data: Dict[str, Any]) -> List[str]:
        labels = ["inactive"] * len(self.pursuers)
        for i, p in enumerate(self.pursuers):
            if p.deactivated:
                continue
            key = ("pursuer", i)
            labels[i] = "capture" if self._has_enemy_neighbor(data, key) else "coverage"
        return labels

    def _effective_task_labels(self, raw_labels: Optional[List[str]] = None) -> List[str]:
        if raw_labels is None:
            raw_labels = self._raw_task_labels_from_map(self._capture_voronoi_map())
        labels = ["inactive"] * len(self.pursuers)
        for i, p in enumerate(self.pursuers):
            if p.deactivated:
                continue
            raw_capture = bool(i < len(raw_labels) and raw_labels[i] == "capture")
            if (self._vct_ls_enabled() and not self._vct_ls_apply_release_delay()) or self._pursuing_release_delay_steps <= 0 or not self._pursuing_flags_initialized:
                labels[i] = "capture" if raw_capture else "coverage"
            else:
                labels[i] = "capture" if bool(getattr(p, "is_pursuing", raw_capture)) else "coverage"
        return labels

    def _task_labels_from_map(
        self,
        data: Dict[str, Any],
        update_effective: bool = True,
        raw_labels: Optional[List[str]] = None,
    ) -> List[str]:
        raw = raw_labels if raw_labels is not None else self._raw_task_labels_from_map(data)
        self.last_raw_task_labels = list(raw)
        if update_effective:
            self._update_effective_pursuing_flags(raw)
        return self._effective_task_labels(raw)

    def _pack_agent_obs(self, idx: int) -> Optional[Dict[str, np.ndarray]]:
        pursuer = self.pursuers[idx]
        if pursuer.deactivated:
            return None
        max_p = int(self.per_cfg.get("max_pursuer_num", 8))
        max_e = int(self.per_cfg.get("max_evader_num", 8))
        max_o = int(self.per_cfg.get("max_obstacle_num", 5))
        data = self._capture_voronoi_map()
        coverage_data = self._coverage_voronoi_map()
        world_frame = str(self.per_cfg.get("observation_frame", "robot")).strip().lower() in {"world", "world_frame"}
        key = ("pursuer", idx)
        adjacency = data.get("adjacency", {}).get(key, set())
        raw_is_pursuing = self._has_enemy_neighbor(data, key)
        is_pursuing = self._effective_is_pursuing(idx, raw_is_pursuing)

        distance_scale = self._distance_scale()
        abs_vel = np.asarray(pursuer.velocity, dtype=float) if world_frame else self._robot_frame(pursuer, pursuer.velocity, True)
        if self.obstacles:
            if self._vct_ls_enabled():
                visible_clearances = [
                    clearance
                    for oi in range(len(self.obstacles))
                    for visible, clearance in [self._vct_ls_obstacle_visible_for_pursuer(idx, oi)]
                    if visible
                ]
                min_obs = min(visible_clearances) if visible_clearances else self._vct_ls_sensing_radius("obstacle")
            else:
                min_obs = min(float(np.linalg.norm(self._position(pursuer) - np.array([o.x, o.y], dtype=float))) for o in self.obstacles)
        else:
            min_obs = self._vct_ls_sensing_radius("obstacle") if self._vct_ls_enabled() else float(self.per_cfg.get("range", 20.0))
        xl, xr, yb, yt = self._bounds()
        left_d = float(pursuer.x - xl)
        right_d = float(pursuer.x - xr)
        nearest_x = left_d if abs(left_d) <= abs(right_d) else right_d
        bottom_d = float(pursuer.y - yb)
        top_d = float(pursuer.y - yt)
        nearest_y = bottom_d if abs(bottom_d) <= abs(top_d) else top_d
        centroid = coverage_data.get("centroids", {}).get(key, self._position(pursuer))
        center_vec = np.asarray(centroid, dtype=float) - self._position(pursuer)
        center_r = center_vec if world_frame else self._robot_frame(pursuer, center_vec, True)
        center_scale = self._center_sqrt_n_scale(coverage_data)
        boundary_mode = str(self.per_cfg.get("boundary_feature_mode", "axis_signed"))
        if boundary_mode == "nearest_vector_robot_oob":
            if world_frame:
                xl, xr, yb, yt = self._bounds()
                position = self._position(pursuer)
                inside = bool(xl <= pursuer.x <= xr and yb <= pursuer.y <= yt)
                if inside:
                    boundary_points = [
                        np.asarray([xl, pursuer.y], dtype=float),
                        np.asarray([xr, pursuer.y], dtype=float),
                        np.asarray([pursuer.x, yb], dtype=float),
                        np.asarray([pursuer.x, yt], dtype=float),
                    ]
                    nearest = min(boundary_points, key=lambda point: float(np.linalg.norm(point - position)))
                else:
                    nearest = np.asarray(
                        [np.clip(pursuer.x, xl, xr), np.clip(pursuer.y, yb, yt)],
                        dtype=float,
                    )
                boundary_r = nearest - position
                is_out_of_bounds = not inside
                self_feat = [
                    float(abs_vel[0]),
                    float(abs_vel[1]),
                    float(min_obs / distance_scale),
                    float(boundary_r[0] / distance_scale),
                    float(boundary_r[1] / distance_scale),
                    float(is_out_of_bounds),
                    float((center_r[0] / distance_scale) * center_scale),
                    float((center_r[1] / distance_scale) * center_scale),
                    float(is_pursuing),
                    float(np.cos(pursuer.theta)),
                    float(np.sin(pursuer.theta)),
                ]
            else:
                boundary_r, is_out_of_bounds = self._nearest_boundary_vector(pursuer)
                self_feat = [
                    float(abs_vel[0]),
                    float(abs_vel[1]),
                    float(min_obs / distance_scale),
                    float(boundary_r[0] / distance_scale),
                    float(boundary_r[1] / distance_scale),
                    float(is_out_of_bounds),
                    float((center_r[0] / distance_scale) * center_scale),
                    float((center_r[1] / distance_scale) * center_scale),
                    float(is_pursuing),
                ]
                if bool(self.per_cfg.get("include_yaw_features", False)):
                    self_feat += [float(np.cos(pursuer.theta)), float(np.sin(pursuer.theta))]
        else:
            self_feat = [
                float(abs_vel[0]),
                float(abs_vel[1]),
                float(min_obs / distance_scale),
                float(nearest_x / distance_scale),
                float(nearest_y / distance_scale),
                float((center_r[0] / distance_scale) * center_scale),
                float((center_r[1] / distance_scale) * center_scale),
                float(is_pursuing),
            ]

        friend_ids = [nk[1] for nk in adjacency if nk[0] == "pursuer" and not self.pursuers[nk[1]].deactivated]
        def friend_sort(j: int) -> Tuple[int, int, float]:
            other_key = ("pursuer", j)
            return (
                0 if self._effective_is_pursuing(j, self._has_enemy_neighbor(data, other_key)) else 1,
                -self._shared_count(data, key, other_key),
                float(np.linalg.norm(self._position(self.pursuers[j]) - self._position(pursuer))),
            )
        friend_ids = sorted(friend_ids, key=friend_sort)
        pursuer_feats: List[List[float]] = []
        for j in friend_ids[:max_p]:
            other = self.pursuers[j]
            if world_frame:
                pos_r = self._position(other) - self._position(pursuer)
                vel_r = np.asarray(other.velocity, dtype=float)
            else:
                pos_r = self._robot_frame(pursuer, self._position(other), False)
                vel_r = self._robot_frame(pursuer, other.velocity, True)
            dist = float(np.linalg.norm(pos_r))
            ang = float(np.arctan2(pos_r[1], pos_r[0]))
            pursuer_feats.append([
                float(pos_r[0] / distance_scale),
                float(pos_r[1] / distance_scale),
                float(vel_r[0]),
                float(vel_r[1]),
                float(dist / distance_scale),
                ang,
                float(self._effective_is_pursuing(j, self._has_enemy_neighbor(data, ("pursuer", j)))),
            ])

        if self._vct_ls_enabled():
            enemy_ids = sorted(
                self._vct_ls_direct_enemy_ids_for_pursuer(idx),
                key=lambda j: float(np.linalg.norm(self._position(self.evaders[j]) - self._position(pursuer))),
            )
        else:
            enemy_id_set = {nk[1] for nk in adjacency if nk[0] == "evader" and not self.evaders[nk[1]].deactivated}
            enemy_id_set.update(self._zone_extra_evader_ids_for_pursuer(idx, data))
            enemy_ids = sorted(
                enemy_id_set,
                key=lambda j: (
                    -self._shared_count(data, key, ("evader", j)),
                    float(np.linalg.norm(self._position(self.evaders[j]) - self._position(pursuer))),
                ),
            )
        evader_feats: List[List[float]] = []
        for j in enemy_ids[:max_e]:
            evader = self.evaders[j]
            if world_frame:
                pos_r = self._position(evader) - self._position(pursuer)
                vel_r = np.asarray(evader.velocity, dtype=float)
            else:
                pos_r = self._robot_frame(pursuer, self._position(evader), False)
                vel_r = self._robot_frame(pursuer, evader.velocity, True)
            dist = float(np.linalg.norm(pos_r))
            ang = float(np.arctan2(pos_r[1], pos_r[0]))
            heading = float(np.arctan2(vel_r[1], vel_r[0])) if np.linalg.norm(vel_r) > 1e-9 else 0.0
            evader_feats.append([
                float(pos_r[0] / distance_scale),
                float(pos_r[1] / distance_scale),
                float(vel_r[0]),
                float(vel_r[1]),
                float(dist / distance_scale),
                ang,
                heading,
            ])

        obstacle_candidates: Dict[int, Tuple[int, float]] = {}
        points = coverage_data.get("points", np.zeros((0, 2), dtype=float))
        owners = coverage_data.get("owners", np.zeros((0,), dtype=np.int64))
        site_idx = coverage_data.get("keys", []).index(key) if key in coverage_data.get("keys", []) else -1
        owned_points = points[owners == site_idx] if site_idx >= 0 and len(points) else np.zeros((0, 2), dtype=float)
        safety_range = float(self.per_cfg.get("range", 20.0))
        for oi, obs in enumerate(self.obstacles):
            center = np.array([obs.x, obs.y], dtype=float)
            dist_center = float(np.linalg.norm(self._position(pursuer) - center))
            if self._vct_ls_enabled():
                visible, clearance = self._vct_ls_obstacle_visible_for_pursuer(idx, oi)
                if visible:
                    obstacle_candidates[oi] = (0, clearance)
                continue
            if coverage_data.get("voronoi_obstacle_mode") == "free_mask_projected":
                intersects_cell = key in coverage_data.get("obstacle_adjacent_sites", {}).get(oi, set())
            else:
                intersects_cell = bool(len(owned_points) and np.any(np.linalg.norm(owned_points - center, axis=1) <= obs.r))
            near_safety = bool(dist_center <= safety_range + obs.r)
            if intersects_cell or near_safety:
                priority = 0 if near_safety else 1
                obstacle_candidates[oi] = (priority, dist_center)
        obstacle_feats: List[List[float]] = []
        for oi, _meta in sorted(obstacle_candidates.items(), key=lambda kv: (kv[1][0], kv[1][1]))[:max_o]:
            obs = self.obstacles[oi]
            if world_frame:
                pos_r = np.asarray([obs.x, obs.y], dtype=float) - self._position(pursuer)
            else:
                pos_r = self._robot_frame(pursuer, np.array([obs.x, obs.y], dtype=float), False)
            dist = float(np.linalg.norm(pos_r))
            ang = float(np.arctan2(pos_r[1], pos_r[0]))
            obstacle_feats.append([
                float(pos_r[0] / distance_scale),
                float(pos_r[1] / distance_scale),
                float(obs.r / distance_scale),
                float(dist / distance_scale),
                ang,
            ])

        masks = [True]
        masks += [True] * min(len(pursuer_feats), max_p) + [False] * max(0, max_p - len(pursuer_feats))
        masks += [True] * min(len(evader_feats), max_e) + [False] * max(0, max_e - len(evader_feats))
        masks += [True] * min(len(obstacle_feats), max_o) + [False] * max(0, max_o - len(obstacle_feats))
        types = [0] + [1] * max_p + [2] * max_e + [3] * max_o
        return {
            "self": np.asarray(self_feat, dtype=np.float32),
            "pursuers": self._pad(pursuer_feats, max_p, 7),
            "evaders": self._pad(evader_feats, max_e, 7),
            "obstacles": self._pad(obstacle_feats, max_o, 5),
            "masks": np.asarray(masks, dtype=bool),
            "types": np.asarray(types, dtype=np.int64),
        }

    def _voradj_coverage_potentials(self, pursuer_positions: np.ndarray, evader_positions: np.ndarray, data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        values = np.zeros(len(self.pursuers), dtype=float)
        data = data if data is not None else self._coverage_voronoi_map(pursuer_positions, evader_positions)
        active = [i for i, p in enumerate(self.pursuers) if not p.deactivated]
        if not active:
            return values
        if self._ce_coverage_enabled():
            for i in active:
                key = ("pursuer", i)
                centroid = data.get("centroids", {}).get(key, pursuer_positions[i])
                _distance, cost = self._ce_center_distance_and_cost(pursuer_positions[i], centroid, len(active))
                values[i] = cost
            return values
        pursuer_counts = np.asarray([data["counts"].get(("pursuer", i), 0) for i in active], dtype=float)
        expected = float(np.sum(pursuer_counts) / max(len(active), 1))
        xl, xr, yb, yt = self._bounds()
        diag = float(np.hypot(xr - xl, yt - yb))
        center_scale = self._center_sqrt_n_scale(data)
        for i in active:
            key = ("pursuer", i)
            centroid = data["centroids"].get(key, pursuer_positions[i])
            center_error = float((np.linalg.norm(pursuer_positions[i] - centroid) / max(diag, 1e-6)) * center_scale)
            count = float(data["counts"].get(key, 0))
            area_error = min(abs(count / max(expected, 1e-9) - 1.0), float(self.reward_cfg.get("coverage_area_error_clip", 1.0)))
            values[i] = float(self.reward_cfg.get("coverage_center_weight", 1.0)) * center_error + float(self.reward_cfg.get("coverage_area_weight", 0.5)) * area_error
        return values

    def _coverage_cell_center_speed_profile(self, participant_ids: List[int], data: Dict[str, Any]) -> Dict[str, Any]:
        penalties = np.zeros(len(self.pursuers), dtype=float)
        participants = [i for i in participant_ids if 0 <= i < len(self.pursuers) and not self.pursuers[i].deactivated]
        if (
            not bool(self.reward_cfg.get("coverage_cell_center_speed_penalty_enabled", False))
            or not participants
            or self.coverage_geometric_success
        ):
            return {
                "participant_ids": [int(i) for i in participants],
                "penalty_sum": 0.0,
                "penalty_mean": 0.0,
                "near_mean": 0.0,
                "speed_excess_mean": 0.0,
                "penalties": penalties,
            }

        xl, xr, yb, yt = self._bounds()
        diag = max(float(np.hypot(xr - xl, yt - yb)), 1e-6)
        far_dist = max(float(self.reward_cfg.get("coverage_cell_center_speed_far_distance", 0.12)), 1e-6)
        full_dist = float(self.reward_cfg.get("coverage_cell_center_speed_full_distance", 0.04))
        if full_dist >= far_dist:
            full_dist = 0.5 * far_dist
        target_speed = max(float(self.reward_cfg.get("coverage_cell_center_speed_target", 0.75)), 0.0)
        max_speed = max(
            float(self.reward_cfg.get("coverage_cell_center_speed_zero", (self.config.get("pursuer", {}) or {}).get("max_speed", 3.0))),
            target_speed + 1e-6,
        )
        coeff = max(float(self.reward_cfg.get("coverage_cell_center_speed_coeff", 0.20)), 0.0)
        cap = max(float(self.reward_cfg.get("coverage_cell_center_speed_penalty_cap", 0.20)), 0.0)
        near_scores: List[float] = []
        speed_excess_scores: List[float] = []
        active_penalties: List[float] = []
        for i in participants:
            key = ("pursuer", i)
            centroid = np.asarray(data.get("centroids", {}).get(key, self._position(self.pursuers[i])), dtype=float)
            center_dist = float((np.linalg.norm(self._position(self.pursuers[i]) - centroid) / diag) * self._center_sqrt_n_scale(data))
            near_score = float(np.clip((far_dist - center_dist) / max(far_dist - full_dist, 1e-6), 0.0, 1.0))
            speed = float(self.pursuers[i].speed)
            speed_excess = float(np.clip((speed - target_speed) / max(max_speed - target_speed, 1e-6), 0.0, 1.0))
            penalty = -coeff * near_score * (speed_excess ** 2)
            if cap > 0.0:
                penalty = max(penalty, -cap)
            penalties[i] = penalty
            near_scores.append(near_score)
            speed_excess_scores.append(speed_excess)
            active_penalties.append(penalty)
        return {
            "participant_ids": [int(i) for i in participants],
            "penalty_sum": float(np.sum(active_penalties)) if active_penalties else 0.0,
            "penalty_mean": float(np.mean(active_penalties)) if active_penalties else 0.0,
            "near_mean": float(np.mean(near_scores)) if near_scores else 0.0,
            "speed_excess_mean": float(np.mean(speed_excess_scores)) if speed_excess_scores else 0.0,
            "penalties": penalties,
        }

    def _ce_coverage_geometry(
        self,
        data: Dict[str, Any],
        strict: bool,
        participant_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        active = [i for i, p in enumerate(self.pursuers) if not p.deactivated]
        participants = active if participant_indices is None else [i for i in participant_indices if i in active]
        counts = np.asarray([data.get("counts", {}).get(("pursuer", i), 0) for i in participants], dtype=float)
        owned_count = float(np.sum(counts))
        if len(participants) >= 2 and owned_count > 0.0:
            fractions = counts / owned_count
            area_cv = float(np.std(fractions) / max(np.mean(fractions), 1e-9))
        else:
            area_cv = float("inf")
        distances: List[float] = []
        costs: List[float] = []
        for i in participants:
            centroid = data.get("centroids", {}).get(("pursuer", i), self._position(self.pursuers[i]))
            distance, cost = self._ce_center_distance_and_cost(self._position(self.pursuers[i]), centroid, len(participants))
            distances.append(distance)
            costs.append(cost)
        prefix = "coverage_ce_success" if strict else "coverage_ce_loose"
        rms_threshold = float(self.reward_cfg.get(f"{prefix}_rms_threshold", self.reward_cfg.get("coverage_ce_success_rms_threshold", 0.05)))
        max_threshold = float(self.reward_cfg.get(f"{prefix}_max_threshold", self.reward_cfg.get("coverage_ce_success_max_threshold", 0.10)))
        rms_error = float(np.sqrt(np.mean(costs))) if costs else float("inf")
        max_error = float(max(distances)) if distances else float("inf")
        min_participants = int(self.reward_cfg.get("coverage_ce_min_active_pursuers", self.reward_cfg.get("min_active_pursuers", 2)))
        xl, xr, yb, yt = self._bounds()
        inside_ratio = float(np.mean([(xl <= self.pursuers[i].x <= xr and yb <= self.pursuers[i].y <= yt) for i in participants])) if participants else 0.0
        converged = bool(len(participants) >= min_participants and rms_error <= rms_threshold and max_error <= max_threshold)
        speeds = [float(self.pursuers[i].speed) for i in participants]
        component_counts = data.get("centroid_component_counts", {})
        disconnected = data.get("centroid_disconnected_ratios", {})
        projection = data.get("centroid_projection_distances", {})
        keys = [("pursuer", i) for i in participants]
        return {
            "participant_ids": [int(i) for i in participants],
            "participant_count": int(len(participants)),
            "participant_owned_grid_count": owned_count,
            "active_pursuer_owned_grid_count": float(sum(data.get("counts", {}).get(("pursuer", i), 0) for i in active)),
            "excluded_active_pursuer_owned_grid_count": float(sum(data.get("counts", {}).get(("pursuer", i), 0) for i in active)) - owned_count,
            "expected_owned_grid_count": owned_count / max(len(participants), 1),
            "area_cv": area_cv,
            "center_ok_ratio": float(np.mean(np.asarray(distances) <= max_threshold)) if distances else 0.0,
            "inside_ratio": inside_ratio,
            "converged_now": converged,
            "max_speed": float(max(speeds)) if speeds else float("inf"),
            "mean_speed": float(np.mean(speeds)) if speeds else float("inf"),
            "geometry_strict": converged,
            "speed_strict": bool(speeds and max(speeds) <= float(self.reward_cfg.get("strict_speed_threshold", 0.15))),
            "ce_center_distances": distances,
            "ce_center_costs": costs,
            "ce_center_rms": rms_error,
            "ce_center_max": max_error,
            "ce_rms_threshold": rms_threshold,
            "ce_max_threshold": max_threshold,
            "ce_component_count_max": int(max([component_counts.get(key, 0) for key in keys], default=0)),
            "ce_disconnected_ratio_max": float(max([disconnected.get(key, 0.0) for key in keys], default=0.0)),
            "ce_projection_distance_max": float(max([projection.get(key, 0.0) for key in keys], default=0.0)),
            "coverage_objective_version": "centroid_energy_v0",
        }

    def _voradj_coverage_geometry(
        self,
        data: Dict[str, Any],
        strict: bool,
        participant_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        if self._ce_coverage_enabled():
            return self._ce_coverage_geometry(data, strict, participant_indices)
        active = [i for i, p in enumerate(self.pursuers) if not p.deactivated]
        participants = active if participant_indices is None else [i for i in participant_indices if i in active]
        active_counts = np.asarray([data["counts"].get(("pursuer", i), 0) for i in active], dtype=float)
        participant_counts = np.asarray([data["counts"].get(("pursuer", i), 0) for i in participants], dtype=float)
        participant_owned_grid_count = float(np.sum(participant_counts))
        active_pursuer_owned_grid_count = float(np.sum(active_counts))
        expected_owned_grid_count = participant_owned_grid_count / max(len(participants), 1)
        if len(participants) < 2:
            area_cv = float("inf")
            center_ok_ratio = 0.0
            inside_ratio = 0.0
            max_speed_val = float("inf")
            converged = False
        else:
            participant_areas = participant_counts / max(participant_owned_grid_count, 1.0)
            area_cv = float(np.std(participant_areas) / max(np.mean(participant_areas), 1e-9))
            xl, xr, yb, yt = self._bounds()
            diag = float(np.hypot(xr - xl, yt - yb))
            center_dists = []
            for i in participants:
                centroid = data["centroids"].get(("pursuer", i), self._position(self.pursuers[i]))
                center_dists.append(float(np.linalg.norm(self._position(self.pursuers[i]) - centroid) / max(diag, 1e-6)))
            center_threshold = float(self.reward_cfg.get("strict_center_distance", 0.08) if strict else self.reward_cfg.get("distribution_voronoi_center_distance_threshold", 0.12))
            center_ratio_threshold = float(self.reward_cfg.get("strict_center_ok_ratio", 0.75) if strict else self.reward_cfg.get("distribution_voronoi_center_ok_ratio_threshold", 0.5))
            area_threshold = float(self.reward_cfg.get("strict_area_cv", 0.30) if strict else self.reward_cfg.get("distribution_area_cv_threshold", 0.40))
            inside_required = float(self.reward_cfg.get("strict_inside_ratio", 0.99) if strict else self.reward_cfg.get("distribution_inside_ratio_threshold", 0.95))
            center_ok_ratio = float(np.mean(np.asarray(center_dists) <= center_threshold))
            inside_ratio = float(np.mean([(xl <= self.pursuers[i].x <= xr and yb <= self.pursuers[i].y <= yt) for i in participants]))
            max_speed_val = float(max(self.pursuers[i].speed for i in participants))
            converged = bool(area_cv <= area_threshold and center_ok_ratio >= center_ratio_threshold and inside_ratio >= inside_required)
            if strict and bool(self.reward_cfg.get("coverage_strict_static_enabled", False)):
                converged = bool(converged and max_speed_val <= float(self.reward_cfg.get("strict_speed_threshold", 0.15)))
        return {
            "participant_ids": [int(i) for i in participants],
            "participant_count": int(len(participants)),
            "participant_owned_grid_count": participant_owned_grid_count,
            "active_pursuer_owned_grid_count": active_pursuer_owned_grid_count,
            "excluded_active_pursuer_owned_grid_count": active_pursuer_owned_grid_count - participant_owned_grid_count,
            "expected_owned_grid_count": expected_owned_grid_count,
            "area_cv": area_cv,
            "center_ok_ratio": center_ok_ratio,
            "inside_ratio": inside_ratio,
            "converged_now": bool(converged),
            "max_speed": max_speed_val,
            "mean_speed": float(np.mean([self.pursuers[i].speed for i in participants])) if participants else float("inf"),
            "geometry_strict": bool(converged),
            "speed_strict": bool(max_speed_val <= float(self.reward_cfg.get("strict_speed_threshold", 0.15))) if participants else False,
        }

    def _voradj_coverage_converged(self, update_hold: bool = False, strict: bool = True, data: Optional[Dict[str, Any]] = None) -> bool:
        data = data if data is not None else self._coverage_voronoi_map()
        metrics = self._voradj_coverage_geometry(data, strict=strict)
        converged = bool(metrics["converged_now"])
        if update_hold:
            self.distribution_hold_steps = self.distribution_hold_steps + 1 if converged else 0
        if self._ce_coverage_enabled():
            hold_required = int(self.reward_cfg.get("coverage_ce_success_hold_steps", 30))
        else:
            hold_required = int(self.reward_cfg.get("strict_hold_steps", 20) if strict else self.reward_cfg.get("distribution_success_hold_steps", 1))
        success = bool(converged and self.distribution_hold_steps >= hold_required)
        self.last_distribution_metrics = {
            **metrics,
            "hold_steps": int(self.distribution_hold_steps),
            "success_after_hold": success,
        }
        return success

    def _coverage_settle_enabled(self) -> bool:
        return bool(not self._ce_coverage_enabled() and self.reward_cfg.get("coverage_settle_enabled", False))

    def _coverage_speed_scores(self, mean_speed: float, max_speed: float) -> Tuple[float, float, float]:
        mean_full = float(self.reward_cfg.get("coverage_settle_mean_full_speed", 0.15))
        mean_zero = float(self.reward_cfg.get("coverage_settle_mean_zero_speed", 0.60))
        max_full = float(self.reward_cfg.get("coverage_settle_max_full_speed", 0.30))
        max_zero = float(self.reward_cfg.get("coverage_settle_max_zero_speed", 1.00))
        mean_denom = max(mean_zero - mean_full, 1e-9)
        max_denom = max(max_zero - max_full, 1e-9)
        mean_upper = max(mean_zero / mean_denom, 0.0)
        max_upper = max(max_zero / max_denom, 0.0)
        if not np.isfinite(mean_speed):
            mean_score = 0.0
        else:
            mean_score = float(np.clip((mean_zero - mean_speed) / mean_denom, 0.0, mean_upper))
        if not np.isfinite(max_speed):
            max_score = 0.0
        else:
            max_score = float(np.clip((max_zero - max_speed) / max_denom, 0.0, max_upper))
        return mean_score, max_score, min(mean_score, max_score)

    def _coverage_early_speed_profile(self, participant_ids: List[int], geometry_metrics: Dict[str, Any]) -> Dict[str, Any]:
        participants = [i for i in participant_ids if 0 <= i < len(self.pursuers) and not self.pursuers[i].deactivated]
        rewards = np.zeros(len(self.pursuers), dtype=float)
        if not bool(self.reward_cfg.get("coverage_early_speed_shaping_enabled", False)) or not participants:
            return {
                "participant_ids": [int(i) for i in participants],
                "gate": 0.0,
                "mean_speed_score": 0.0,
                "reward_sum": 0.0,
                "reward_mean": 0.0,
                "rewards": rewards,
            }

        area_cv = float(geometry_metrics.get("area_cv", float("inf")))
        center_ratio = float(geometry_metrics.get("center_ok_ratio", 0.0))
        inside_ratio = float(geometry_metrics.get("inside_ratio", 0.0))
        area_zero = float(self.reward_cfg.get("coverage_early_speed_area_zero", 0.70))
        area_full = float(self.reward_cfg.get("coverage_early_speed_area_full", self.reward_cfg.get("strict_area_cv", 0.30)))
        center_zero = float(self.reward_cfg.get("coverage_early_speed_center_zero", 0.50))
        center_full = float(self.reward_cfg.get("coverage_early_speed_center_full", self.reward_cfg.get("strict_center_ok_ratio", 0.75)))
        inside_zero = float(self.reward_cfg.get("coverage_early_speed_inside_zero", 0.95))
        inside_full = float(self.reward_cfg.get("coverage_early_speed_inside_full", self.reward_cfg.get("strict_inside_ratio", 0.99)))

        area_score = float(np.clip((area_zero - area_cv) / max(area_zero - area_full, 1e-9), 0.0, 1.0))
        center_score = float(np.clip((center_ratio - center_zero) / max(center_full - center_zero, 1e-9), 0.0, 1.0))
        inside_score = float(np.clip((inside_ratio - inside_zero) / max(inside_full - inside_zero, 1e-9), 0.0, 1.0))
        gate = float(np.clip(min(area_score, center_score, inside_score), 0.0, 1.0))

        max_reward = max(float(self.reward_cfg.get("coverage_early_speed_max_reward", 3.0)), 0.0)
        full_speed = float(self.reward_cfg.get("coverage_early_speed_full_speed", self.reward_cfg.get("coverage_settle_success_max_speed", 0.30)))
        zero_speed = float(self.reward_cfg.get("coverage_early_speed_zero_speed", self.reward_cfg.get("coverage_settle_max_zero_speed", 1.00)))
        denom = max(zero_speed - full_speed, 1e-9)
        speed_scores: List[float] = []
        for i in participants:
            speed_score = float(np.clip((zero_speed - float(self.pursuers[i].speed)) / denom, 0.0, 1.0))
            rewards[i] = max_reward * gate * speed_score
            speed_scores.append(speed_score)
        active_rewards = [float(rewards[i]) for i in participants]
        return {
            "participant_ids": [int(i) for i in participants],
            "gate": gate,
            "area_score": area_score,
            "center_score": center_score,
            "inside_score": inside_score,
            "mean_speed_score": float(np.mean(speed_scores)) if speed_scores else 0.0,
            "reward_sum": float(np.sum(active_rewards)) if active_rewards else 0.0,
            "reward_mean": float(np.mean(active_rewards)) if active_rewards else 0.0,
            "rewards": rewards,
        }

    @staticmethod
    def _sigmoid(value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-float(np.clip(value, -60.0, 60.0)))))

    def _coverage_motion_enabled(self) -> bool:
        return bool(self.reward_cfg.get("coverage_motion_penalty_enabled", False))

    def _coverage_motion_success_enabled(self) -> bool:
        return bool(self.reward_cfg.get("coverage_motion_success_enabled", False))

    def _coverage_settle_terminate_on_success(self) -> bool:
        return bool(self.reward_cfg.get("coverage_settle_terminate_on_success", True))

    def _coverage_motion_gate(self, geometry_metrics: Dict[str, Any]) -> float:
        if not (self._coverage_motion_enabled() or self._coverage_motion_success_enabled()):
            return 0.0
        if bool(geometry_metrics.get("converged_now", False)) or self.coverage_geometric_success:
            return 1.0
        mode = str(self.reward_cfg.get("coverage_motion_gate_mode", "soft")).strip().lower()
        area_cv = float(geometry_metrics.get("area_cv", float("inf")))
        center_ratio = float(geometry_metrics.get("center_ok_ratio", 0.0))
        inside_ratio = float(geometry_metrics.get("inside_ratio", 0.0))
        area_threshold = float(self.reward_cfg.get("strict_area_cv", 0.30))
        center_threshold = float(self.reward_cfg.get("strict_center_ok_ratio", 0.75))
        inside_threshold = float(self.reward_cfg.get("strict_inside_ratio", 0.99))
        if mode == "hard":
            return 1.0 if (area_cv <= area_threshold and center_ratio >= center_threshold and inside_ratio >= inside_threshold) else 0.0
        area_tau = max(float(self.reward_cfg.get("coverage_motion_gate_area_tau", 0.08)), 1e-9)
        center_tau = max(float(self.reward_cfg.get("coverage_motion_gate_center_tau", 0.10)), 1e-9)
        inside_tau = max(float(self.reward_cfg.get("coverage_motion_gate_inside_tau", 0.03)), 1e-9)
        soft_gate = (
            self._sigmoid((area_threshold - area_cv) / area_tau)
            * self._sigmoid((center_ratio - center_threshold) / center_tau)
            * self._sigmoid((inside_ratio - inside_threshold) / inside_tau)
        )
        pre_scale = float(self.reward_cfg.get("coverage_motion_pre_geometry_gate_scale", 0.25))
        return float(np.clip(pre_scale * soft_gate, 0.0, 1.0))

    def _action_accel_turn(self, idx: int, actions: List[Optional[int]]) -> Tuple[float, float]:
        if idx >= len(actions) or actions[idx] is None:
            return 0.0, 0.0
        if self.continuous_control:
            diagnostics = getattr(self.pursuers[idx], "last_action_diagnostics", {}) or {}
            return float(diagnostics.get("actual_acceleration", 0.0)), float(
                diagnostics.get("validated_angular_velocity", diagnostics.get("commanded_angular_velocity", 0.0))
            )
        try:
            action = int(actions[idx])
            accel, turn = self.pursuers[idx].action_list[action]
        except Exception:
            return 0.0, 0.0
        return float(accel), float(turn)

    def _coverage_motion_profile(
        self,
        participant_ids: List[int],
        actions: List[Optional[int]],
        geometry_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        participants = [i for i in participant_ids if 0 <= i < len(self.pursuers) and not self.pursuers[i].deactivated]
        gate = self._coverage_motion_gate(geometry_metrics)
        penalties = np.zeros(len(self.pursuers), dtype=float)
        if not participants:
            return {
                "participant_ids": [],
                "gate": float(gate),
                "score": 0.0,
                "mean_speed": float("inf"),
                "max_speed": float("inf"),
                "max_accel": float("inf"),
                "max_turn": float("inf"),
                "success_now": False,
                "penalties": penalties,
                "penalty_sum": 0.0,
                "penalty_mean": 0.0,
            }

        speed_scale = max(float(self.reward_cfg.get("coverage_motion_speed_scale", 0.60)), 1e-9)
        accel_scale = max(float(self.reward_cfg.get("coverage_motion_accel_scale", 0.40)), 1e-9)
        turn_scale = max(float(self.reward_cfg.get("coverage_motion_turn_scale", 0.5235987755982988)), 1e-9)
        speed_weight = float(self.reward_cfg.get("coverage_motion_speed_weight", 0.55))
        accel_weight = float(self.reward_cfg.get("coverage_motion_accel_weight", 0.25))
        turn_weight = float(self.reward_cfg.get("coverage_motion_turn_weight", 0.20))
        coeff = float(self.reward_cfg.get("coverage_motion_penalty_coeff", 0.50))
        cost_cap = max(float(self.reward_cfg.get("coverage_motion_penalty_cap", 0.75)), 0.0)

        scores: List[float] = []
        speeds: List[float] = []
        accels: List[float] = []
        turns: List[float] = []
        for i in participants:
            speed = float(self.pursuers[i].speed)
            accel, turn = self._action_accel_turn(i, actions)
            abs_accel = abs(float(accel))
            abs_turn = abs(float(turn))
            speeds.append(speed)
            accels.append(abs_accel)
            turns.append(abs_turn)
            speed_cost = 1.0 - float(np.exp(-((speed / speed_scale) ** 2)))
            accel_cost = 1.0 - float(np.exp(-((abs_accel / accel_scale) ** 2)))
            turn_cost = 1.0 - float(np.exp(-((abs_turn / turn_scale) ** 2)))
            raw_cost = speed_weight * speed_cost + accel_weight * accel_cost + turn_weight * turn_cost
            penalty = -gate * coeff * raw_cost
            if cost_cap > 0.0:
                penalty = max(penalty, -cost_cap)
            penalties[i] = float(penalty)
            scores.append(float(np.exp(-((speed / speed_scale) ** 2)) * np.exp(-((abs_accel / accel_scale) ** 2)) * np.exp(-((abs_turn / turn_scale) ** 2))))

        max_speed = float(max(speeds))
        mean_speed = float(np.mean(speeds))
        max_accel = float(max(accels))
        max_turn = float(max(turns))
        speed_ok = bool(max_speed <= float(self.reward_cfg.get("coverage_motion_success_max_speed", self.reward_cfg.get("coverage_settle_success_max_speed", 0.30))))
        accel_ok = True
        if bool(self.reward_cfg.get("coverage_motion_success_require_accel", False)):
            accel_ok = bool(max_accel <= float(self.reward_cfg.get("coverage_motion_success_max_accel", 0.05)))
        turn_ok = True
        if bool(self.reward_cfg.get("coverage_motion_success_require_turn", False)):
            turn_ok = bool(max_turn <= float(self.reward_cfg.get("coverage_motion_success_max_turn", 0.05)))
        success_now = bool(geometry_metrics.get("converged_now", False) and speed_ok and accel_ok and turn_ok)
        active_penalties = [float(penalties[i]) for i in participants]
        return {
            "participant_ids": [int(i) for i in participants],
            "gate": float(gate),
            "score": float(np.mean(scores)) if scores else 0.0,
            "mean_speed": mean_speed,
            "max_speed": max_speed,
            "max_accel": max_accel,
            "max_turn": max_turn,
            "speed_ok": bool(speed_ok),
            "accel_ok": bool(accel_ok),
            "turn_ok": bool(turn_ok),
            "success_now": bool(success_now),
            "penalties": penalties,
            "penalty_sum": float(np.sum(active_penalties)) if active_penalties else 0.0,
            "penalty_mean": float(np.mean(active_penalties)) if active_penalties else 0.0,
        }

    def _coverage_deceleration_reward(
        self,
        strict_geometry: bool,
        before_participant_ids: List[int],
        after_participant_ids: List[int],
        before_speeds: Dict[int, float],
        collision_or_death: bool,
    ) -> float:
        if not self._coverage_settle_enabled() or not strict_geometry or collision_or_death:
            return 0.0
        if before_participant_ids != after_participant_ids or not after_participant_ids:
            return 0.0
        mean_before = float(np.mean([before_speeds[i] for i in after_participant_ids]))
        mean_after = float(np.mean([self.pursuers[i].speed for i in after_participant_ids]))
        scale = max(float(self.reward_cfg.get("coverage_settle_deceleration_scale", 0.4)), 1e-9)
        coeff = float(self.reward_cfg.get("coverage_settle_deceleration_coeff", 5.0))
        return coeff * float(np.clip((mean_before - mean_after) / scale, -1.0, 1.0))

    def _latch_coverage_geometric_success(self, metrics: Dict[str, Any]) -> None:
        self.coverage_geometric_success = True
        self.coverage_geometric_latch_mean_speed = float(metrics.get("mean_speed", float("inf")))
        self.coverage_geometric_latch_max_speed = float(metrics.get("max_speed", float("inf")))
        self.settle_hold_steps = 0
        self.coverage_settle_elapsed_steps = 0
        if self._coverage_settle_enabled():
            self.post_capture_grace_remaining = int(self.reward_cfg.get("coverage_settle_window_steps", 40))

    def _coverage_geometric_latch_reward(self, post_capture: bool = False) -> float:
        if self._coverage_settle_enabled():
            if post_capture:
                return float(
                    self.reward_cfg.get(
                        "post_capture_geometric_latch_reward",
                        self.reward_cfg.get("coverage_geometric_latch_reward", 0.0),
                    )
                )
            return float(self.reward_cfg.get("coverage_geometric_latch_reward", 0.0))
        if post_capture:
            return float(
                self.reward_cfg.get(
                    "post_capture_coverage_success_reward",
                    self.reward_cfg.get("coverage_success_reward", 120.0),
                )
            )
        return float(self.reward_cfg.get("coverage_success_reward", 120.0))

    def _advance_coverage_settle_window(
        self,
        strict_geometry: bool,
        mean_speed: float,
        max_speed: float,
        motion_success_now: Optional[bool] = None,
    ) -> Tuple[float, float]:
        mean_score, max_score, speed_score = self._coverage_speed_scores(mean_speed, max_speed)
        self.coverage_mean_score = mean_score
        self.coverage_max_score = max_score
        self.coverage_speed_score = speed_score
        if (
            not self._coverage_settle_enabled()
            or not self.coverage_geometric_success
            or self.coverage_settle_timeout
            or self.post_capture_grace_remaining <= 0
        ):
            return 0.0, 0.0

        motion_mode = self._coverage_motion_success_enabled()
        terminate_on_success = self._coverage_settle_terminate_on_success()
        if self.coverage_settled_success and (not motion_mode or terminate_on_success):
            return 0.0, 0.0

        self.coverage_settle_elapsed_steps += 1
        repeat_reward = 0.0
        terminal_reward = 0.0
        if motion_mode:
            success_now = bool(motion_success_now)
            if strict_geometry:
                if success_now:
                    repeat_reward = float(self.reward_cfg.get("coverage_settle_motion_repeat_reward", 20.0))
                else:
                    repeat_mode = str(self.reward_cfg.get("coverage_settle_geometric_repeat_mode", "fixed")).strip().lower()
                    if repeat_mode in {"linear_speed", "speed_linear", "reward_linear"}:
                        min_reward = float(
                            self.reward_cfg.get(
                                "coverage_settle_speed_shaping_min_reward",
                                self.reward_cfg.get("coverage_settle_geometric_repeat_reward", 5.0),
                            )
                        )
                        max_reward = float(self.reward_cfg.get("coverage_settle_speed_shaping_max_reward", 15.0))
                        speed_fraction = float(np.clip(speed_score, 0.0, 1.0))
                        repeat_reward = min_reward + max(max_reward - min_reward, 0.0) * speed_fraction
                    else:
                        repeat_reward = float(self.reward_cfg.get("coverage_settle_geometric_repeat_reward", 5.0))
            if success_now:
                self.settle_hold_steps += 1
            else:
                self.settle_hold_steps = 0

            hold_required = int(self.reward_cfg.get("coverage_settle_hold_steps", 1))
            if self.settle_hold_steps >= hold_required and not self.coverage_settled_success:
                self.coverage_settled_success = True
                self.coverage_geometric_to_settled_steps = int(self.coverage_settle_elapsed_steps)
                terminal_reward = float(self.reward_cfg.get("coverage_settled_terminal_reward", 240.0))
                if terminate_on_success:
                    self.post_capture_grace_remaining = 0
                    return repeat_reward, terminal_reward
        else:
            if strict_geometry:
                repeat_reward = float(self.reward_cfg.get("coverage_settle_repeat_reward", 20.0)) * speed_score
            # Reward-score shape and success termination are separate semantics.
            # The score may exceed one near zero speed, while settled success keeps
            # the explicit 0.15/0.30 gate unless independently configured.
            mean_threshold = float(self.reward_cfg.get("coverage_settle_success_mean_speed", 0.15))
            max_threshold = float(self.reward_cfg.get("coverage_settle_success_max_speed", 0.30))
            if strict_geometry and mean_speed <= mean_threshold and max_speed <= max_threshold:
                self.settle_hold_steps += 1
            else:
                self.settle_hold_steps = 0

            hold_required = int(self.reward_cfg.get("coverage_settle_hold_steps", 10))
            if self.settle_hold_steps >= hold_required:
                self.coverage_settled_success = True
                self.coverage_geometric_to_settled_steps = int(self.coverage_settle_elapsed_steps)
                self.post_capture_grace_remaining = 0
                terminal_reward = float(self.reward_cfg.get("coverage_settled_terminal_reward", 120.0))
                return repeat_reward, terminal_reward

        self.post_capture_grace_remaining -= 1
        if self.post_capture_grace_remaining <= 0:
            self.post_capture_grace_remaining = 0
            if not self.coverage_settled_success:
                self.coverage_settle_timeout = True
        return repeat_reward, terminal_reward


    def step(self, pursuer_actions: List[Optional[int]], evader_actions: Optional[List[Optional[int]]] = None) -> StepResult:
        if evader_actions is None:
            evader_actions = [None] * len(self.evaders)
        before_p = np.asarray([self._position(p) for p in self.pursuers], dtype=float)
        before_e = np.asarray([self._position(e) for e in self.evaders], dtype=float) if self.evaders else np.zeros((0, 2))
        before_evader_velocities = np.asarray([np.asarray(e.velocity, dtype=float).copy() for e in self.evaders], dtype=float) if self.evaders else np.zeros((0, 2))
        before_settle_participant_ids = [i for i, p in enumerate(self.pursuers) if not p.deactivated]
        before_settle_speeds = {i: float(self.pursuers[i].speed) for i in before_settle_participant_ids}
        before_settle_phase_eligible = bool(not self.evaders or all(e.deactivated for e in self.evaders))
        before_data = self._capture_voronoi_map(before_p, before_e)
        before_coverage_data = self._coverage_voronoi_map(before_p, before_e)
        before_raw_labels = self._raw_task_labels_from_map(before_data)
        before_labels = self._task_labels_from_map(before_data, update_effective=False, raw_labels=before_raw_labels)
        before_cov = self._voradj_coverage_potentials(
            before_p,
            before_e,
            data=before_coverage_data,
        )
        for p, action in zip(self.pursuers, pursuer_actions):
            self._move_robot(p, action)
        for e, action in zip(self.evaders, evader_actions):
            self._move_robot(e, action)
        self._invalidate_voronoi_cache()
        self._refresh_collisions()
        self._record_soft_boundary_out_of_bounds()
        self._zone_update_metrics()
        after_p = np.asarray([self._position(p) for p in self.pursuers], dtype=float)
        after_e = np.asarray([self._position(e) for e in self.evaders], dtype=float) if self.evaders else np.zeros((0, 2))
        data = self._capture_voronoi_map(after_p, after_e)
        coverage_data = self._coverage_voronoi_map(after_p, after_e)
        self._store_current_voronoi_map(data)
        self._store_current_voronoi_map(coverage_data)
        after_cov = self._voradj_coverage_potentials(
            after_p,
            after_e,
            data=coverage_data,
        )
        next_raw_labels = self._raw_task_labels_from_map(data)
        next_labels = self._task_labels_from_map(data, update_effective=True, raw_labels=next_raw_labels)
        self.last_task_labels = next_labels
        rewards = np.zeros(len(self.pursuers), dtype=float)
        reward_coverage_component = np.zeros(len(self.pursuers), dtype=float)
        reward_capture_component = np.zeros(len(self.pursuers), dtype=float)
        reward_safety_component = np.zeros(len(self.pursuers), dtype=float)
        reward_terminal_component = np.zeros(len(self.pursuers), dtype=float)
        reward_speed_repeat_component = np.zeros(len(self.pursuers), dtype=float)
        reward_early_speed_component = np.zeros(len(self.pursuers), dtype=float)
        reward_cell_center_speed_component = np.zeros(len(self.pursuers), dtype=float)
        reward_deceleration_component = np.zeros(len(self.pursuers), dtype=float)
        reward_settled_terminal_component = np.zeros(len(self.pursuers), dtype=float)
        reward_motion_penalty_component = np.zeros(len(self.pursuers), dtype=float)
        reward_ce_center_component = np.zeros(len(self.pursuers), dtype=float)
        reward_ce_control_component = np.zeros(len(self.pursuers), dtype=float)
        reward_ce_pbrs_component = np.zeros(len(self.pursuers), dtype=float)
        reward_ce_terminal_correction_component = np.zeros(len(self.pursuers), dtype=float)
        reward_support_blend_capture_component = np.zeros(len(self.pursuers), dtype=float)
        reward_support_blend_coverage_component = np.zeros(len(self.pursuers), dtype=float)
        support_reward_blend_flags = np.zeros(len(self.pursuers), dtype=bool)
        ce_reward_applied = np.zeros(len(self.pursuers), dtype=bool)
        ce_pbrs_reset_reasons = ["none"] * len(self.pursuers)

        def apply_ce_pbrs_reset(index: int, center_cost: float, reason: str) -> None:
            if not ce_reward_applied[index] or ce_pbrs_reset_reasons[index] != "none":
                return
            correction = self._ce_terminal_pbrs_correction(center_cost)
            rewards[index] += correction
            reward_coverage_component[index] += correction
            reward_ce_pbrs_component[index] += correction
            reward_ce_terminal_correction_component[index] += correction
            ce_pbrs_reset_reasons[index] = reason
        infos = [
            {
                "state": "normal",
                "task_label": before_labels[i],
                "next_task_label": next_labels[i],
            }
            for i in range(len(self.pursuers))
        ]
        active_targets = [j for j, e in enumerate(self.evaders) if not e.deactivated]
        coverage_phi_clip = float(self.reward_cfg.get("coverage_phi_clip", 0.05))
        coverage_progress_coeff = float(self.reward_cfg.get("coverage_progress_coeff", 1.0))
        coverage_phi_coeff = float(self.reward_cfg.get("coverage_phi_coeff", 0.15))
        timestep_penalty = float(self.env_cfg.get("timestep_penalty", -1.0))
        emergency_penalties = self._emergency_proximity_penalties()

        support_reward_blend_enabled = self._vct_ls_support_reward_blend_enabled()
        support_capture_weight, support_coverage_weight = self._vct_ls_support_reward_weights()

        def support_reward_blend_active(index: int) -> bool:
            return bool(
                support_reward_blend_enabled
                and active_targets
                and before_labels[index] == "coverage"
                and self._support_pursuing_friend_ids(index, before_labels, before_data)
            )

        def capture_reward_candidates(index: int, support_mode: bool = False) -> List[int]:
            if self._vct_ls_enabled():
                direct_targets = self._vct_ls_direct_enemy_ids_for_pursuer(index, before_p, before_e)
                if direct_targets or not support_mode:
                    return direct_targets
                mode = self._support_reward_capture_target_mode()
                if mode in {"neighbor_visible", "neighbor_direct", "friend_visible"}:
                    neighbor_targets: List[int] = []
                    for neighbor_idx in self._support_pursuing_friend_ids(index, before_labels, before_data):
                        neighbor_targets.extend(self._vct_ls_direct_enemy_ids_for_pursuer(neighbor_idx, before_p, before_e))
                    return sorted(set(neighbor_targets))
                if mode in {"nearest_active", "all_active"}:
                    return list(active_targets)
                return []

            adjacent_targets = [
                nk[1]
                for nk in before_data.get("adjacency", {}).get(("pursuer", index), set())
                if nk[0] == "evader" and not self.evaders[nk[1]].deactivated
            ]
            adjacent_targets = sorted(set(adjacent_targets).union(self._zone_extra_evader_ids_for_pursuer(index, before_data)))
            return adjacent_targets or list(active_targets)

        def capture_task_reward(index: int, candidates: List[int]) -> float:
            if self._ring_importance_ms_enabled():
                configured_penalty = self.reward_cfg.get("capture_timestep_penalty", 0.0)
                task_reward = timestep_penalty if configured_penalty is None else float(configured_penalty)
                reward_candidates = candidates
                if self._vct_ls_enabled():
                    reward_candidates = self._vct_ls_direct_enemy_ids_for_pursuer(index, before_p, before_e)
                if reward_candidates:
                    target_id = min(reward_candidates, key=lambda j: np.linalg.norm(before_p[index] - before_e[j]))
                    task_reward += float(self.reward_cfg.get("omega_ring_ms", 2.0)) * self._ring_importance_ms_reward(
                        index,
                        target_id,
                        before_p,
                        before_e,
                        before_evader_velocities,
                        after_p,
                    )
                return float(task_reward)

            task_reward = timestep_penalty
            if candidates:
                target_id = min(candidates, key=lambda j: np.linalg.norm(after_p[index] - after_e[j]))
                d_before = np.linalg.norm(before_p[index] - before_e[target_id])
                d_after = np.linalg.norm(after_p[index] - after_e[target_id])
                task_reward += float(self.reward_cfg.get("omega_approach", 1.0)) * np.clip(
                    d_before - d_after,
                    -float(self.reward_cfg.get("c_d", 3.0)),
                    float(self.reward_cfg.get("c_d", 3.0)),
                )
                task_reward += float(self.reward_cfg.get("omega_mean_shift", 2.0)) * self._mean_shift_reward(index, target_id, before_p, before_e, after_p)
                task_reward += float(self.reward_cfg.get("omega_front", 0.5)) * self._front_reward(index, target_id, after_p, after_e)
            return float(task_reward)

        def support_capture_task_reward(index: int, candidates: List[int]) -> float:
            mode = self._support_reward_capture_component_mode()
            if mode not in {"approach_only", "legacy_approach_only", "attraction_only"}:
                return capture_task_reward(index, candidates)
            if not candidates:
                return 0.0
            # Match the legacy mix-capture attraction item exactly, without its
            # timestep, old mean-shift, front, or direct-pursuer ring-MS terms.
            target_id = min(candidates, key=lambda j: np.linalg.norm(after_p[index] - after_e[j]))
            d_before = np.linalg.norm(before_p[index] - before_e[target_id])
            d_after = np.linalg.norm(after_p[index] - after_e[target_id])
            support_cfg = self.config.get("voradj", {}) or {}
            weight = float(support_cfg.get("support_reward_approach_weight", 1.0))
            progress_clip = float(
                support_cfg.get("support_reward_approach_clip", self.reward_cfg.get("c_d", 3.0))
            )
            return float(weight * np.clip(d_before - d_after, -progress_clip, progress_clip))

        def coverage_task_reward(index: int) -> Tuple[float, Dict[str, float]]:
            if self._ce_coverage_enabled():
                accel, turn = self._action_accel_turn(index, pursuer_actions)
                max_accel = float(max(abs(float(x)) for x in self.pursuers[index].a))
                max_turn = float(max(abs(float(x)) for x in self.pursuers[index].w))
                control_cost, _control_terms = normalized_control_cost(
                    speed=float(self.pursuers[index].speed),
                    acceleration=accel,
                    angular_velocity=turn,
                    max_speed=float(self.pursuers[index].max_speed),
                    max_acceleration=max_accel,
                    max_angular_velocity=max_turn,
                    speed_weight=float(self.reward_cfg.get("coverage_ce_speed_weight", 0.05)),
                    acceleration_weight=float(self.reward_cfg.get("coverage_ce_acceleration_weight", 0.002)),
                    angular_velocity_weight=float(self.reward_cfg.get("coverage_ce_angular_velocity_weight", 0.001)),
                )
                task_reward, ce_terms = ce_transition_reward(
                    before_center_cost=float(before_cov[index]),
                    after_center_cost=float(after_cov[index]),
                    control_cost=control_cost,
                    reward_scale=float(self.reward_cfg.get("coverage_ce_reward_scale", 10.0)),
                    gamma=float(
                        (self.config.get("discount", {}) or {}).get(
                            "gamma",
                            (self.config.get("iqn", {}) or {}).get("gamma", 0.99),
                        )
                    ),
                    pbrs_enabled=bool(self.reward_cfg.get("coverage_ce_pbrs_enabled", True)),
                    pbrs_kappa=float(self.reward_cfg.get("coverage_ce_pbrs_kappa", 1.0)),
                    terminal=False,
                )
                return float(task_reward), {str(k): float(v) for k, v in ce_terms.items()}

            task_reward = coverage_progress_coeff * float(np.clip(before_cov[index] - after_cov[index], -coverage_phi_clip, coverage_phi_clip))
            task_reward -= coverage_phi_coeff * after_cov[index]
            task_reward += timestep_penalty
            return float(task_reward), {}

        def add_coverage_ce_terms(index: int, ce_terms: Dict[str, float], scale: float = 1.0) -> None:
            if not ce_terms or scale == 0.0:
                return
            ce_reward_applied[index] = True
            reward_ce_center_component[index] += scale * float(ce_terms.get("base_center", 0.0))
            reward_ce_control_component[index] += scale * float(ce_terms.get("control", 0.0))
            reward_ce_pbrs_component[index] += scale * float(ce_terms.get("pbrs", 0.0))

        for i, p in enumerate(self.pursuers):
            if p.deactivated:
                continue
            if before_labels[i] == "capture" and active_targets:
                task_reward = capture_task_reward(i, capture_reward_candidates(i, support_mode=False))
                rewards[i] += task_reward
                reward_capture_component[i] += task_reward
            elif support_reward_blend_active(i):
                capture_reward = support_capture_task_reward(i, capture_reward_candidates(i, support_mode=True))
                coverage_reward, ce_terms = coverage_task_reward(i)
                blended_capture = support_capture_weight * capture_reward
                blended_coverage = support_coverage_weight * coverage_reward
                rewards[i] += blended_capture + blended_coverage
                reward_capture_component[i] += blended_capture
                reward_coverage_component[i] += blended_coverage
                reward_support_blend_capture_component[i] += blended_capture
                reward_support_blend_coverage_component[i] += blended_coverage
                support_reward_blend_flags[i] = True
                add_coverage_ce_terms(i, ce_terms, scale=support_coverage_weight)
            else:
                task_reward, ce_terms = coverage_task_reward(i)
                rewards[i] += task_reward
                reward_coverage_component[i] += task_reward
                add_coverage_ce_terms(i, ce_terms)
            safety_penalty = float(emergency_penalties[i])
            if safety_penalty:
                rewards[i] += safety_penalty
                reward_safety_component[i] += safety_penalty

        coverage_success_bonus = 0.0
        coverage_hold_bonus = 0.0
        pure_coverage_scene = len(self.evaders) == 0
        success_hold_enabled = bool(pure_coverage_scene or not active_targets)
        if not success_hold_enabled:
            self.distribution_hold_steps = 0
        coverage_success = self._voradj_coverage_converged(
            update_hold=success_hold_enabled,
            strict=True,
            data=coverage_data,
        )
        if not success_hold_enabled:
            coverage_success = False
        converged_now = bool(self.last_distribution_metrics.get("converged_now", False))
        normal_coverage_metrics = dict(self.last_distribution_metrics)
        active_pursuers = [i for i, p in enumerate(self.pursuers) if not p.deactivated]
        hold_phase = "pure_coverage" if pure_coverage_scene else ("pre_capture" if active_targets else "post_capture")
        hold_eligible = [i for i in active_pursuers if before_labels[i] == "coverage"]
        hold_support_candidates = [
            i
            for i in hold_eligible
            if any(
                nk[0] == "pursuer" and before_labels[nk[1]] == "capture"
                for nk in before_data.get("adjacency", {}).get(("pursuer", i), set())
            )
        ]
        if hold_phase == "pre_capture":
            hold_geometry_metrics = self._voradj_coverage_geometry(
                coverage_data,
                strict=True,
                participant_indices=hold_eligible,
            )
            hold_geometry_scope = "coverage_agents_only"
        else:
            hold_geometry_metrics = dict(normal_coverage_metrics)
            hold_geometry_scope = "all_active_pursuers"
        hold_converged_now = bool(hold_geometry_metrics.get("converged_now", False))
        motion_penalty_profile = {
            "participant_ids": [],
            "gate": 0.0,
            "score": 0.0,
            "mean_speed": float("inf"),
            "max_speed": float("inf"),
            "max_accel": float("inf"),
            "max_turn": float("inf"),
            "success_now": False,
            "penalty_sum": 0.0,
            "penalty_mean": 0.0,
        }
        reward_motion_penalty = 0.0
        if self._coverage_motion_enabled():
            motion_penalty_ids = hold_eligible if active_targets else active_pursuers
            motion_penalty_profile = self._coverage_motion_profile(
                motion_penalty_ids,
                pursuer_actions,
                hold_geometry_metrics,
            )
            motion_penalties = np.asarray(motion_penalty_profile.get("penalties", np.zeros(len(self.pursuers))), dtype=float)
            for i in motion_penalty_ids:
                penalty = float(motion_penalties[i])
                if penalty:
                    rewards[i] += penalty
                    reward_coverage_component[i] += penalty
                    reward_motion_penalty_component[i] += penalty
            reward_motion_penalty = float(np.sum(reward_motion_penalty_component))
            self.coverage_motion_gate = float(motion_penalty_profile.get("gate", 0.0))
            self.coverage_motion_score = float(motion_penalty_profile.get("score", 0.0))
            self.coverage_motion_max_accel = float(motion_penalty_profile.get("max_accel", 0.0))
            self.coverage_motion_max_turn = float(motion_penalty_profile.get("max_turn", 0.0))
        else:
            self.coverage_motion_gate = 0.0
            self.coverage_motion_score = 0.0
            self.coverage_motion_max_accel = 0.0
            self.coverage_motion_max_turn = 0.0
            self.coverage_motion_success_now = False
        cell_center_speed_profile = {
            "participant_ids": [],
            "penalty_sum": 0.0,
            "penalty_mean": 0.0,
            "near_mean": 0.0,
            "speed_excess_mean": 0.0,
        }
        reward_cell_center_speed = 0.0
        if not self.coverage_geometric_success and bool(self.reward_cfg.get("coverage_cell_center_speed_penalty_enabled", False)):
            cell_center_ids = hold_eligible if active_targets else active_pursuers
            cell_center_speed_profile = self._coverage_cell_center_speed_profile(cell_center_ids, coverage_data)
            cell_center_penalties = np.asarray(cell_center_speed_profile.get("penalties", np.zeros(len(self.pursuers))), dtype=float)
            for i in cell_center_ids:
                penalty = float(cell_center_penalties[i])
                if penalty:
                    rewards[i] += penalty
                    reward_coverage_component[i] += penalty
                    reward_cell_center_speed_component[i] += penalty
            reward_cell_center_speed = float(np.sum(reward_cell_center_speed_component))
        # Hold rewards are intentionally uncapped in this diagnostic version.
        # They apply only to agents whose action was selected in the coverage
        # role. A non-pursuing agent adjacent to a pursuing teammate remains a
        # coverage agent and is tracked separately as a support candidate.
        if (not self._ce_coverage_enabled()) and hold_converged_now and not self.post_capture_coverage_success and hold_eligible:
            coverage_hold_bonus = float(self.reward_cfg.get("coverage_hold_reward", 1.0))
            for i in hold_eligible:
                rewards[i] += coverage_hold_bonus
                reward_coverage_component[i] += coverage_hold_bonus
            self.coverage_hold_reward_claim_steps += 1
            self.coverage_hold_reward_steps_by_phase[hold_phase] += 1

        captured_events = self._loose_capture_events()
        if captured_events:
            capture_adjacency_data = data
            captured_evader_ids = {int(event["evader_id"]) for event in captured_events}
            for event in captured_events:
                self.evaders[event["evader_id"]].deactivated = True
                factor = 1.0 if event.get("capture_type") == "stationary" else (TWO_PI / max(len(event["participants"]), 1)) * np.exp(-float(np.std(event["angles"])))
                for i in event["participants"]:
                    terminal_reward = float(self.env_cfg.get("goal_reward", 120.0)) * factor
                    rewards[i] += terminal_reward
                    reward_terminal_component[i] += terminal_reward
            if self.capture_snapshot is None and all(e.deactivated for e in self.evaders):
                self.capture_snapshot = {
                    "step": int(self.episode_step + 1),
                    "positions": [[float(p.x), float(p.y)] for p in self.pursuers],
                    "active_mask": [bool(not p.deactivated) for p in self.pursuers],
                }
            self._clear_release_delay_after_single_enemy_capture(capture_adjacency_data, captured_evader_ids)
            if (
                all(e.deactivated for e in self.evaders)
                and self._ce_pbrs_reset_mode() == "phase_and_all_terminal"
            ):
                for i in range(len(self.pursuers)):
                    apply_ce_pbrs_reset(
                        i,
                        float(after_cov[i]),
                        "capture_phase_boundary",
                    )
        self.last_capture_events = captured_events
        if captured_events:
            # A strict hold accumulated on the mixed ownership map cannot count
            # toward recovery after the evader site disappears.
            self.distribution_hold_steps = 0
            coverage_success = False
            converged_now = False
            self._invalidate_voronoi_cache()
            data = self._capture_voronoi_map(after_p, after_e)
            coverage_data = self._coverage_voronoi_map(after_p, after_e)
            self._store_current_voronoi_map(data)
            self._store_current_voronoi_map(coverage_data)
            next_raw_labels = self._raw_task_labels_from_map(data)
            next_labels = self._task_labels_from_map(data, update_effective=True, raw_labels=next_raw_labels)
            self.last_task_labels = next_labels
            for i in range(len(infos)):
                infos[i]["next_task_label"] = next_labels[i]

        active_evaders = [j for j, e in enumerate(self.evaders) if not e.deactivated]
        post_capture_phase = bool((not pure_coverage_scene) and len(active_evaders) == 0)
        settle_enabled = self._coverage_settle_enabled()
        settle_phase_eligible = bool(
            settle_enabled
            and before_settle_phase_eligible
            and (pure_coverage_scene or post_capture_phase)
            and not captured_events
        )
        current_mean_speed = float(normal_coverage_metrics.get("mean_speed", float("inf")))
        current_max_speed = float(normal_coverage_metrics.get("max_speed", float("inf")))
        if settle_phase_eligible:
            (
                self.coverage_mean_score,
                self.coverage_max_score,
                self.coverage_speed_score,
            ) = self._coverage_speed_scores(current_mean_speed, current_max_speed)
        else:
            self.coverage_mean_score = 0.0
            self.coverage_max_score = 0.0
            self.coverage_speed_score = 0.0
        motion_success_now = False
        settle_motion_profile = motion_penalty_profile
        if settle_phase_eligible and self._coverage_motion_success_enabled():
            settle_motion_profile = self._coverage_motion_profile(
                active_pursuers,
                pursuer_actions,
                normal_coverage_metrics,
            )
            motion_success_now = bool(settle_motion_profile.get("success_now", False))
            self.coverage_motion_gate = float(settle_motion_profile.get("gate", 0.0))
            self.coverage_motion_score = float(settle_motion_profile.get("score", 0.0))
            self.coverage_motion_max_accel = float(settle_motion_profile.get("max_accel", 0.0))
            self.coverage_motion_max_turn = float(settle_motion_profile.get("max_turn", 0.0))
            self.coverage_motion_success_now = motion_success_now
        elif not settle_phase_eligible:
            self.coverage_motion_success_now = False
        early_speed_profile = {
            "participant_ids": [],
            "gate": 0.0,
            "mean_speed_score": 0.0,
            "reward_sum": 0.0,
            "reward_mean": 0.0,
        }
        reward_early_speed = 0.0
        if (
            settle_phase_eligible
            and not self.coverage_geometric_success
            and not self.post_capture_coverage_success
            and bool(self.reward_cfg.get("coverage_early_speed_shaping_enabled", False))
        ):
            early_speed_profile = self._coverage_early_speed_profile(active_pursuers, normal_coverage_metrics)
            early_speed_rewards = np.asarray(early_speed_profile.get("rewards", np.zeros(len(self.pursuers))), dtype=float)
            for i in active_pursuers:
                early_reward = float(early_speed_rewards[i])
                if early_reward:
                    rewards[i] += early_reward
                    reward_coverage_component[i] += early_reward
                    reward_early_speed_component[i] += early_reward
            reward_early_speed = float(np.sum(reward_early_speed_component))
        reward_deceleration = 0.0
        if settle_phase_eligible:
            collision_or_death = bool(
                before_settle_participant_ids != active_pursuers
                or any(self.pursuers[i].collision for i in before_settle_participant_ids)
            )
            reward_deceleration = self._coverage_deceleration_reward(
                strict_geometry=converged_now,
                before_participant_ids=before_settle_participant_ids,
                after_participant_ids=active_pursuers,
                before_speeds=before_settle_speeds,
                collision_or_death=collision_or_death,
            )
            if reward_deceleration:
                for i in active_pursuers:
                    rewards[i] += reward_deceleration
                    reward_coverage_component[i] += reward_deceleration
                    reward_deceleration_component[i] += reward_deceleration
        reward_speed_repeat = 0.0
        reward_settled_terminal = 0.0
        if pure_coverage_scene:
            if coverage_success and not self.post_capture_coverage_success:
                coverage_success_bonus = 0.0 if self._ce_coverage_enabled() else self._coverage_geometric_latch_reward(post_capture=False)
                if coverage_success_bonus:
                    for i in active_pursuers:
                        rewards[i] += coverage_success_bonus
                        reward_terminal_component[i] += coverage_success_bonus
                self.post_capture_coverage_success = True
                self.post_capture_step = int(self.episode_step + 1)
                self.post_capture_coverage_step = self.post_capture_step
                self._latch_coverage_geometric_success(normal_coverage_metrics)
                if self._ce_coverage_enabled():
                    self.coverage_settled_success = True
                    self.coverage_geometric_to_settled_steps = 0
                    self.post_capture_grace_remaining = 0
                    for i in active_pursuers:
                        apply_ce_pbrs_reset(
                            i,
                            float(after_cov[i]),
                            "coverage_success",
                        )
                elif not settle_enabled:
                    self.post_capture_grace_remaining = int(self.reward_cfg.get("distribution_converged_grace_steps", 0))
            elif self.post_capture_coverage_success and self.post_capture_grace_remaining > 0:
                if settle_enabled:
                    reward_speed_repeat, reward_settled_terminal = self._advance_coverage_settle_window(
                        strict_geometry=converged_now,
                        mean_speed=current_mean_speed,
                        max_speed=current_max_speed,
                        motion_success_now=motion_success_now,
                    )
                    coverage_success_bonus = reward_speed_repeat + reward_settled_terminal
                    for i in active_pursuers:
                        if reward_speed_repeat:
                            rewards[i] += reward_speed_repeat
                            reward_coverage_component[i] += reward_speed_repeat
                            reward_speed_repeat_component[i] += reward_speed_repeat
                        if reward_settled_terminal:
                            rewards[i] += reward_settled_terminal
                            reward_terminal_component[i] += reward_settled_terminal
                            reward_settled_terminal_component[i] += reward_settled_terminal
                else:
                    repeat_bonus = 0.0
                    if converged_now and bool(self.reward_cfg.get("coverage_success_reward_repeat", False)):
                        repeat_bonus = float(self.reward_cfg.get("coverage_success_repeat_reward", 0.0))
                    if repeat_bonus:
                        coverage_success_bonus = repeat_bonus
                        for i in active_pursuers:
                            rewards[i] += repeat_bonus
                            reward_terminal_component[i] += repeat_bonus
                    self.post_capture_grace_remaining -= 1
        elif post_capture_phase and not captured_events:
            if not self.post_capture_started:
                self.post_capture_started = True
                self.post_capture_step = 0
                self.post_capture_grace_remaining = 0
            self.post_capture_step += 1
            if coverage_success and not self.post_capture_coverage_success:
                coverage_success_bonus = 0.0 if self._ce_coverage_enabled() else self._coverage_geometric_latch_reward(post_capture=True)
                if coverage_success_bonus:
                    for i in active_pursuers:
                        rewards[i] += coverage_success_bonus
                        reward_terminal_component[i] += coverage_success_bonus
                self.post_capture_coverage_success = True
                self.post_capture_coverage_step = self.post_capture_step
                self._latch_coverage_geometric_success(normal_coverage_metrics)
                if self._ce_coverage_enabled():
                    self.coverage_settled_success = True
                    self.coverage_geometric_to_settled_steps = 0
                    self.post_capture_grace_remaining = 0
                    for i in active_pursuers:
                        apply_ce_pbrs_reset(
                            i,
                            float(after_cov[i]),
                            "coverage_success",
                        )
                elif not settle_enabled:
                    self.post_capture_grace_remaining = int(self.reward_cfg.get("post_capture_coverage_grace_steps", 40))
            elif self.post_capture_coverage_success and self.post_capture_grace_remaining > 0:
                if settle_enabled:
                    reward_speed_repeat, reward_settled_terminal = self._advance_coverage_settle_window(
                        strict_geometry=converged_now,
                        mean_speed=current_mean_speed,
                        max_speed=current_max_speed,
                        motion_success_now=motion_success_now,
                    )
                    coverage_success_bonus = reward_speed_repeat + reward_settled_terminal
                    for i in active_pursuers:
                        if reward_speed_repeat:
                            rewards[i] += reward_speed_repeat
                            reward_coverage_component[i] += reward_speed_repeat
                            reward_speed_repeat_component[i] += reward_speed_repeat
                        if reward_settled_terminal:
                            rewards[i] += reward_settled_terminal
                            reward_terminal_component[i] += reward_settled_terminal
                            reward_settled_terminal_component[i] += reward_settled_terminal
                else:
                    repeat_enabled = bool(self.reward_cfg.get("post_capture_success_reward_repeat", True))
                    repeat_reward = float(self.reward_cfg.get("post_capture_success_repeat_reward", self.reward_cfg.get("post_capture_hold_reward", 1.0)))
                    hold = repeat_reward if (converged_now and repeat_enabled) else 0.0
                    if hold:
                        coverage_success_bonus = hold
                        for i in active_pursuers:
                            rewards[i] += hold
                            reward_coverage_component[i] += hold
                    self.post_capture_grace_remaining -= 1

        boundary_penalties = self._boundary_penalties()
        self.last_boundary_penalty_count = int(np.count_nonzero(boundary_penalties))
        self.last_boundary_penalty_sum = float(np.sum(boundary_penalties))
        rewards += boundary_penalties
        reward_safety_component += boundary_penalties
        boundary_proximity_penalties = self._boundary_proximity_penalties()
        self.last_boundary_proximity_penalty_count = int(np.count_nonzero(boundary_proximity_penalties))
        self.last_boundary_proximity_penalty_sum = float(np.sum(boundary_proximity_penalties))
        rewards += boundary_proximity_penalties
        reward_safety_component += boundary_proximity_penalties
        for i, p in enumerate(self.pursuers):
            if p.collision:
                collision_penalty = float(self.env_cfg.get("collision_penalty", -80.0))
                rewards[i] += collision_penalty
                reward_safety_component[i] += collision_penalty
                infos[i]["state"] = "deactivated after collision" if p.deactivated else "collision"
        all_captured = bool(self.evaders and all(e.deactivated and not e.collision for e in self.evaders))
        evader_lost = bool(self.evaders and all(e.deactivated for e in self.evaders) and not all_captured)
        zone_metrics = self._zone_metrics()
        zone_breach_done = bool(
            zone_metrics.get("zone_breach_event", False)
            and bool(self._zone_config().get("zone_done_on_breach", True))
        )
        self.episode_step += 1
        self.total_steps += 1
        timeout = self.episode_step >= self.episode_max_length
        pre_capture_max_length = int(self.env_cfg.get("pre_capture_max_length", 0))
        pre_capture_timeout = bool(
            active_evaders
            and pre_capture_max_length > 0
            and self.episode_step >= pre_capture_max_length
        )
        post_window = int(self.reward_cfg.get("post_capture_coverage_window_steps", 300))
        voradj_cfg = (self.config.get("voradj", {}) or {})
        capture_terminal_done = bool(
            all_captured and bool(voradj_cfg.get("capture_episode_ends_on_capture", False))
        )
        if settle_enabled:
            if self._coverage_settle_terminate_on_success():
                coverage_phase_done = bool(
                    (pure_coverage_scene or post_capture_phase)
                    and (self.coverage_settled_success or self.coverage_settle_timeout)
                )
            else:
                coverage_phase_done = bool(
                    (pure_coverage_scene or post_capture_phase)
                    and (
                        self.coverage_settle_timeout
                        or (self.coverage_geometric_success and self.post_capture_grace_remaining <= 0)
                    )
                )
        else:
            coverage_phase_done = bool(
                (pure_coverage_scene and self.post_capture_coverage_success and self.post_capture_grace_remaining <= 0)
                or (post_capture_phase and self.post_capture_coverage_success and self.post_capture_grace_remaining <= 0)
            )
        voradj_done = bool(
            coverage_phase_done
            or capture_terminal_done
            or zone_breach_done
            or pre_capture_timeout
            or (post_capture_phase and self.post_capture_started and not self.post_capture_coverage_success and self.post_capture_step >= post_window)
        )
        # Replay phase describes the state in which the stored action was
        # selected. A transition that captures the final evader therefore
        # remains pre_capture/pursuing rather than being mislabeled as recovery.
        phase_label = hold_phase
        dones: List[bool] = []
        too_few = sum(not q.deactivated for q in self.pursuers) < int(self.reward_cfg.get("min_active_pursuers", 2))
        for i, p in enumerate(self.pursuers):
            done = bool(p.deactivated or timeout or voradj_done or evader_lost or too_few)
            if done and self._ce_pbrs_reset_mode() == "phase_and_all_terminal":
                apply_ce_pbrs_reset(i, float(after_cov[i]), "terminal")
            if timeout and infos[i]["state"] == "normal":
                infos[i]["state"] = "too long episode"
            elif pre_capture_timeout and infos[i]["state"] == "normal":
                infos[i]["state"] = "pre-capture timeout"
            elif capture_terminal_done and infos[i]["state"] == "normal":
                infos[i]["state"] = "capture completed"
            elif zone_breach_done and infos[i]["state"] == "normal":
                infos[i]["state"] = "zone breach"
            elif voradj_done and infos[i]["state"] == "normal":
                infos[i]["state"] = "voradj completed"
            elif evader_lost and infos[i]["state"] == "normal":
                infos[i]["state"] = "evader collision"
            infos[i]["replay_metadata"] = {
                "task_label": before_labels[i],
                "next_task_label": next_labels[i],
                "task_switched": bool(before_labels[i] != next_labels[i]),
                "raw_task_label": before_raw_labels[i],
                "next_raw_task_label": next_raw_labels[i],
                "phase": phase_label,
                "reward_coverage": float(reward_coverage_component[i]),
                "reward_capture": float(reward_capture_component[i]),
                "reward_safety": float(reward_safety_component[i]),
                "reward_terminal": float(reward_terminal_component[i]),
                "reward_speed_repeat": float(reward_speed_repeat_component[i]),
                "reward_early_speed": float(reward_early_speed_component[i]),
                "reward_deceleration": float(reward_deceleration_component[i]),
                "reward_settled_terminal": float(reward_settled_terminal_component[i]),
                "reward_motion_penalty": float(reward_motion_penalty_component[i]),
                "reward_ce_center": float(reward_ce_center_component[i]),
                "reward_ce_control": float(reward_ce_control_component[i]),
                "reward_ce_pbrs": float(reward_ce_pbrs_component[i]),
                "reward_ce_terminal_correction": float(reward_ce_terminal_correction_component[i]),
                "reward_support_blend_capture": float(reward_support_blend_capture_component[i]),
                "reward_support_blend_coverage": float(reward_support_blend_coverage_component[i]),
                "coverage_ce_pbrs_reset_reason": ce_pbrs_reset_reasons[i],
                "coverage_ce_pbrs_reset_mode": self._ce_pbrs_reset_mode(),
                "coverage_ce_speed_weight": float(
                    self.reward_cfg.get("coverage_ce_speed_weight", 0.05)
                ),
                "coverage_ce_acceleration_weight": float(
                    self.reward_cfg.get("coverage_ce_acceleration_weight", 0.002)
                ),
                "coverage_ce_angular_velocity_weight": float(
                    self.reward_cfg.get("coverage_ce_angular_velocity_weight", 0.001)
                ),
                "reward_total": float(rewards[i]),
                "enemy_neighbor_count": int(before_raw_labels[i] == "capture"),
                "effective_pursuing": bool(before_labels[i] == "capture"),
                "friend_neighbor_count": int(sum(1 for nk in before_data.get("adjacency", {}).get(("pursuer", i), set()) if nk[0] == "pursuer")) if before_labels[i] != "inactive" else 0,
                "support_candidate": bool(self._support_pursuing_friend_ids(i, before_labels, before_data)),
                "support_pursuing_friend_count": int(len(self._support_pursuing_friend_ids(i, before_labels, before_data))),
                "support_reward_blend_active": bool(support_reward_blend_flags[i]),
                "support_reward_capture_weight": float(support_capture_weight if support_reward_blend_flags[i] else 0.0),
                "support_reward_coverage_weight": float(support_coverage_weight if support_reward_blend_flags[i] else 0.0),
                "support_reward_capture_target_mode": self._support_reward_capture_target_mode(),
                "support_reward_capture_component_mode": self._support_reward_capture_component_mode(),
                "perception_topology_version": self._perception_topology_version(),
                "vct_ls_enabled": bool(self._vct_ls_enabled()),
                "vct_ls_direct_enemy_count": int(len(self._vct_ls_direct_enemy_ids_for_pursuer(i, before_p, before_e))) if self._vct_ls_enabled() and before_labels[i] != "inactive" else 0,
            }
            if self.continuous_control:
                diagnostics = dict(getattr(self.pursuers[i], "last_action_diagnostics", {}) or {})
                infos[i]["action_diagnostics"] = diagnostics
                infos[i]["replay_metadata"]["action_diagnostics"] = dict(diagnostics)
            dones.append(done)
        capture_count = int(sum(1 for x in next_labels if x == "capture"))
        coverage_count = int(sum(1 for x in next_labels if x == "coverage"))
        raw_capture_count = int(sum(1 for x in next_raw_labels if x == "capture"))
        raw_coverage_count = int(sum(1 for x in next_raw_labels if x == "coverage"))
        task_switch_count = int(sum(a != b for a, b in zip(before_labels, next_labels) if a != "inactive"))
        active_count = max(capture_count + coverage_count, 1)
        enemy_neighbor_ratio = capture_count / active_count
        active_evader_count = int(sum(not e.deactivated for e in self.evaders))
        direct_detect_counts = self._vct_ls_direct_detect_counts(after_p, after_e) if self._vct_ls_enabled() else {}
        miss_evader = False
        if active_evader_count:
            if self._vct_ls_enabled():
                miss_evader = any(int(direct_detect_counts.get(j, 0)) == 0 for j, e in enumerate(self.evaders) if not e.deactivated)
            else:
                for j, e in enumerate(self.evaders):
                    if e.deactivated:
                        continue
                    ev_key = ("evader", j)
                    if not any(nk[0] == "pursuer" for nk in data.get("adjacency", {}).get(ev_key, set())):
                        miss_evader = True
                        break
        support_count = int(sum(1 for i, label in enumerate(next_labels) if label == "coverage" and any(nk[0] == "pursuer" and next_labels[nk[1]] == "capture" for nk in data.get("adjacency", {}).get(("pursuer", i), set()))))
        self.last_voradj_metrics = {
            "coverage_objective_version": str(self.reward_cfg.get("coverage_objective_version", "legacy")),
            "coverage_ce_pbrs_reset_mode": self._ce_pbrs_reset_mode(),
            "coverage_ce_speed_weight": float(
                self.reward_cfg.get("coverage_ce_speed_weight", 0.05)
            ),
            "coverage_ce_acceleration_weight": float(
                self.reward_cfg.get("coverage_ce_acceleration_weight", 0.002)
            ),
            "coverage_ce_angular_velocity_weight": float(
                self.reward_cfg.get("coverage_ce_angular_velocity_weight", 0.001)
            ),
            "voronoi_obstacle_mode": str(coverage_data.get("voronoi_obstacle_mode", "legacy_assign")),
            "capture_adjacency_obstacle_mode": str(data.get("voronoi_obstacle_mode", "legacy_assign")),
            "coverage_target_obstacle_mode": str(coverage_data.get("voronoi_obstacle_mode", "legacy_assign")),
            "perception_topology_version": self._perception_topology_version(),
            "friendly_comm_site_scope": str(data.get("site_scope", "all_active_entities")),
            "friendly_comm_include_evaders_as_sites": bool(data.get("include_evaders_as_sites", True)),
            "vct_ls_enabled": bool(self._vct_ls_enabled()),
            "vct_ls_enemy_sensing_radius": float(self._vct_ls_sensing_radius("enemy")),
            "vct_ls_obstacle_sensing_radius": float(self._vct_ls_sensing_radius("obstacle")),
            "vct_ls_surface_sensing": bool(self._vct_ls_use_surface_distance()),
            "vct_ls_release_delay_applied": bool(self._vct_ls_enabled() and self._vct_ls_apply_release_delay()),
            "vct_ls_detected_evader_count": int(sum(1 for value in direct_detect_counts.values() if int(value) > 0)) if self._vct_ls_enabled() else active_evader_count,
            "vct_ls_all_evaders_invisible": bool(self._vct_ls_enabled() and active_evader_count > 0 and all(int(value) == 0 for value in direct_detect_counts.values())),
            "vct_ls_direct_detector_counts": {str(int(k)): int(v) for k, v in direct_detect_counts.items()},
            "voronoi_free_grid_count": int(coverage_data.get("free_grid_count", len(coverage_data.get("points", [])))),
            "voronoi_blocked_grid_count": int(coverage_data.get("blocked_grid_count", 0)),
            "ce_center_rms": float(normal_coverage_metrics.get("ce_center_rms", float("inf"))),
            "ce_center_max": float(normal_coverage_metrics.get("ce_center_max", float("inf"))),
            "ce_component_count_max": int(normal_coverage_metrics.get("ce_component_count_max", 0)),
            "ce_disconnected_ratio_max": float(normal_coverage_metrics.get("ce_disconnected_ratio_max", 0.0)),
            "ce_projection_distance_max": float(normal_coverage_metrics.get("ce_projection_distance_max", 0.0)),
            "capture_agent_count": capture_count,
            "coverage_agent_count": coverage_count,
            "raw_capture_agent_count": raw_capture_count,
            "raw_coverage_agent_count": raw_coverage_count,
            "enemy_neighbor_ratio": float(enemy_neighbor_ratio),
            "raw_enemy_neighbor_ratio": float(raw_capture_count / max(raw_capture_count + raw_coverage_count, 1)),
            "miss_evader_step": bool(miss_evader),
            "support_candidate_count": support_count,
            "support_candidate_ratio": float(support_count / active_count),
            "support_reward_blend_enabled": bool(support_reward_blend_enabled),
            "support_reward_blend_active_count": int(np.count_nonzero(support_reward_blend_flags)),
            "support_reward_blend_active_ratio": float(np.count_nonzero(support_reward_blend_flags) / active_count),
            "support_reward_capture_weight": float(support_capture_weight),
            "support_reward_coverage_weight": float(support_coverage_weight),
            "support_reward_capture_target_mode": self._support_reward_capture_target_mode(),
            "support_reward_capture_component_mode": self._support_reward_capture_component_mode(),
            "reward_support_blend_capture_sum": float(np.sum(reward_support_blend_capture_component)),
            "reward_support_blend_coverage_sum": float(np.sum(reward_support_blend_coverage_component)),
            "smoothed_pursuing_agent_count": int(sum(bool(getattr(p, "is_pursuing", False)) for p in self.pursuers if not p.deactivated)),
            "is_pursuing_release_delay_steps": int(self._pursuing_release_delay_steps),
            "is_pursuing_release_counters": [int(value) for value in self._pursuing_release_counters],
            "post_capture_coverage_success": bool(self.post_capture_coverage_success),
            "post_capture_coverage_step": -1 if self.post_capture_coverage_step is None else int(self.post_capture_coverage_step),
            "post_capture_window_expired": bool(post_capture_phase and self.post_capture_step >= post_window and not self.post_capture_coverage_success),
            "post_capture_grace_remaining": int(self.post_capture_grace_remaining) if post_capture_phase else 0,
            "post_capture_success_reward_repeat": bool(self.reward_cfg.get("post_capture_success_reward_repeat", True)),
            "post_capture_success_repeat_reward": float(self.reward_cfg.get("post_capture_success_repeat_reward", self.reward_cfg.get("post_capture_hold_reward", 1.0))),
            "post_capture_hold_reward": float(self.reward_cfg.get("post_capture_hold_reward", 1.0)),
            "capture_episode_ends_on_capture": bool(voradj_cfg.get("capture_episode_ends_on_capture", False)),
            "capture_terminal_done": bool(capture_terminal_done),
            "capture_episode_success_on_capture": bool(voradj_cfg.get("capture_episode_success_on_capture", False)),
            "pure_coverage_scene": bool(pure_coverage_scene),
            "pure_coverage_success": bool(pure_coverage_scene and self.post_capture_coverage_success),
            "pure_coverage_grace_remaining": int(self.post_capture_grace_remaining) if pure_coverage_scene else 0,
            "pure_coverage_done_grace_steps": int(self.reward_cfg.get("distribution_converged_grace_steps", 0)) if pure_coverage_scene else 0,
            "coverage_success_reward_repeat": bool(self.reward_cfg.get("coverage_success_reward_repeat", False)),
            "coverage_success_repeat_reward": float(self.reward_cfg.get("coverage_success_repeat_reward", 0.0)),
            "coverage_settle_enabled": bool(settle_enabled),
            "coverage_geometric_success": bool(self.coverage_geometric_success),
            "coverage_settled_success": bool(self.coverage_settled_success),
            "coverage_settle_timeout": bool(self.coverage_settle_timeout),
            "settle_hold_steps": int(self.settle_hold_steps),
            "settle_elapsed_steps": int(self.coverage_settle_elapsed_steps),
            "mean_score": float(self.coverage_mean_score),
            "max_score": float(self.coverage_max_score),
            "speed_score": float(self.coverage_speed_score),
            "coverage_motion_gate": float(self.coverage_motion_gate),
            "coverage_motion_score": float(self.coverage_motion_score),
            "coverage_motion_max_accel": float(self.coverage_motion_max_accel),
            "coverage_motion_max_turn": float(self.coverage_motion_max_turn),
            "coverage_motion_success_now": bool(self.coverage_motion_success_now),
            "coverage_motion_penalty_sum": float(reward_motion_penalty),
            "coverage_motion_penalty_mean": float(motion_penalty_profile.get("penalty_mean", 0.0)),
            "coverage_motion_profile": {k: v for k, v in settle_motion_profile.items() if k != "penalties"},
            "coverage_early_speed_enabled": bool(self.reward_cfg.get("coverage_early_speed_shaping_enabled", False)),
            "coverage_early_speed_gate": float(early_speed_profile.get("gate", 0.0)),
            "coverage_early_speed_mean_score": float(early_speed_profile.get("mean_speed_score", 0.0)),
            "coverage_early_speed_reward_sum": float(reward_early_speed),
            "coverage_early_speed_reward_mean": float(early_speed_profile.get("reward_mean", 0.0)),
            "coverage_early_speed_profile": {k: v for k, v in early_speed_profile.items() if k != "rewards"},
            "coverage_cell_center_speed_profile": {k: v for k, v in cell_center_speed_profile.items() if k != "penalties"},
            "geometric_latch_mean_speed": self.coverage_geometric_latch_mean_speed,
            "geometric_latch_max_speed": self.coverage_geometric_latch_max_speed,
            "geometric_to_settled_steps": -1 if self.coverage_geometric_to_settled_steps is None else int(self.coverage_geometric_to_settled_steps),
            "coverage_hold_reward_claim_steps": int(self.coverage_hold_reward_claim_steps),
            "coverage_hold_reward_steps_by_phase": dict(self.coverage_hold_reward_steps_by_phase),
            "hold_reward_eligible_count": int(len(hold_eligible)),
            "hold_reward_support_candidate_count": int(len(hold_support_candidates)),
            "hold_geometry_scope": hold_geometry_scope,
            "hold_geometry_metrics": dict(hold_geometry_metrics),
            "success_hold_enabled": bool(success_hold_enabled),
            "success_hold_steps": int(self.distribution_hold_steps),
            "task_switch_count": task_switch_count,
            **zone_metrics,
        }
        self.last_reward_terms = {
            "reward_mean": float(np.mean(rewards)),
            "reward_coverage_mean": float(np.mean(reward_coverage_component)),
            "reward_capture_mean": float(np.mean(reward_capture_component)),
            "reward_safety_mean": float(np.mean(reward_safety_component)),
            "reward_terminal_mean": float(np.mean(reward_terminal_component)),
            "capture_count": float(len(captured_events)),
            "coverage_success": float(coverage_success),
            "coverage_converged_now": float(converged_now),
            "coverage_hold_steps": float(self.distribution_hold_steps),
            "coverage_success_bonus": float(coverage_success_bonus),
            "coverage_hold_bonus": float(coverage_hold_bonus),
            "coverage_progress_coeff": coverage_progress_coeff,
            "reward_speed_repeat": float(reward_speed_repeat),
            "reward_early_speed": float(reward_early_speed),
            "reward_cell_center_speed": float(reward_cell_center_speed),
            "reward_deceleration": float(reward_deceleration),
            "reward_settled_terminal": float(reward_settled_terminal),
            "reward_motion_penalty": float(reward_motion_penalty),
            "reward_ce_center_mean": float(np.mean(reward_ce_center_component)),
            "reward_ce_control_mean": float(np.mean(reward_ce_control_component)),
            "reward_ce_pbrs_mean": float(np.mean(reward_ce_pbrs_component)),
            "reward_ce_terminal_correction_mean": float(np.mean(reward_ce_terminal_correction_component)),
            "reward_support_blend_capture_mean": float(np.mean(reward_support_blend_capture_component)),
            "reward_support_blend_coverage_mean": float(np.mean(reward_support_blend_coverage_component)),
            "support_reward_blend_active_count": int(np.count_nonzero(support_reward_blend_flags)),
            "coverage_motion_gate": float(self.coverage_motion_gate),
            "coverage_motion_score": float(self.coverage_motion_score),
            "coverage_motion_success_now": float(self.coverage_motion_success_now),
            "coverage_early_speed_gate": float(early_speed_profile.get("gate", 0.0)),
            "coverage_early_speed_reward_sum": float(reward_early_speed),
            "coverage_cell_center_speed_penalty_sum": float(reward_cell_center_speed),
            "coverage_cell_center_speed_penalty_mean": float(cell_center_speed_profile.get("penalty_mean", 0.0)),
            "coverage_cell_center_speed_near_mean": float(cell_center_speed_profile.get("near_mean", 0.0)),
            "coverage_cell_center_speed_excess_mean": float(cell_center_speed_profile.get("speed_excess_mean", 0.0)),
            "coverage_motion_max_accel": float(self.coverage_motion_max_accel),
            "coverage_motion_max_turn": float(self.coverage_motion_max_turn),
            "hold_reward_eligible_count": int(len(hold_eligible)),
            "hold_reward_support_candidate_count": int(len(hold_support_candidates)),
            "hold_geometry_scope": hold_geometry_scope,
            "hold_geometry_converged_now": float(hold_converged_now),
            "success_hold_enabled": float(success_hold_enabled),
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
            **self.last_voradj_metrics,
        }
        return StepResult(self.get_observations(), rewards, dones, infos)

    def episode_record(self, task: Optional[str] = None) -> Dict[str, Any]:
        captured = bool(self.evaders and all(e.deactivated and not e.collision for e in self.evaders))
        collision_free = not any(p.collision for p in self.pursuers)
        all_active = all(not p.deactivated for p in self.pursuers)
        old_hold = int(self.distribution_hold_steps)
        old_metrics = dict(self.last_distribution_metrics)
        loose = self._voradj_coverage_converged(update_hold=False, strict=False)
        loose_metrics = dict(self.last_distribution_metrics)
        strict = self._voradj_coverage_converged(update_hold=False, strict=True)
        strict_metrics = dict(self.last_distribution_metrics)
        self.distribution_hold_steps = old_hold
        self.last_distribution_metrics = old_metrics
        task_name = task or self.task
        pure_coverage_scene = len(self.evaders) == 0
        voradj_cfg = (self.config.get("voradj", {}) or {})
        capture_success_on_capture = bool(voradj_cfg.get("capture_episode_success_on_capture", False))
        settle_enabled = self._coverage_settle_enabled()
        episode_end_speeds = [float(p.speed) for p in self.pursuers if not p.deactivated]
        episode_end_mean_speed = float(np.mean(episode_end_speeds)) if episode_end_speeds else float("inf")
        episode_end_max_speed = float(max(episode_end_speeds)) if episode_end_speeds else float("inf")
        settle_mean_threshold = float(self.reward_cfg.get("coverage_settle_success_mean_speed", 0.15))
        settle_max_threshold = float(self.reward_cfg.get("coverage_settle_success_max_speed", 0.30))
        episode_end_stationary = bool(
            episode_end_mean_speed <= settle_mean_threshold
            and episode_end_max_speed <= settle_max_threshold
        )
        strict_area_threshold = float(self.reward_cfg.get("strict_area_cv", 0.30))
        strict_center_threshold = float(self.reward_cfg.get("strict_center_ok_ratio", 0.75))
        strict_inside_threshold = float(self.reward_cfg.get("strict_inside_ratio", 0.99))
        strict_area_cv_value = float(strict_metrics.get("area_cv", float("inf")))
        cv_loose_threshold = float(self.reward_cfg.get("coverage_cv_loose_area_cv_threshold", 0.15))
        coverage_cv015_success = bool(strict_area_cv_value <= cv_loose_threshold)
        strict_area_ok = bool(strict_area_cv_value <= strict_area_threshold)
        strict_center_ok = bool(float(strict_metrics.get("center_ok_ratio", 0.0)) >= strict_center_threshold)
        strict_inside_ok = bool(float(strict_metrics.get("inside_ratio", 0.0)) >= strict_inside_threshold)
        post_capture_episode = bool(captured and not pure_coverage_scene)
        zone_metrics = self._zone_metrics()
        self.last_voradj_metrics.update({
            "episode_end_mean_speed": episode_end_mean_speed,
            "episode_end_max_speed": episode_end_max_speed,
            **zone_metrics,
        })
        if pure_coverage_scene:
            if self._ce_coverage_enabled():
                episode_success = bool(self.post_capture_coverage_success or strict)
                episode_success_mode = "pure_centroid_energy_coverage"
            elif settle_enabled:
                episode_success = bool(self.coverage_settled_success)
                episode_success_mode = "pure_settled_coverage"
            else:
                episode_success = bool(self.post_capture_coverage_success or strict)
                episode_success_mode = "pure_strict_coverage"
        elif capture_success_on_capture:
            episode_success = bool(captured)
            episode_success_mode = "capture_only"
        else:
            if settle_enabled:
                episode_success = bool(captured and self.coverage_settled_success and collision_free)
                episode_success_mode = "capture_then_post_settled_coverage"
            else:
                episode_success = bool(captured and self.post_capture_coverage_success and collision_free)
                episode_success_mode = "capture_then_post_coverage"
        return {
            "task": task_name,
            "length": int(self.episode_step),
            "captured": captured,
            "fully_capture": bool(captured and collision_free and all_active),
            "episode_success": episode_success,
            "episode_success_mode": episode_success_mode,
            "collision_free": collision_free,
            "collision_event": not collision_free,
            "all_pursuers_active": all_active,
            "active_pursuers": int(sum(not p.deactivated for p in self.pursuers)),
            "coverage_loose_success": bool(loose),
            "coverage_strict_success": bool(strict),
            "coverage_training_success": bool(strict),
            "coverage_training_success_mode": "centroid_energy_v0" if self._ce_coverage_enabled() else "strict",
            "coverage_objective_version": str(self.reward_cfg.get("coverage_objective_version", "legacy")),
            "coverage_ce_center_rms": float(strict_metrics.get("ce_center_rms", float("inf"))),
            "coverage_ce_center_max": float(strict_metrics.get("ce_center_max", float("inf"))),
            "coverage_geometric_success": bool(self.coverage_geometric_success),
            "coverage_settled_success": bool(self.coverage_settled_success),
            "coverage_settle_timeout": bool(self.coverage_settle_timeout),
            "settle_hold_steps": int(self.settle_hold_steps),
            "episode_end_stationary": episode_end_stationary,
            "post_capture_episode": post_capture_episode,
            "post_capture_stationary_success": bool(post_capture_episode and episode_end_stationary),
            "coverage_strict_area_ok": strict_area_ok,
            "coverage_strict_center_ok": strict_center_ok,
            "coverage_strict_inside_ok": strict_inside_ok,
            "coverage_cv_loose_area_cv_threshold": cv_loose_threshold,
            "coverage_cv015_success": coverage_cv015_success,
            "coverage_cv_loose_success": coverage_cv015_success,
            "post_capture_cv015_success": bool(post_capture_episode and coverage_cv015_success),
            "coverage_strict_area_cv": strict_area_cv_value,
            "coverage_strict_center_ok_ratio": float(strict_metrics.get("center_ok_ratio", 0.0)),
            "coverage_strict_inside_ratio": float(strict_metrics.get("inside_ratio", 0.0)),
            "speed_score": float(self.coverage_speed_score),
            "mean_score": float(self.coverage_mean_score),
            "max_score": float(self.coverage_max_score),
            "geometric_latch_mean_speed": self.coverage_geometric_latch_mean_speed,
            "geometric_latch_max_speed": self.coverage_geometric_latch_max_speed,
            "episode_end_mean_speed": episode_end_mean_speed,
            "episode_end_max_speed": episode_end_max_speed,
            "geometric_to_settled_steps": -1 if self.coverage_geometric_to_settled_steps is None else int(self.coverage_geometric_to_settled_steps),
            "reward_speed_repeat": float(self.last_reward_terms.get("reward_speed_repeat", 0.0)),
            "reward_early_speed": float(self.last_reward_terms.get("reward_early_speed", 0.0)),
            "reward_deceleration": float(self.last_reward_terms.get("reward_deceleration", 0.0)),
            "reward_settled_terminal": float(self.last_reward_terms.get("reward_settled_terminal", 0.0)),
            "reward_motion_penalty": float(self.last_reward_terms.get("reward_motion_penalty", 0.0)),
            "coverage_motion_gate": float(self.last_voradj_metrics.get("coverage_motion_gate", 0.0)),
            "coverage_motion_score": float(self.last_voradj_metrics.get("coverage_motion_score", 0.0)),
            "coverage_motion_success_now": bool(self.last_voradj_metrics.get("coverage_motion_success_now", False)),
            "coverage_motion_max_accel": float(self.last_voradj_metrics.get("coverage_motion_max_accel", 0.0)),
            "coverage_motion_max_turn": float(self.last_voradj_metrics.get("coverage_motion_max_turn", 0.0)),
            "distribution_metrics": loose_metrics,
            "coverage_strict_distribution_metrics": strict_metrics,
            "reward_terms": dict(self.last_reward_terms),
            "voradj_metrics": dict(self.last_voradj_metrics),
            "zone_metrics": dict(zone_metrics),
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
            "pursuer_positions": [[float(p.x), float(p.y)] for p in self.pursuers if not p.deactivated],
            "pursuer_all_positions": [[float(p.x), float(p.y)] for p in self.pursuers],
            "pursuer_active_mask": [bool(not p.deactivated) for p in self.pursuers],
            "capture_snapshot": None if self.capture_snapshot is None else {
                "step": int(self.capture_snapshot["step"]),
                "positions": [list(position) for position in self.capture_snapshot["positions"]],
                "active_mask": list(self.capture_snapshot["active_mask"]),
            },
        }
