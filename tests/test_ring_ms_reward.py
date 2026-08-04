from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cocap_voradj.envs.base import Obstacle
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.trainer import deep_update, load_config, set_global_config


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

A3_CONFIG = ROOT / "configs/experiments/voradj_a3_apfnew_sqrtn_20260723/a3_apfnew_sqrtn_4v1_scratch_mix_2m.yaml"


def _config(**overrides):
    base = load_config(str(A3_CONFIG))
    patch = {
        "device": "cpu",
        "env": {
            "width": 120.0,
            "height": 120.0,
            "num_pursuers": 4,
            "num_evaders": 1,
            "num_obstacles": 0,
            "episode_max_length": 20,
            "init_speed": 0.0,
        },
        "reward": {
            "voradj_grid_size": 24,
            "capture_reward_mode": "ring_importance_ms_v0",
            "capture_timestep_penalty": 0.0,
            "omega_ring_ms": 2.0,
            "ring_ms_inner_extra_margin": 0.5,
            "ring_ms_preferred_center_radius": 8.0,
            "ring_ms_outer_center_radius": 10.5,
            "ring_ms_radial_sigma": 2.0,
            "ring_ms_cell_size": 1.0,
            "ring_ms_angle_alpha": 0.25,
            "ring_ms_angle_weight_min": 0.75,
            "ring_ms_angle_weight_max": 1.25,
            "min_active_pursuers": 4,
        },
        "voradj": {
            "perception_topology_version": "friendly_voronoi_comm_v0",
            "friendly_comm_obstacle_mode": "free_mask_projected",
            "voronoi_obstacle_mode": "free_mask_projected",
            "enemy_sensing_radius": 20.0,
            "obstacle_sensing_radius": 20.0,
            "local_sensing_uses_surface_distance": True,
            "vct_ls_apply_release_delay": True,
        },
        "zone_demo": {"enabled": False},
    }
    return deep_update(deep_update(base, patch), overrides)


def _env(config=None) -> VorAdjEnv:
    cfg = config or _config()
    set_global_config(cfg)
    env = VorAdjEnv(cfg, seed=20260731)
    env.reset()
    env.obstacles = []
    return env


def _place(env: VorAdjEnv, pursuers, evader=(60.0, 60.0), evader_velocity=(0.0, 0.0)) -> None:
    for pursuer, xy in zip(env.pursuers, pursuers):
        env._reset_robot(pursuer, np.asarray(xy, dtype=float), theta=0.0)
    env._reset_robot(env.evaders[0], np.asarray(evader, dtype=float), theta=0.0)
    env.evaders[0].velocity = np.asarray(evader_velocity, dtype=float)
    env.obstacles = [Obstacle(100.0, 100.0, 1.0)]
    env._invalidate_voronoi_cache()


def test_ring_ms_angle_weight_reverts_to_uniform_when_evader_static() -> None:
    env = _env()
    evader = np.asarray([60.0, 60.0], dtype=float)
    front = np.asarray([68.0, 60.0], dtype=float)
    back = np.asarray([52.0, 60.0], dtype=float)

    assert env._ring_ms_angle_weight(front, evader, np.asarray([0.1, 0.0])) == 1.0
    assert env._ring_ms_angle_weight(back, evader, np.asarray([0.1, 0.0])) == 1.0

    assert env._ring_ms_angle_weight(front, evader, np.asarray([1.2, 0.0])) > 1.0
    assert env._ring_ms_angle_weight(back, evader, np.asarray([1.2, 0.0])) < 1.0


def test_ring_ms_target_uses_compact_reward_ring_not_detection_radius() -> None:
    env = _env()
    _place(
        env,
        pursuers=[(42.0, 60.0), (20.0, 20.0), (20.0, 100.0), (100.0, 20.0)],
        evader=(60.0, 60.0),
        evader_velocity=(1.2, 0.0),
    )
    before_p = np.asarray([env._position(p) for p in env.pursuers], dtype=float)
    before_e = np.asarray([env._position(e) for e in env.evaders], dtype=float)
    before_v = np.asarray([env.evaders[0].velocity], dtype=float)

    target = env._ring_ms_target(0, 0, before_p, before_e, before_v)

    assert target is not None
    target_radius = float(np.linalg.norm(target - before_e[0]))
    assert 6.7 <= target_radius <= 10.5
    assert abs(target_radius - 8.0) < 0.75


def test_ring_ms_vct_ls_direct_visibility_candidates_are_self_local() -> None:
    env = _env(_config(voradj={"enemy_sensing_radius": 8.0}))
    _place(
        env,
        pursuers=[(50.0, 60.0), (35.0, 60.0), (20.0, 100.0), (100.0, 20.0)],
        evader=(60.0, 60.0),
        evader_velocity=(1.0, 0.0),
    )
    before_p = np.asarray([env._position(p) for p in env.pursuers], dtype=float)
    before_e = np.asarray([env._position(e) for e in env.evaders], dtype=float)

    assert env._vct_ls_direct_enemy_ids_for_pursuer(0, before_p, before_e) == [0]
    assert env._vct_ls_direct_enemy_ids_for_pursuer(1, before_p, before_e) == []


def test_ring_ms_reward_is_positive_when_moving_toward_pre_action_target() -> None:
    env = _env()
    _place(
        env,
        pursuers=[(42.0, 60.0), (20.0, 20.0), (20.0, 100.0), (100.0, 20.0)],
        evader=(60.0, 60.0),
        evader_velocity=(1.2, 0.0),
    )
    before_p = np.asarray([env._position(p) for p in env.pursuers], dtype=float)
    before_e = np.asarray([env._position(e) for e in env.evaders], dtype=float)
    before_v = np.asarray([env.evaders[0].velocity], dtype=float)
    target = env._ring_ms_target(0, 0, before_p, before_e, before_v)
    assert target is not None
    direction = target - before_p[0]
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    after_p = before_p.copy()
    after_p[0] = before_p[0] + direction

    reward = env._ring_importance_ms_reward(0, 0, before_p, before_e, before_v, after_p)

    assert reward > 0.0


def test_ring_ms_phase_occupancy_can_mask_occupied_front_sector() -> None:
    env = _env(
        _config(
            reward={
                "ring_ms_occupancy_mode": "phase_sector",
                "ring_ms_phase_occupancy_width": np.deg2rad(35.0),
                "ring_ms_phase_occupancy_radial_margin": 4.0,
            }
        )
    )
    _place(
        env,
        pursuers=[(42.0, 60.0), (68.0, 60.0), (20.0, 100.0), (100.0, 20.0)],
        evader=(60.0, 60.0),
        evader_velocity=(1.2, 0.0),
    )
    before_p = np.asarray([env._position(p) for p in env.pursuers], dtype=float)
    before_e = np.asarray([env._position(e) for e in env.evaders], dtype=float)
    before_v = np.asarray([env.evaders[0].velocity], dtype=float)

    target = env._ring_ms_target(0, 0, before_p, before_e, before_v)

    assert target is not None
    # With another pursuer occupying the evader-front phase, the target should not stay on that exact ray.
    assert abs(float(target[1] - before_e[0][1])) > 0.5 or float(target[0]) < before_e[0][0]
