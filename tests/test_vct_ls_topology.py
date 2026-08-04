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

from tools.rollout_voradj_visual import load_config as load_rollout_config, snapshot_env, snapshot_voronoi_data

A3_CONFIG = (
    ROOT
    / "configs/experiments/voradj_a3_apfnew_sqrtn_20260723"
    / "a3_apfnew_sqrtn_4v1_scratch_mix_2m.yaml"
)
FINAL_CONFIG = (
    ROOT
    / "configs/experiments/cr_ms_support_approach_ce_curriculum_20260802"
    / "stage1_4p1e1obs_scratch2m.yaml"
)


def _config(**overrides):
    base = load_config(str(A3_CONFIG))
    patch = {
        "device": "cpu",
        "env": {
            "num_pursuers": 4,
            "num_evaders": 1,
            "num_obstacles": 1,
            "episode_max_length": 20,
            "init_speed": 0.0,
        },
        "reward": {
            "voradj_grid_size": 24,
            "min_active_pursuers": 4,
        },
        "voradj": {
            "perception_topology_version": "friendly_voronoi_comm_v0",
            "friendly_comm_obstacle_mode": "free_mask_projected",
            "voronoi_obstacle_mode": "free_mask_projected",
            "enemy_sensing_radius": 8.0,
            "obstacle_sensing_radius": 6.0,
            "local_sensing_uses_surface_distance": True,
            "is_pursuing_release_delay_steps": 10,
            "vct_ls_apply_release_delay": True,
        },
        "zone_demo": {"enabled": False},
    }
    return deep_update(deep_update(base, patch), overrides)


def _env(config):
    set_global_config(config)
    env = VorAdjEnv(config, seed=20260730)
    env.reset()
    return env


def _place_square_with_evader(env: VorAdjEnv, evader_xy=(40.0, 30.0)) -> None:
    positions = [
        (30.0, 30.0),
        (60.0, 30.0),
        (30.0, 60.0),
        (60.0, 60.0),
    ]
    for pursuer, xy in zip(env.pursuers, positions):
        env._reset_robot(pursuer, np.asarray(xy, dtype=float), theta=0.0)
    env._reset_robot(env.evaders[0], np.asarray(evader_xy, dtype=float), theta=0.0)
    env.obstacles = [Obstacle(90.0, 90.0, 2.0)]
    env._invalidate_voronoi_cache()


def test_vct_ls_capture_graph_uses_pursuers_only_and_free_mask() -> None:
    env = _env(_config())
    _place_square_with_evader(env, evader_xy=(80.0, 30.0))

    capture_data = env._capture_voronoi_map()
    coverage_data = env._coverage_voronoi_map()

    assert env._vct_ls_enabled() is True
    assert capture_data is env._capture_voronoi_map()
    assert coverage_data is env._coverage_voronoi_map()
    assert capture_data["site_scope"] == "pursuer_only"
    assert coverage_data["site_scope"] == "pursuer_only"
    assert all(key[0] == "pursuer" for key in capture_data["keys"])
    assert all(key[0] == "pursuer" for key in coverage_data["keys"])
    assert capture_data["voronoi_obstacle_mode"] == "free_mask_projected"
    assert capture_data["blocked_grid_count"] > 0


def test_legacy_voradj_still_uses_evaders_as_voronoi_sites() -> None:
    config = _config(
        voradj={
            "perception_topology_version": "legacy_voradj",
            "capture_adjacency_obstacle_mode": "legacy_assign",
            "voronoi_obstacle_mode": "legacy_assign",
        }
    )
    env = _env(config)
    _place_square_with_evader(env, evader_xy=(80.0, 30.0))

    capture_data = env._capture_voronoi_map()

    assert env._vct_ls_enabled() is False
    assert capture_data["site_scope"] == "all_active_entities"
    assert any(key[0] == "evader" for key in capture_data["keys"])
    assert "blocked_grid_count" not in capture_data


def test_vct_ls_direct_detection_and_one_hop_support_tokens() -> None:
    env = _env(_config())
    _place_square_with_evader(env, evader_xy=(40.0, 30.0))
    data = env._capture_voronoi_map()
    labels = env._task_labels_from_map(data, update_effective=True)
    obs = env.get_observations()

    max_p = int(env.per_cfg.get("max_pursuer_num", 8))
    max_e = int(env.per_cfg.get("max_evader_num", 8))
    evader_slice = slice(1 + max_p, 1 + max_p + max_e)

    assert env._vct_ls_direct_enemy_ids_for_pursuer(0) == [0]
    assert env._vct_ls_direct_enemy_ids_for_pursuer(1) == []
    assert labels[0] == "capture"
    assert labels[1] == "coverage"
    assert obs[0] is not None and obs[1] is not None
    assert obs[0]["self"][-1] == 1.0
    assert obs[1]["self"][-1] == 0.0
    assert np.any(obs[0]["masks"][evader_slice])
    assert not np.any(obs[1]["masks"][evader_slice])
    assert np.any(obs[1]["pursuers"][:, -1] == 1.0)

    result = env.step([4, 4, 4, 4], [4])
    assert result.infos[1]["replay_metadata"]["support_candidate"] is True
    assert result.infos[1]["replay_metadata"]["vct_ls_direct_enemy_count"] == 0
    assert env.last_voradj_metrics["vct_ls_enabled"] is True
    assert env.last_voradj_metrics["friendly_comm_include_evaders_as_sites"] is False


def test_vct_ls_support_reward_blend_is_opt_in() -> None:
    env = _env(
        _config(
            voradj={
                "support_reward_blend_enabled": True,
                "support_reward_capture_weight": 0.5,
                "support_reward_coverage_weight": 0.5,
                "support_reward_capture_target_mode": "neighbor_visible",
            }
        )
    )
    _place_square_with_evader(env, evader_xy=(40.0, 30.0))
    data = env._capture_voronoi_map()
    env._task_labels_from_map(data, update_effective=True)

    result = env.step([4, 4, 4, 4], [4])
    metadata = result.infos[1]["replay_metadata"]

    assert metadata["task_label"] == "coverage"
    assert metadata["support_candidate"] is True
    assert metadata["vct_ls_direct_enemy_count"] == 0
    assert metadata["support_reward_blend_active"] is True
    assert metadata["support_reward_capture_weight"] == 0.5
    assert metadata["support_reward_coverage_weight"] == 0.5
    assert metadata["reward_capture"] == metadata["reward_support_blend_capture"]
    assert metadata["reward_coverage"] == metadata["reward_support_blend_coverage"]
    assert env.last_voradj_metrics["support_reward_blend_enabled"] is True
    assert env.last_voradj_metrics["support_reward_blend_active_count"] >= 1

def test_cr_ms_support_can_use_legacy_approach_only_component() -> None:
    env = _env(
        _config(
            reward={
                "capture_reward_mode": "ring_importance_ms_v0",
                "capture_timestep_penalty": 0.0,
                "omega_ring_ms": 2.0,
                "omega_approach": 0.0,
            },
            voradj={
                "support_reward_blend_enabled": True,
                "support_reward_capture_weight": 0.5,
                "support_reward_coverage_weight": 0.5,
                "support_reward_capture_target_mode": "neighbor_visible",
                "support_reward_capture_component_mode": "approach_only",
                "support_reward_approach_weight": 1.0,
                "support_reward_approach_clip": 3.0,
            },
        )
    )
    _place_square_with_evader(env, evader_xy=(40.0, 30.0))
    # Pursuer 1 is outside its own sensing radius but is a one-hop support
    # neighbor. Point it toward the neighbor-visible evader and accelerate.
    env._reset_robot(env.pursuers[1], np.asarray((60.0, 30.0), dtype=float), theta=np.pi)
    env._invalidate_voronoi_cache()
    data = env._capture_voronoi_map()
    env._task_labels_from_map(data, update_effective=True)

    support_distance_before = np.linalg.norm(env._position(env.pursuers[1]) - env._position(env.evaders[0]))
    result = env.step([4, 7, 4, 4], [4])
    metadata = result.infos[1]["replay_metadata"]

    assert metadata["task_label"] == "coverage"
    assert metadata["vct_ls_direct_enemy_count"] == 0
    assert metadata["support_reward_blend_active"] is True
    assert metadata["support_reward_capture_component_mode"] == "approach_only"
    assert metadata["reward_support_blend_capture"] > 0.0
    support_distance_after = np.linalg.norm(env._position(env.pursuers[1]) - env._position(env.evaders[0]))
    expected_capture_component = 0.5 * np.clip(support_distance_before - support_distance_after, -3.0, 3.0)
    assert np.isclose(metadata["reward_support_blend_capture"], expected_capture_component)


def test_vct_ls_obstacle_tokens_are_local_radius_only() -> None:
    env = _env(_config(voradj={"obstacle_sensing_radius": 4.0}))
    _place_square_with_evader(env, evader_xy=(80.0, 30.0))
    env.obstacles = [Obstacle(55.0, 30.0, 1.0)]
    env._invalidate_voronoi_cache()

    obs = env.get_observations()
    max_p = int(env.per_cfg.get("max_pursuer_num", 8))
    max_e = int(env.per_cfg.get("max_evader_num", 8))
    obstacle_slice = slice(1 + max_p + max_e, None)

    assert obs[0] is not None and obs[1] is not None
    assert not np.any(obs[0]["masks"][obstacle_slice])
    assert np.any(obs[1]["masks"][obstacle_slice])


def test_final_config_enables_vct_ls_without_enabling_zone() -> None:
    config = load_config(str(FINAL_CONFIG))

    assert config["run_name"] == "crms_supportapproach_ce_curr_stage1_4p1e1obs_scratch2m_20260802_run1"
    assert config["device"] == "cuda:0"
    assert config["total_timesteps"] == 2_000_000
    assert config["train_mode"] == "voradj_mixed_coverage"
    assert config["env"]["num_pursuers"] == 4
    assert config["env"]["num_evaders"] == 1
    assert config["env"]["num_obstacles"] == 1
    assert config.get("zone_demo", {}).get("enabled", False) is False
    assert config["reward"]["coverage_objective_version"] == "centroid_energy_v0"

    voradj = config["voradj"]
    assert voradj["perception_topology_version"] == "friendly_voronoi_comm_v0"
    assert voradj["friendly_comm_obstacle_mode"] == "free_mask_projected"
    assert voradj["enemy_sensing_radius"] == 20.0
    assert voradj["obstacle_sensing_radius"] == 20.0
    assert voradj["local_sensing_uses_surface_distance"] is True
    assert voradj["vct_ls_apply_release_delay"] is True
    assert voradj["center_sqrt_n_normalization_enabled"] is True

    assert config.get("pretrained", {}).get("path") is None


def test_vct_ls_rollout_tool_loads_final_extends_config() -> None:
    config = load_rollout_config(FINAL_CONFIG)

    assert config["pursuer"]["max_speed"] == 3.0
    assert config["evader"]["max_speed"] == 3.5
    assert config["voradj"]["perception_topology_version"] == "friendly_voronoi_comm_v0"
    assert config["voradj"]["vct_ls_apply_release_delay"] is True

def test_vct_ls_visual_snapshot_uses_phase_specific_pursuer_only_topology() -> None:
    env = _env(_config())
    _place_square_with_evader(env, evader_xy=(80.0, 30.0))

    capture_data = snapshot_voronoi_data(env, "capture")
    coverage_data = snapshot_voronoi_data(env, "coverage")

    assert capture_data["site_scope"] == "pursuer_only"
    assert coverage_data["site_scope"] == "pursuer_only"
    assert not any(key[0] == "evader" for key in capture_data["keys"])
    assert not any(key[0] == "evader" for key in coverage_data["keys"])


def test_vct_ls_visual_snapshot_records_sensing_circle_metadata() -> None:
    env = _env(_config())
    _place_square_with_evader(env, evader_xy=(80.0, 30.0))

    snapshot = snapshot_env(env, "mix", "capture", 0, 0, 0)

    assert snapshot["sensing"] == {
        "vct_ls_enabled": True,
        "enemy_sensing_radius": 8.0,
        "obstacle_sensing_radius": 6.0,
        "surface_sensing": True,
    }
    active_pursuers = [item for item in snapshot["pursuers"] if item["active"]]
    assert len(active_pursuers) == 4
    assert all(item["sensing_radius"] == 8.0 for item in active_pursuers)
    assert all(item["enemy_sensing_radius"] == 8.0 for item in active_pursuers)
    assert all(item["obstacle_sensing_radius"] == 6.0 for item in active_pursuers)
