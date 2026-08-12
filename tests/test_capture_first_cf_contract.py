from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.continuous.formal_config import resolve_ladder_config, scene_config
from cocap_voradj.training.trainer import set_global_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/experiments/parallel_ce_legacy_voradj_20260809"
CF0 = CONFIG_DIR / "legacy_voradj_cf0_capture_first_local_4p1e1obs_100k_aw.yaml"
CF1 = CONFIG_DIR / "legacy_voradj_cf1_capture_first_global_enemy_4p1e1obs_100k_aw.yaml"


def test_cf0_is_formal_capture_only_with_stable_sac_contract() -> None:
    config = resolve_ladder_config(CF0)
    capture = scene_config(config, "capture")
    assert config["training"]["scene_cycle"] == ["capture"]
    assert config["training"]["total_env_steps"] == 100000
    assert config["training"]["update_every_env_steps"] == 4
    assert config["training"]["gradient_steps"] == 1
    assert config["training"]["batch_size"] == 128
    assert config["training"]["grad_clip_norm"] is None
    assert config["training"]["optimizer_unit"] == "joint_transition_all_active_agents"
    assert config["training"]["replay_sampling"] == "uniform_joint"
    assert config["masac"]["actor_lr"] == 1e-4
    assert config["masac"]["critic_lr"] == 1e-4
    assert config["masac"]["tau"] == 0.005
    assert config["action"]["mode"] == "acceleration_angular_velocity_body"
    assert config["perception"]["global_evader_visibility"] is False
    assert config["voradj"]["perception_topology_version"] == "legacy_voradj"
    assert config["reward"]["capture_reward_mode"] == "legacy"
    assert config["reward"]["k_required"] == 3
    assert config["reward"]["legacy_capture_reward_fallback_all_active"] is False
    assert capture["env"]["num_pursuers"] == 4
    assert capture["env"]["num_evaders"] == 1
    assert capture["env"]["num_obstacles"] == 1
    assert capture["env"]["pre_capture_max_length"] == 1000
    assert capture["env"]["episode_max_length"] == 1000
    assert capture["evader"]["autonomous"] is True
    assert capture["voradj"]["capture_episode_ends_on_capture"] is True
    assert capture["voradj"]["capture_episode_success_on_capture"] is True


def test_cf1_only_changes_enemy_broadcast_and_audit_metadata() -> None:
    local = resolve_ladder_config(CF0)
    broadcast = resolve_ladder_config(CF1)
    for config in (local, broadcast):
        config.pop("run_name", None)
        config.pop("seed", None)
        config.pop("experiment_metadata", None)
    local["perception"]["global_evader_visibility"] = True
    assert local == broadcast


def test_legacy_global_broadcast_has_enemy_position_velocity_for_every_active_agent() -> None:
    config = scene_config(resolve_ladder_config(CF1), "capture")
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=2026081302)
    env.reset()
    evader = env.evaders[0]
    evader.x, evader.y = 65.0, 60.0
    evader.velocity = np.asarray([1.25, -0.75], dtype=float)
    for index, pursuer in enumerate(env.pursuers):
        pursuer.x, pursuer.y = 5.0 + index, 6.0 + index
        pursuer.theta = 0.15 * index
        pursuer.deactivated = False
    env._invalidate_voronoi_cache()
    distance_scale = env._distance_scale()
    evader_mask_offset = 1 + env.per_cfg["max_pursuer_num"]
    invisible_count = 0
    for pursuer, obs in zip(env.pursuers, env.get_observations()):
        assert obs is not None and bool(obs["masks"][evader_mask_offset])
        relative = np.asarray([evader.x - pursuer.x, evader.y - pursuer.y], dtype=float)
        expected_position = env._robot_frame(pursuer, np.asarray([evader.x, evader.y], dtype=float), False) / distance_scale
        expected_velocity = env._robot_frame(pursuer, evader.velocity, True)
        np.testing.assert_allclose(obs["evaders"][0, :2], expected_position, atol=1e-6)
        np.testing.assert_allclose(obs["evaders"][0, 2:4], expected_velocity, atol=1e-6)


def test_cf0_invisible_agents_have_no_enemy_token_or_capture_candidate() -> None:
    config = scene_config(resolve_ladder_config(CF0), "capture")
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=2026081301)
    env.reset()
    env.evaders[0].x, env.evaders[0].y = 95.0, 95.0
    for index, pursuer in enumerate(env.pursuers):
        pursuer.x, pursuer.y = 5.0 + index, 5.0
        pursuer.deactivated = False
    env._invalidate_voronoi_cache()
    evader_mask_offset = 1 + env.per_cfg["max_pursuer_num"]
    invisible_count = 0
    for index, obs in enumerate(env.get_observations()):
        assert obs is not None
        if not bool(obs["masks"][evader_mask_offset]):
            invisible_count += 1
        assert env._vct_ls_direct_enemy_ids_for_pursuer(index) == []
    assert invisible_count >= 1



def test_cf0_gate_uses_real_capture_events_not_collision_terminations() -> None:
    from tools.supervise_capture_first_cf import evaluate_gate

    rows = [
        {
            "step": step,
            "window_env_steps": 1000,
            "terminated_count": 4,
            "collision_count": 4,
            "d1_mean": 20.0,
            "d1_min": 10.0,
            "fraction_steps_any_in_ring": 0.0,
            "fraction_steps_2plus_in_ring": 0.0,
            "fraction_steps_3plus_in_ring": 0.0,
            "max_num_in_ring": 0,
        }
        for step in range(1000, 100001, 1000)
    ]
    assert evaluate_gate(rows, capture_events=0)["passed"] is False
    result = evaluate_gate(rows, capture_events=1)
    assert result["passed"] is True
    assert result["strong_signals"]["real_capture"] is True


def test_representative_rollout_selection_is_unique_and_bounded() -> None:
    from tools.run_representative_masac_rollouts import choose

    rows = [
        {
            "seed": seed,
            "captured": False,
            "episode_success": False,
            "collision_event": seed % 2 == 0,
            "min_min_distance": float(seed),
            "distance_progress": float(10 - seed),
        }
        for seed in range(20)
    ]
    selected = choose(rows, 5)
    assert len(selected) == 5
    assert len({int(row["seed"]) for row, _reason in selected}) == 5
