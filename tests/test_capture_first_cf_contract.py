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
CF2 = CONFIG_DIR / "legacy_voradj_cf2_capture_first_global_support_full_4p1e1obs_100k_aw.yaml"
CF3 = CONFIG_DIR / "legacy_voradj_cf3_capture_first_local_support_full_4p1e1obs_100k_aw.yaml"


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
    assert config["reward"]["capture_stationary_enabled"] is True
    assert config["reward"]["capture_stationary_speed_threshold"] == 0.2
    assert config["reward"]["capture_stationary_hold_steps"] == 10
    assert config["reward"]["capture_stationary_min_pursuers"] == 2
    assert config["reward"]["min_active_pursuers"] == 2
    assert config["reward"]["coverage_ce_min_active_pursuers"] == 2
    assert config["voradj"]["is_pursuing_release_delay_steps"] == 10
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



def test_cf0_gate_is_diagnostic_only_and_always_starts_cf2_scratch() -> None:
    from tools.supervise_capture_first_cf import evaluate_gate, training_command

    fail = evaluate_gate([], capture_events=0)
    assert fail["controls_training_branch"] is False
    assert fail["next_action"] == "CF0_STOPS_AT_100K_AND_CF2_STARTS_FROM_SCRATCH"
    command = training_command(fail)
    assert command[command.index("--total-steps") + 1] == "100000"
    assert command[command.index("--device") + 1] == "cuda:0"
    assert str(CF2) == command[command.index("--config") + 1]
    assert "--resume-checkpoint" not in command
    assert "--resume-replay" not in command
    passed = evaluate_gate([], capture_events=1)
    assert training_command(passed) == command


def test_c1_225k_handoff_starts_cf1_from_scratch_on_gpu1() -> None:
    from tools.supervise_c1_225k_to_cf1 import cf1_command

    command = cf1_command()
    assert command[command.index("--device") + 1] == "cuda:1"
    assert command[command.index("--total-steps") + 1] == "100000"
    assert "--resume-checkpoint" not in command
    assert "--resume-replay" not in command
    assert "--resume-step" not in command


def _place_legacy_chain(env: VorAdjEnv) -> tuple[dict, list[str]]:
    for robot, xy in zip(
        env.pursuers,
        [(28.0, 30.0), (40.0, 30.0), (55.0, 30.0), (70.0, 30.0)],
    ):
        env._reset_robot(robot, np.asarray(xy, dtype=float), theta=0.0)
    env._reset_robot(env.evaders[0], np.asarray((20.0, 30.0), dtype=float), theta=0.0)
    env.obstacles = []
    env._invalidate_voronoi_cache()
    data = env._capture_voronoi_map()
    raw_labels = env._raw_task_labels_from_map(data)
    env.last_task_labels = env._task_labels_from_map(
        data,
        update_effective=True,
        raw_labels=raw_labels,
    )
    return data, raw_labels


def test_cf2_contract_is_cf1_plus_legacy_support_full_capture_and_coverage() -> None:
    cf1 = resolve_ladder_config(CF1)
    cf2 = resolve_ladder_config(CF2)
    assert cf2["perception"]["global_evader_visibility"] is True
    assert cf2["voradj"]["perception_topology_version"] == "legacy_voradj"
    assert cf2["voradj"]["support_reward_blend_enabled"] is True
    assert cf2["voradj"]["legacy_voradj_support_reward_blend_enabled"] is True
    assert cf2["voradj"]["support_reward_capture_weight"] == 1.0
    assert cf2["voradj"]["support_reward_coverage_weight"] == 1.0
    assert cf2["voradj"]["support_reward_capture_component_mode"] == "capture_task"
    assert cf2["training"]["scene_cycle"] == ["capture"]
    assert cf2["training"]["total_env_steps"] == 100000
    for key in ("update_every_env_steps", "gradient_steps", "batch_size", "grad_clip_norm"):
        assert cf2["training"][key] == cf1["training"][key]
    for key in ("actor_lr", "critic_lr", "alpha_lr", "tau"):
        assert cf2["masac"][key] == cf1["masac"][key]
    assert cf1["voradj"].get("support_reward_blend_enabled", False) is False
    assert cf1["voradj"].get("legacy_voradj_support_reward_blend_enabled", False) is False


def test_cf2_global_broadcast_does_not_change_legacy_adjacency_roles_or_rewards() -> None:
    config = scene_config(resolve_ladder_config(CF2), "capture")
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=2026081303)
    env.reset()
    data, raw_labels = _place_legacy_chain(env)

    assert data["adjacency"][("pursuer", 0)] >= {("evader", 0), ("pursuer", 1)}
    assert data["adjacency"][("pursuer", 1)] >= {("pursuer", 0), ("pursuer", 2)}
    assert data["adjacency"][("pursuer", 2)] >= {("pursuer", 1)}
    assert raw_labels[:3] == ["capture", "coverage", "coverage"]
    assert [env._task_reward_role(i, raw_labels, data) for i in range(3)] == [
        "capture",
        "support",
        "coverage",
    ]

    evader_offset = 1 + env.per_cfg["max_pursuer_num"]
    assert all(bool(obs["masks"][evader_offset]) for obs in env.get_observations() if obs is not None)

    result = env.step([[0.0, 0.0]] * 4, [None])
    capture = result.infos[0]["replay_metadata"]
    support = result.infos[1]["replay_metadata"]
    coverage = result.infos[2]["replay_metadata"]

    assert capture["reward_role"] == "capture"
    assert capture["reward_capture"] != 0.0
    assert capture["reward_coverage"] == 0.0
    assert capture["reward_total"] == capture["reward_capture"]

    assert support["reward_role"] == "support"
    assert support["support_reward_blend_active"] is True
    assert support["support_reward_capture_weight"] == 1.0
    assert support["support_reward_coverage_weight"] == 1.0
    assert support["reward_capture"] == support["reward_support_blend_capture"]
    assert support["reward_coverage"] == support["reward_support_blend_coverage"]
    assert support["reward_capture"] != 0.0
    assert support["reward_coverage"] != 0.0
    assert np.isclose(
        support["reward_total"],
        support["reward_capture"] + support["reward_coverage"] + support["reward_safety"],
    )

    assert coverage["reward_role"] == "coverage"
    assert coverage["support_reward_blend_active"] is False
    assert coverage["reward_capture"] == 0.0
    assert coverage["reward_coverage"] != 0.0
    assert coverage["reward_total"] == coverage["reward_coverage"]


def test_cf1_legacy_support_mode_remains_disabled() -> None:
    config = scene_config(resolve_ladder_config(CF1), "capture")
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=2026081302)
    env.reset()
    _place_legacy_chain(env)
    result = env.step([[0.0, 0.0]] * 4, [None])
    support_slot = result.infos[1]["replay_metadata"]

    assert env._legacy_voradj_support_reward_blend_enabled() is False
    assert env._vct_ls_support_reward_blend_enabled() is False
    assert support_slot["support_reward_blend_active"] is False
    assert support_slot["reward_capture"] == 0.0
    assert support_slot["reward_coverage"] != 0.0



def test_cf3_contract_differs_from_cf2_only_by_local_enemy_observation_and_audit() -> None:
    cf2 = resolve_ladder_config(CF2)
    cf3 = resolve_ladder_config(CF3)
    for config in (cf2, cf3):
        config.pop("run_name", None)
        config.pop("seed", None)
        config.pop("experiment_metadata", None)
    assert cf2["perception"]["global_evader_visibility"] is True
    assert cf3["perception"]["global_evader_visibility"] is False
    cf2["perception"]["global_evader_visibility"] = False
    assert cf2 == cf3
    assert cf3["voradj"]["legacy_voradj_support_reward_blend_enabled"] is True
    assert cf3["voradj"]["support_reward_capture_weight"] == 1.0
    assert cf3["voradj"]["support_reward_coverage_weight"] == 1.0


def test_cf3_local_support_observes_pursuing_friend_without_enemy_oracle() -> None:
    config = scene_config(resolve_ladder_config(CF3), "capture")
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=2026081304)
    env.reset()
    data, raw_labels = _place_legacy_chain(env)
    env.pursuers[0].velocity = np.asarray([0.7, -0.2], dtype=float)
    env.pursuers[1].velocity = np.asarray([-0.1, 0.3], dtype=float)
    env._invalidate_voronoi_cache()
    data = env._capture_voronoi_map()
    raw_labels = env._raw_task_labels_from_map(data)
    env.last_task_labels = env._task_labels_from_map(data, update_effective=True, raw_labels=raw_labels)
    assert [env._task_reward_role(i, raw_labels, data) for i in range(3)] == [
        "capture", "support", "coverage",
    ]

    observations = env.get_observations()
    capture_obs, support_obs, coverage_obs = observations[:3]
    assert capture_obs is not None and support_obs is not None and coverage_obs is not None
    evader_offset = 1 + env.per_cfg["max_pursuer_num"]
    assert bool(capture_obs["masks"][evader_offset])
    assert not bool(support_obs["masks"][evader_offset])
    assert not bool(coverage_obs["masks"][evader_offset])

    # P0 is the first friendly token for P1 because pursuing friends sort first.
    assert bool(support_obs["masks"][1])
    distance_scale = env._distance_scale()
    expected_position = env._robot_frame(env.pursuers[1], env._position(env.pursuers[0]), False) / distance_scale
    expected_velocity = env._robot_frame(env.pursuers[1], env.pursuers[0].velocity, True)
    np.testing.assert_allclose(support_obs["pursuers"][0, :2], expected_position, atol=1e-6)
    np.testing.assert_allclose(support_obs["pursuers"][0, 2:4], expected_velocity, atol=1e-6)
    assert support_obs["pursuers"][0, 6] == 1.0

    capture_friend_ids = env._support_pursuing_friend_ids(2, raw_labels, data)
    assert capture_friend_ids == []
    result = env.step([[0.0, 0.0]] * 4, [None])
    capture_meta, support_meta, coverage_meta = [info["replay_metadata"] for info in result.infos[:3]]
    assert capture_meta["reward_role"] == "capture"
    assert capture_meta["reward_capture"] != 0.0 and capture_meta["reward_coverage"] == 0.0
    assert support_meta["reward_role"] == "support"
    assert support_meta["support_enemy_token_visible"] is False
    assert support_meta["support_has_pursuing_friend"] is True
    assert support_meta["support_reward_capture_weight"] == 1.0
    assert support_meta["support_reward_coverage_weight"] == 1.0
    assert support_meta["reward_capture"] != 0.0
    assert support_meta["reward_coverage"] != 0.0
    assert coverage_meta["reward_role"] == "coverage"
    assert coverage_meta["support_enemy_token_visible"] is False
    assert coverage_meta["support_has_pursuing_friend"] is False
    assert coverage_meta["reward_capture"] == 0.0
    assert coverage_meta["reward_coverage"] != 0.0



def test_cf2_100k_supervisor_freezes_and_resumes_exactly_to_200k() -> None:
    from tools.supervise_cf2_continue200 import CONFIG, FROZEN, continuation_command

    command = continuation_command()
    assert command[command.index("--config") + 1] == str(CONFIG)
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--total-steps") + 1] == "200000"
    assert command[command.index("--resume-step") + 1] == "100000"
    assert command[command.index("--resume-checkpoint") + 1] == str(FROZEN / "trainer.pt")
    assert command[command.index("--resume-replay") + 1] == str(FROZEN / "replay.pkl")
    assert "--resume-fork" not in command
