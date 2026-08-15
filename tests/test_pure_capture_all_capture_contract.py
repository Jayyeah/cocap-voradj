from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.continuous.formal_config import resolve_ladder_config, scene_config
from cocap_voradj.training.trainer import set_global_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/experiments/parallel_ce_legacy_voradj_20260809"
CF3 = CONFIG_DIR / "legacy_voradj_cf3_capture_first_local_support_full_4p1e1obs_100k_aw.yaml"
PURE_CAPTURE = CONFIG_DIR / "legacy_voradj_b_pure_capture_local_allcapture_4p1e1obs_100k_aw.yaml"


def _make_env(config_path: Path, seed: int = 2026081601) -> VorAdjEnv:
    config = scene_config(resolve_ladder_config(config_path), "capture")
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=seed)
    env.reset()
    return env


def _place_chain(env: VorAdjEnv) -> tuple[dict, list[str]]:
    # P0 -> enemy, P1 -> P0, P2 -> P1. P3 is deliberately farther away.
    for robot, xy in zip(
        env.pursuers,
        [(28.0, 30.0), (40.0, 30.0), (55.0, 30.0), (70.0, 30.0)],
    ):
        env._reset_robot(robot, np.asarray(xy, dtype=float), theta=float(np.pi))
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


def test_pure_capture_config_is_strict_local_scratch_single_task_contract() -> None:
    config = resolve_ladder_config(PURE_CAPTURE)
    capture = scene_config(config, "capture")

    assert config["device"] == "cuda:1"
    assert config["seed"] == 2026081304
    assert config["training"]["total_env_steps"] == 100000
    assert config["training"]["scene_cycle"] == ["capture"]
    assert config["training"]["checkpoint_interval_env_steps"] == 25000
    assert config["training"]["diagnostic_eval_interval_env_steps"] == 25000
    assert config["training"]["grad_clip_norm"] is None
    assert config["training"]["update_every_env_steps"] == 4
    assert config["training"]["gradient_steps"] == 1
    assert config["training"]["batch_size"] == 128
    assert config["perception"]["global_evader_visibility"] is False
    assert config["voradj"]["perception_topology_version"] == "legacy_voradj"
    assert config["voradj"]["pure_capture_all_capture_enabled"] is True
    assert config["voradj"]["support_reward_blend_enabled"] is False
    assert config["voradj"]["legacy_voradj_support_reward_blend_enabled"] is False
    assert config["voradj"]["support_reward_capture_component_mode"] == "approach_only"
    assert config["reward"]["legacy_capture_reward_fallback_all_active"] is False
    assert config["reward"]["min_active_pursuers"] == 2
    assert config["reward"]["coverage_ce_min_active_pursuers"] == 2
    assert config["env"]["collision_semantics"] == "synchronized_swept_v1"
    assert capture["voradj"]["capture_episode_ends_on_capture"] is True
    assert capture["voradj"]["capture_episode_success_on_capture"] is True
    assert capture["env"]["episode_max_length"] == 1000
    assert capture["env"]["pre_capture_max_length"] == 1000
    assert config["evaluation"]["episodes_per_scene"] == 20
    assert config["evaluation"]["checkpoint_diagnostic_rollout_cap"] == 1000
    assert config["experiment_metadata"]["training_start"] == "scratch_new_replay"


def test_pure_capture_roles_and_observable_dense_reward_partition() -> None:
    env = _make_env(PURE_CAPTURE)
    data, raw_labels = _place_chain(env)

    assert raw_labels[:3] == ["capture", "coverage", "coverage"]
    assert [env._task_reward_role(i, env.last_task_labels, data) for i in range(3)] == [
        "capture",
        "support",
        "coverage",
    ]
    observations = env.get_observations()
    evader_offset = 1 + env.per_cfg["max_pursuer_num"]
    assert bool(observations[0]["masks"][evader_offset])
    assert not bool(observations[1]["masks"][evader_offset])
    assert not bool(observations[2]["masks"][evader_offset])
    # The one-hop informed pursuer sees the effective pursuing friend only.
    assert bool(observations[1]["masks"][1])
    assert observations[1]["pursuers"][0, 6] == 1.0

    result = env.step([[0.4, 0.0]] * 4, [None])
    direct, informed, uninformed = [
        info["replay_metadata"] for info in result.infos[:3]
    ]

    assert direct["capture_objective_role"] == "direct_capture"
    assert direct["reward_capture_approach"] > 0.0
    assert np.isclose(
        direct["reward_capture"],
        direct["reward_capture_approach"]
        + direct["reward_capture_mean_shift"]
        + direct["reward_capture_front"],
    )
    assert direct["reward_coverage"] == 0.0

    assert informed["capture_objective_role"] == "one_hop_informed"
    assert informed["capture_objective_target_resolved"] is True
    assert informed["capture_objective_friend_count"] >= 1
    assert informed["support_enemy_token_visible"] is False
    assert informed["support_has_pursuing_friend"] is True
    assert informed["reward_capture_approach"] > 0.0
    assert informed["reward_capture_mean_shift"] == 0.0
    assert informed["reward_capture_front"] == 0.0
    assert informed["reward_capture"] == informed["reward_capture_approach"]
    assert informed["reward_coverage"] == 0.0

    assert uninformed["capture_objective_role"] == "uninformed"
    assert uninformed["support_enemy_token_visible"] is False
    assert uninformed["support_has_pursuing_friend"] is False
    assert uninformed["reward_capture"] == 0.0
    assert uninformed["reward_capture_approach"] == 0.0
    assert uninformed["reward_capture_mean_shift"] == 0.0
    assert uninformed["reward_capture_front"] == 0.0
    assert uninformed["reward_coverage"] == 0.0
    assert uninformed["reward_ce_pbrs"] == 0.0

    metrics = env.last_voradj_metrics
    assert metrics["pure_capture_all_capture_enabled"] is True
    assert metrics["pure_capture_role_direct_capture_agent_count"] == 1
    assert metrics["pure_capture_role_one_hop_informed_agent_count"] >= 1
    assert metrics["pure_capture_role_uninformed_coverage_sum"] == 0.0
    assert env.last_reward_terms["hold_reward_eligible_count"] == 0


def test_k10_held_self_without_enemy_is_never_mislabeled_direct_or_oracle_rewarded() -> None:
    env = _make_env(PURE_CAPTURE)
    _place_chain(env)
    adjacency = {
        ("pursuer", 0): {("pursuer", 1)},
        ("pursuer", 1): {("pursuer", 0), ("evader", 0)},
        ("evader", 0): {("pursuer", 1)},
    }
    data = {"adjacency": adjacency}
    raw_labels = ["coverage", "capture", "coverage", "coverage"]
    effective_labels = ["capture", "capture", "coverage", "coverage"]

    # Core corrected-contract role remains K10 capture, while the B reward
    # partition admits only the target resolvable through the pursuing friend.
    assert env._task_reward_role(0, effective_labels, data) == "capture"
    role, targets, friends = env._pure_capture_observable_partition(
        0, raw_labels, effective_labels, data
    )
    assert role == "one_hop_informed"
    assert targets == [0]
    assert friends == [1]

    # Partition labels describe the action-selection state even if this step's
    # movement subsequently deactivated an entity.
    env.pursuers[1].deactivated = True
    env.evaders[0].deactivated = True
    role, targets, friends = env._pure_capture_observable_partition(
        1, raw_labels, effective_labels, data
    )
    assert role == "direct_capture"
    assert targets == [0]
    assert friends == []

    # If the effective friend is itself only K10-held and has no current enemy
    # edge, the self receives no oracle target and is explicitly uninformed.
    no_target_data = {
        "adjacency": {
            ("pursuer", 0): {("pursuer", 1)},
            ("pursuer", 1): {("pursuer", 0)},
        }
    }
    role, targets, friends = env._pure_capture_observable_partition(
        0,
        ["coverage"] * 4,
        effective_labels,
        no_target_data,
    )
    assert role == "uninformed"
    assert targets == []
    assert friends == []


@pytest.mark.parametrize("capture_type", ["normal", "stationary"])
def test_pure_capture_terminal_reward_is_shared_once_with_every_active_pursuer(
    monkeypatch: pytest.MonkeyPatch,
    capture_type: str,
) -> None:
    env = _make_env(PURE_CAPTURE)
    _place_chain(env)
    event = {
        "evader_id": 0,
        "participants": [0, 1, 2],
        "angles": [0.0, 2.0, 4.0],
        "capture_type": capture_type,
    }
    monkeypatch.setattr(env, "_loose_capture_events", lambda: [event])

    result = env.step([[0.0, 0.0]] * 4, [None])
    terminal_rewards = [
        info["replay_metadata"]["reward_terminal"] for info in result.infos
    ]
    assert all(value > 0.0 for value in terminal_rewards)
    assert len(set(np.round(terminal_rewards, 10))) == 1
    assert all(
        info["replay_metadata"]["pure_capture_team_terminal_shared"]
        for info in result.infos
    )
    assert env.last_voradj_metrics["pure_capture_terminal_shared_recipient_count"] == 4
    assert all(result.dones)


def test_absent_and_explicitly_disabled_gate_leave_cf3_outputs_identical() -> None:
    base = scene_config(resolve_ladder_config(CF3), "capture")
    disabled = copy.deepcopy(base)
    disabled.setdefault("voradj", {})["pure_capture_all_capture_enabled"] = False

    env_a = VorAdjEnv(copy.deepcopy(base), seed=17)
    env_b = VorAdjEnv(copy.deepcopy(disabled), seed=17)
    env_a.reset()
    env_b.reset()
    _place_chain(env_a)
    _place_chain(env_b)
    actions = [[0.2, 0.0]] * 4
    result_a = env_a.step(actions, [None])
    result_b = env_b.step(actions, [None])

    np.testing.assert_array_equal(result_a.rewards, result_b.rewards)
    assert result_a.dones == result_b.dones
    assert result_a.infos == result_b.infos
    assert env_a.last_reward_terms == env_b.last_reward_terms
    assert env_a.last_voradj_metrics == env_b.last_voradj_metrics


def test_pure_capture_role_reward_and_angular_metrics_are_persisted() -> None:
    from tools.run_continuous_ctde_training import (
        _diagnostic_eval_contract,
        _metrics_record,
        _role_reward_rows,
    )

    pure_config = resolve_ladder_config(PURE_CAPTURE)
    cf3_config = resolve_ladder_config(CF3)
    assert _diagnostic_eval_contract(pure_config, None) == (20, 1000)
    assert _diagnostic_eval_contract(pure_config, 3) == (3, 1000)
    assert _diagnostic_eval_contract(cf3_config, None) == (4, 400)

    env = _make_env(PURE_CAPTURE)
    _place_chain(env)
    result = env.step([[0.4, 0.0]] * 4, [None])
    rows = _role_reward_rows(result.infos, np.ones(4, dtype=bool))

    class _Replay:
        focal_index_sizes: dict[str, int] = {}

        def __len__(self) -> int:
            return 1

    geometry = [
        {
            "d1": 8.5,
            "d2": 9.0,
            "d3": 20.0,
            "d4": 30.0,
            "closing": 0.1,
            "fraction_closing": 0.5,
            "abs_bearing_error": 0.2,
            "turn_direction_correct_rate": 1.0,
            "num_within_8": 0,
            "num_in_ring_8_10_5": 2,
            "ring_largest_angular_gap_rad": 4.0,
            "ring_pairwise_angular_separation_mean_rad": 2.0,
            "ring_pairwise_angular_separation_min_rad": 2.0,
        }
    ]
    record = _metrics_record(
        step=25000,
        update_count=1,
        window_updates=[],
        replay=_Replay(),
        sampling_stats={},
        action_norms=[],
        speeds=[],
        terminated_count=0,
        truncated_count=0,
        collision_count=0,
        scene_counts={"capture": 1},
        origin_counts={"map_random": 1},
        geometry=geometry,
        role_rewards=rows,
        collision_types_by_step=[["agent_agent"], [], ["obstacle"]],
        window_wall_time_s=1.0,
        window_env_steps=1,
    )

    assert record["pure_capture_role_direct_capture_agent_steps"] == 1
    assert record["pure_capture_role_one_hop_informed_agent_steps"] >= 1
    assert record["pure_capture_role_uninformed_reward_coverage_mean"] == 0.0
    assert record["pure_capture_role_one_hop_informed_reward_capture_mean_shift_mean"] == 0.0
    assert record["ring_2plus_angular_frame_count"] == 1
    assert record["ring_largest_angular_gap_rad_mean"] == 4.0
    assert record["ring_pairwise_angular_separation_min_rad_p50"] == 2.0
    assert record["agent_agent_collision_transition_count"] == 1
    assert record["obstacle_collision_transition_count"] == 1
