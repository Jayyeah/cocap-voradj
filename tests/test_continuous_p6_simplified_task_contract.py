from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.trainer import load_config
from tools.run_continuous_p6_screening import (
    SCENE_SETS,
    FORMAL_TRAINER_CONFIG_PATH,
    _contract,
    _make_trainer,
    _scene_config,
    _derive_transition_metadata,
    _split_termination_flags,
    _stack_obs_with_padding,
)


def test_simplified_scene_sets_are_explicit() -> None:
    assert SCENE_SETS["full"] == ("pure_ce", "capture", "mixed_crms")
    assert SCENE_SETS["pure_coverage"] == ("pure_ce",)
    assert SCENE_SETS["broadcast_capture"] == ("capture",)
    assert SCENE_SETS["broadcast_mixed"] == ("mixed_crms",)


def test_formal_p6_profile_is_explicit_and_frozen() -> None:
    config = load_config(str(FORMAL_TRAINER_CONFIG_PATH))
    assert config["training_profile"] == "formal_p6"
    assert config["continuous_encoder"]["hidden_dim"] == 256
    assert config["continuous_encoder"]["num_layers"] == 4
    assert config["actor"]["hidden_dim"] == 256
    assert config["local_sac"]["actor_lr"] == 1e-4
    assert config["warmup_steps"] == 5000
    assert config["grad_clip_norm"] == 10.0
    trainer = _make_trainer("local", 0.8, "cpu", trainer_profile="formal_p6")
    assert trainer.training_profile == "formal_p6"
    assert trainer.warmup_steps == 5000
    assert trainer.config.hidden_dim == 256
    assert trainer.config.actor_lr == 1e-4
    assert trainer.config.grad_clip_norm == 10.0


def test_pure_coverage_ablation_keeps_acceleration_dynamics() -> None:
    config = _scene_config("pure_ce", 0.8, coverage_control_ablation=True)
    assert config["action_mode"] == "acceleration_2d_body"
    assert config["env"]["num_evaders"] == 0
    assert config["v_max"] == 3.0
    reward = config["reward"]
    assert reward["coverage_ce_speed_weight"] == 0.0
    assert reward["coverage_ce_acceleration_weight"] == 0.0
    assert reward["coverage_ce_angular_velocity_weight"] == 0.0
    assert reward["coverage_cell_center_speed_penalty_enabled"] is False


def test_broadcast_capture_only_changes_visibility_topology() -> None:
    config = _scene_config("capture", 0.8, global_evader_visibility=True)
    assert config["action_mode"] == "acceleration_2d_body"
    assert config["env"]["num_evaders"] == 1
    assert config["perception"]["global_evader_visibility"] is True


def test_specialized_contract_cannot_cross_resume_full_run() -> None:
    base = _contract("local", 0.8, 1)
    pure = _contract("local", 0.8, 1, scene_set="pure_coverage", coverage_control_ablation=True)
    broadcast = _contract("local", 0.8, 1, scene_set="broadcast_capture", global_evader_visibility=True)
    low_noise = _contract("local", 0.8, 1, scene_set="pure_coverage", actor_log_std_max=-1.0)
    assert "scene_set" not in base
    assert pure["scene_set"] == "pure_coverage"
    assert pure["coverage_control_ablation"] is True
    assert broadcast["global_evader_visibility"] is True
    assert low_noise["actor_log_std_max"] == -1.0
    assert pure != base
    assert broadcast != base
    assert low_noise != base


def test_deactivated_next_observation_keeps_fixed_agent_slot() -> None:
    before = {
        "self": np.arange(4 * 3, dtype=np.float32).reshape(4, 3),
        "pursuers": np.ones((4, 2, 2), dtype=np.float32),
    }
    next_slots = [
        {"self": np.full(3, 2.0, dtype=np.float32), "pursuers": np.full((2, 2), 2.0, dtype=np.float32)},
        None,
        {"self": np.full(3, 3.0, dtype=np.float32), "pursuers": np.full((2, 2), 3.0, dtype=np.float32)},
        {"self": np.full(3, 4.0, dtype=np.float32), "pursuers": np.full((2, 2), 4.0, dtype=np.float32)},
    ]
    stacked = _stack_obs_with_padding(next_slots, before)
    assert stacked["self"].shape == (4, 3)
    assert stacked["pursuers"].shape == (4, 2, 2)
    np.testing.assert_array_equal(stacked["self"][1], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(stacked["pursuers"][1], np.zeros((2, 2), dtype=np.float32))


def test_deactivated_next_observation_rejects_wrong_agent_slot_shape() -> None:
    before = {"self": np.zeros((2, 3), dtype=np.float32)}
    with np.testing.assert_raises(ValueError):
        _stack_obs_with_padding(
            [{"self": np.zeros(3, dtype=np.float32)}, {"self": np.zeros(2, dtype=np.float32)}],
            before,
        )


def test_timeout_is_truncated_but_collision_and_capture_are_terminated() -> None:
    terminated, truncated = _split_termination_flags(
        [True, True, True, False],
        [
            {"state": "too long episode"},
            {"state": "deactivated after collision"},
            {"state": "capture completed"},
            {"state": "normal"},
        ],
    )
    np.testing.assert_array_equal(terminated, np.asarray([False, True, True, False]))
    np.testing.assert_array_equal(truncated, np.asarray([True, False, False, False]))


def test_replay_metadata_uses_real_state_and_event_sources() -> None:
    env = SimpleNamespace(
        evaders=[object()],
        pursuers=[],
        last_capture_events=[{"evader_id": 0}],
        last_reward_terms={"coverage_success": 1.0},
        post_capture_coverage_success=False,
        coverage_geometric_success=False,
        coverage_settled_success=False,
    )
    metadata, sources = _derive_transition_metadata(
        env,
        scene="capture",
        before_active_target=True,
        before_coverage_success=False,
        infos=[
            {
                "state": "normal",
                "replay_metadata": {
                    "phase": "pre_capture",
                    "task_label": "coverage",
                    "next_task_label": "capture",
                },
            }
        ],
    )
    assert metadata["regime"] == "active_target"
    assert metadata["coverage_only"] is False
    assert set(metadata["event_ids"]) == {"discovery", "capture", "ce_success"}
    assert sources["regime"] == "env.active_evaders_before_step"
    assert sources["event:capture"] == "env.last_capture_events"
    assert sources["event:ce_success"] == "env.coverage_success_latch"


def test_replay_metadata_marks_post_capture_as_coverage_only() -> None:
    env = SimpleNamespace(
        evaders=[SimpleNamespace(deactivated=True)],
        pursuers=[],
        last_capture_events=[],
        last_reward_terms={"coverage_success": 0.0},
        post_capture_coverage_success=False,
        coverage_geometric_success=False,
        coverage_settled_success=False,
    )
    metadata, _ = _derive_transition_metadata(
        env,
        scene="capture",
        before_active_target=False,
        before_coverage_success=False,
        infos=[
            {
                "state": "normal",
                "replay_metadata": {
                    "phase": "post_capture",
                    "task_label": "coverage",
                    "next_task_label": "coverage",
                },
            }
        ],
    )
    assert metadata["regime"] == "coverage_only"
    assert metadata["active_target"] is False
    assert metadata["event_ids"] == []


def test_runner_splits_real_continuous_timeout_as_truncation() -> None:
    config = _scene_config("pure_ce", 0.8, coverage_control_ablation=True)
    config["env"]["episode_max_length"] = 128
    env = VorAdjEnv(config, seed=2026080514)
    env.reset()
    result = None
    for _ in range(128):
        result = env.step(
            [np.zeros(2, dtype=np.float32) for _ in env.pursuers],
            [],
        )
        if all(result.dones):
            break

    assert result is not None
    assert env.episode_step == 128
    terminated, truncated = _split_termination_flags(result.dones, result.infos)
    assert all(result.dones)
    assert not bool(terminated.any())
    assert bool(truncated.all())
    assert all(info["state"] == "too long episode" for info in result.infos)


def test_runner_preserves_real_capture_to_post_capture_as_nonterminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _scene_config("capture", 0.8)
    env = VorAdjEnv(config, seed=2026080515)
    env.reset()
    env._pursuing_flags_initialized = True
    for index, pursuer in enumerate(env.pursuers):
        pursuer.is_pursuing = index != 0
    env._pursuing_release_counters = [7] * len(env.pursuers)

    calls = 0

    def capture_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            return [
                {
                    "evader_id": 0,
                    "participants": [1],
                    "angles": [0.0],
                    "capture_type": "stationary",
                }
            ]
        return []

    monkeypatch.setattr(env, "_loose_capture_events", capture_once)
    zero_actions = [np.zeros(2, dtype=np.float32) for _ in env.pursuers]
    first = env.step(zero_actions, [None])
    first_terminated, first_truncated = _split_termination_flags(first.dones, first.infos)
    assert env.evaders[0].deactivated is True
    assert not bool(first_terminated.any())
    assert not bool(first_truncated.any())
    assert first.infos[0]["replay_metadata"]["phase"] == "pre_capture"
    first_metadata, first_sources = _derive_transition_metadata(
        env,
        scene="capture",
        before_active_target=True,
        before_coverage_success=False,
        infos=first.infos,
    )
    assert "capture" in first_metadata["event_ids"]
    assert first_sources["event:capture"] == "env.last_capture_events"

    second = env.step(zero_actions, [None])
    second_terminated, second_truncated = _split_termination_flags(second.dones, second.infos)
    assert env.post_capture_started is True
    assert second.infos[0]["replay_metadata"]["phase"] == "post_capture"
    assert not bool(second_terminated.any())
    assert not bool(second_truncated.any())


def test_runner_derives_real_ce_success_event() -> None:
    config = _scene_config("pure_ce", 0.8, coverage_control_ablation=True)
    config["reward"]["coverage_ce_success_hold_steps"] = 5
    config["env"]["episode_max_length"] = 30
    env = VorAdjEnv(config, seed=2026080518)
    env.reset()
    for _ in range(5):
        data = env._voronoi_map()
        for index, pursuer in enumerate(env.pursuers):
            centroid = np.asarray(data["centroids"][("pursuer", index)], dtype=float)
            pursuer.x, pursuer.y = float(centroid[0]), float(centroid[1])
            pursuer.speed = 0.0
            pursuer.velocity = np.zeros(2, dtype=np.float32)
        env._invalidate_voronoi_cache()
    assert env._voradj_coverage_geometry(env._voronoi_map(), strict=True)["converged_now"] is True

    before_success = False
    ce_success_seen = False
    for _ in range(5):
        result = env.step([np.zeros(2, dtype=np.float32) for _ in env.pursuers], [])
        metadata, sources = _derive_transition_metadata(
            env,
            scene="pure_ce",
            before_active_target=False,
            before_coverage_success=before_success,
            infos=result.infos,
        )
        if "ce_success" in metadata["event_ids"]:
            ce_success_seen = True
            assert sources["event:ce_success"] == "env.coverage_success_latch"
        before_success = bool(
            env.post_capture_coverage_success
            or env.coverage_geometric_success
            or env.coverage_settled_success
            or float(env.last_reward_terms.get("coverage_success", 0.0)) > 0.0
        )
    assert ce_success_seen is True
