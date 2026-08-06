"""Contract tests for the positive-feedback ladder configs and runner wiring."""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest

from cocap_voradj.dynamics.continuous_action import AccelerationAngularVelocityActionAdapter
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.continuous.formal_config import resolve_ladder_config, scene_config
from cocap_voradj.training.trainer import load_config, set_global_config


ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT / "configs/experiments/positive_feedback_ladder_20260807/stage2_simple_aw.yaml"
STAGE3A = ROOT / "configs/experiments/positive_feedback_ladder_20260807/stage3a_pure_ce_aw.yaml"


def test_stage2_ladder_config_contract() -> None:
    config = resolve_ladder_config(STAGE2)
    assert config["dynamics"]["profile"] == "continuous_aw_v1"
    assert config["action"]["mode"] == "acceleration_angular_velocity_body"
    assert abs(float(config["action"]["a_max"]) - 0.4) < 1e-9
    assert abs(float(config["action"]["w_max"]) - float(np.pi / 6.0)) < 1e-9
    assert config["env"]["num_pursuers"] == 1
    assert config["env"]["num_evaders"] == 1
    assert config["env"]["num_obstacles"] == 0
    assert config["perception"]["global_evader_visibility"] is True
    assert config["yaw"]["init"] == "legacy_random"
    assert config["training"]["batch_size"] == 128
    assert config["training"]["grad_clip_norm"] == 0.5
    assert config["evaluation"]["diagnostic_rollout_cap"] == 400


def test_stage3a_ladder_config_contract() -> None:
    config = resolve_ladder_config(STAGE3A)
    assert config["action"]["mode"] == "acceleration_angular_velocity_body"
    assert config["env"]["num_pursuers"] == 4
    assert config["env"]["num_evaders"] == 0
    assert config["env"]["num_obstacles"] == 0
    scene = scene_config(config, "pure_ce")
    assert scene["env"]["episode_max_length"] == 1500
    assert scene["env"]["pursuer_spawn_mode"] == "inner_random_cluster"


def test_stage2_env_adapter_and_stationary_target() -> None:
    config = resolve_ladder_config(STAGE2)
    scene = scene_config(config, "capture")
    set_global_config(scene)
    env = VorAdjEnv(scene, seed=2026080701)
    env.reset()
    assert isinstance(env.action_adapter, AccelerationAngularVelocityActionAdapter)
    assert abs(float(env.action_adapter.a_max) - 0.4) < 1e-9
    assert abs(float(env.action_adapter.w_max) - float(np.pi / 6.0)) < 1e-9
    target_before = np.asarray([env.evaders[0].x, env.evaders[0].y], dtype=float)
    outcome = env.step([[0.0, 0.0]], [None])
    target_after = np.asarray([env.evaders[0].x, env.evaders[0].y], dtype=float)
    assert np.allclose(target_before, target_after, atol=1e-9)
    assert len(outcome.rewards) == 1


def test_stage1_aw_bridge_single_step_exact_parity() -> None:
    old_config = load_config(
        str(ROOT / "configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml")
    )
    new_config = deepcopy(old_config)
    new_config["action_mode"] = "acceleration_angular_velocity_body"
    new_config["a_max"] = 0.4
    new_config["w_max"] = float(np.pi / 6.0)
    new_config["v_max"] = 3.0
    new_config["decision_dt"] = 0.5
    new_config["yaw"] = {"init": "legacy_random"}
    new_config["pursuer"]["action_mode"] = "acceleration_angular_velocity_body"
    new_config["pursuer"]["a_max"] = 0.4
    new_config["pursuer"]["w_max"] = float(np.pi / 6.0)
    set_global_config(old_config)
    old_env = VorAdjEnv(old_config, seed=2026081201)
    old_env.reset()
    set_global_config(new_config)
    new_env = VorAdjEnv(new_config, seed=2026081201)
    new_env.reset()
    for idx in range(len(old_env.pursuers)):
        old_env._move_robot(old_env.pursuers[idx], 8)
        new_env._move_robot(new_env.pursuers[idx], (0.4, float(np.pi / 6.0)))
        old_state = (
            old_env.pursuers[idx].x,
            old_env.pursuers[idx].y,
            old_env.pursuers[idx].theta,
            old_env.pursuers[idx].speed,
            *old_env.pursuers[idx].velocity,
        )
        new_state = (
            new_env.pursuers[idx].x,
            new_env.pursuers[idx].y,
            new_env.pursuers[idx].theta,
            new_env.pursuers[idx].speed,
            *new_env.pursuers[idx].velocity,
        )
        assert np.allclose(old_state, new_state, atol=1e-9)
