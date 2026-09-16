from __future__ import annotations

import copy

import numpy as np
import pytest

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.td3_stage1_contract import (
    ACTION_MODE,
    A_MAX,
    W_MAX,
    ACTION_ONLY_DELTA,
    assert_action_only_delta,
    aw9_grid,
    check_td3_env,
    discrete_task_config,
    make_td3_env,
    td3_task_config,
)
from cocap_voradj.training.trainer import set_global_config


@pytest.mark.parametrize("task", ["coverage", "capture"])
def test_td3_contract_changes_only_action_api(task: str) -> None:
    discrete = discrete_task_config(task)
    continuous = td3_task_config(task)
    assert set(assert_action_only_delta(discrete, continuous)) == set(ACTION_ONLY_DELTA)
    assert continuous["action_mode"] == ACTION_MODE
    assert continuous["env"]["action_mode"] == ACTION_MODE
    assert continuous["pursuer"]["action_mode"] == ACTION_MODE
    assert continuous["action"] == {"mode": ACTION_MODE, "a_max": A_MAX, "w_max": W_MAX}
    assert continuous["yaw"] == {"init": "legacy_random"}


def test_contract_drift_is_rejected() -> None:
    discrete = discrete_task_config("capture")
    continuous = td3_task_config("capture")
    continuous["reward"]["capture_reward"] = 123.0
    with pytest.raises(AssertionError, match="contract drift"):
        assert_action_only_delta(discrete, continuous)


@pytest.mark.parametrize("task", ["coverage", "capture"])
def test_live_td3_env_contract(task: str) -> None:
    env, _ = make_td3_env(task, 2026091601)
    facts = check_td3_env(env, task)
    assert facts["action_mode"] == ACTION_MODE
    assert facts["global_enemy_flag"] is False
    np.testing.assert_allclose(aw9_grid(), env.pursuers[0].action_list, atol=1e-12)


def _robot_state(robot) -> np.ndarray:
    return np.asarray(
        [robot.x, robot.y, robot.theta, robot.speed, *robot.velocity], dtype=np.float64
    )


@pytest.mark.parametrize("task", ["coverage", "capture"])
@pytest.mark.parametrize("action_index", range(9))
def test_current_normsense_aw9_single_step_dynamics_parity(
    task: str, action_index: int
) -> None:
    seed = 2026091617
    discrete_cfg = discrete_task_config(task)
    continuous_cfg = td3_task_config(task)
    set_global_config(discrete_cfg)
    discrete_env = VorAdjEnv(copy.deepcopy(discrete_cfg), seed=seed)
    discrete_env.reset()
    set_global_config(continuous_cfg)
    continuous_env = VorAdjEnv(copy.deepcopy(continuous_cfg), seed=seed)
    continuous_env.reset()
    physical = aw9_grid()[action_index]
    for old_robot, new_robot in zip(discrete_env.pursuers, continuous_env.pursuers):
        np.testing.assert_allclose(_robot_state(old_robot), _robot_state(new_robot), atol=1e-12)
        discrete_env._move_robot(old_robot, action_index)
        continuous_env._move_robot(new_robot, physical)
        np.testing.assert_allclose(_robot_state(old_robot), _robot_state(new_robot), atol=1e-9)
