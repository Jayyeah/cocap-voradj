from __future__ import annotations

import math

import numpy as np
import pytest

from cocap_voradj.dynamics.continuous_action import vxy9_body_grid
from cocap_voradj.dynamics.robot import Robot
from cocap_voradj.envs.base import CoCapEnv


def _config(action_mode: str) -> dict:
    return {
        "action_mode": action_mode,
        "v_max": 3.0,
        "decision_dt": 0.5,
        "action": {"servo_acceleration_limit": 0.4},
        "env": {
            "action_mode": action_mode,
            "width": 55.0,
            "height": 55.0,
            "num_pursuers": 1,
            "num_evaders": 0,
            "num_obstacles": 0,
        },
        "perception": {},
        "reward": {},
    }


def _robot() -> Robot:
    robot = Robot(0)
    robot.robot_type = "pursuer"
    robot.max_speed = 3.0
    robot.coefficient_water_resistance = 0.4 / 3.0
    robot.start = np.asarray([27.5, 27.5], dtype=float)
    robot.init_theta = 0.0
    robot.init_speed = 0.0
    robot.reset_state(np.zeros(2, dtype=float))
    return robot


def test_vxy9_grid_is_the_proven_cartesian_order_and_norm() -> None:
    grid = vxy9_body_grid(3.0)
    component = 3.0 / math.sqrt(2.0)
    expected = np.asarray(
        [(x, y) for x in (-component, 0.0, component) for y in (-component, 0.0, component)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(grid, expected, rtol=0.0, atol=0.0)
    assert np.max(np.linalg.norm(grid, axis=1)) <= 3.0 + 1e-6


def test_discrete_vxy9_maps_iqn_index_while_vector_mode_stays_continuous() -> None:
    discrete = CoCapEnv(_config("vxy9"), task="voradj", seed=1)
    assert discrete.action_size == 9
    robot = _robot()
    discrete._move_robot(robot, 8)
    np.testing.assert_allclose(robot.action_history[-1], vxy9_body_grid(3.0)[8], atol=1e-7)
    with pytest.raises(ValueError, match="integer scalar"):
        discrete._move_robot(_robot(), np.asarray([0.0, 0.0], dtype=np.float32))

    continuous = CoCapEnv(_config("desired_velocity_2d_body"), task="voradj", seed=1)
    assert continuous.action_size == 2
    continuous._move_robot(_robot(), np.asarray([0.0, 0.0], dtype=np.float32))


@pytest.mark.parametrize("action", [-1, 9, 1.0, np.asarray([1], dtype=np.int64)])
def test_discrete_vxy9_rejects_non_contract_actions(action) -> None:
    env = CoCapEnv(_config("discrete_desired_velocity_2d_body"), task="voradj", seed=1)
    with pytest.raises(ValueError):
        env._move_robot(_robot(), action)
