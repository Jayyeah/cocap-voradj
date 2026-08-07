from __future__ import annotations

import numpy as np
import pytest

from cocap_voradj.dynamics.continuous_action import (
    ActionContractError,
    AccelerationActionAdapter,
    body_to_world,
    world_to_body,
)
from cocap_voradj.dynamics.robot import Robot


def test_acceleration_action_contract_only_limits_a_max() -> None:
    adapter = AccelerationActionAdapter(a_max=0.8, decision_dt=0.5)
    action, diagnostics = adapter.validate_with_diagnostics([0.4, 0.0])
    assert np.linalg.norm(action) <= 0.8 + 1e-9
    assert not diagnostics.action_rejected
    adapter.validate(action)


def test_body_world_rotation_round_trip_and_zero() -> None:
    body = np.asarray([1.2, -0.7])
    world = body_to_world(body, np.pi / 2.0)
    assert np.allclose(world, [0.7, 1.2], atol=1e-8)
    assert np.allclose(world_to_body(world, np.pi / 2.0), body, atol=1e-8)
    assert np.allclose(body_to_world([0.0, 0.0], 1.7), [0.0, 0.0])


def test_invalid_shape_nan_and_unfeasible_command_raise() -> None:
    adapter = AccelerationActionAdapter(a_max=0.8, decision_dt=0.5)
    with pytest.raises(ActionContractError):
        adapter.validate_with_diagnostics([0.0, 0.0, 0.0])
    with pytest.raises(ActionContractError):
        adapter.validate_with_diagnostics([np.nan, 0.0])
    with pytest.raises(ActionContractError):
        adapter.validate([1.0, 0.0])


def test_robot_acceleration_update_uses_all_ten_substeps_and_trapezoids() -> None:
    robot = Robot(0)
    robot.x = 0.0
    robot.y = 0.0
    robot.theta = 0.0
    robot.max_speed = 3.0
    robot.velocity = np.zeros(2, dtype=float)
    callbacks = []
    robot.update_state_acceleration_body([0.4, 0.0], substep_callback=lambda: callbacks.append(robot.x))
    assert len(callbacks) == 10
    assert np.isclose(robot.x, 0.05, atol=1e-8)
    assert np.isclose(robot.y, 0.0, atol=1e-8)


def test_environment_speed_cap_is_separate_from_acceleration_action() -> None:
    robot = Robot(1)
    robot.x = 0.0
    robot.y = 0.0
    robot.theta = 0.0
    robot.max_speed = 3.0
    robot.velocity = np.asarray([2.8, 0.0], dtype=float)
    speed_limited = robot.update_state_acceleration_body([0.8, 0.0])
    assert speed_limited is True
    assert np.isclose(robot.speed, 3.0, atol=1e-8)
    assert np.linalg.norm([0.8, 0.0]) <= 0.8 + 1e-8
