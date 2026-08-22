from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from maadpg_reproduction.config import (
    AgentLimits,
    TargetPolicyConfig,
    default_environment_config,
)
from maadpg_reproduction.dynamics import (
    canonical_normalized_action,
    integrate_usv,
)
from maadpg_reproduction.env import MAADPGPursuitEnv
from maadpg_reproduction.geometry import maximum_angular_gap, point_in_triangle
from maadpg_reproduction.lidar import cast_lidar, ray_circle_distance
from maadpg_reproduction.reward import compute_reward_terms
from maadpg_reproduction.success import evaluate_capture
from maadpg_reproduction.target_policy import StationaryTargetPolicy


def _ring(target: np.ndarray, radius: float, angles_deg=(0.0, 120.0, 240.0)):
    angles = np.deg2rad(np.asarray(angles_deg))
    return target + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _fixed_options() -> dict:
    return {
        "pursuer_positions": np.array([[0.4, 0.4], [0.4, 1.6], [1.6, 0.4]]),
        "target_position": np.array([1.4, 1.4]),
        "obstacles": [],
    }


def test_default_config_is_hashable_and_matches_eq24_width():
    config = default_environment_config()
    assert config.pursuer_count == 3
    assert config.pursuer_observation_dim == 26
    assert config.pursuer_limits == AgentLimits(vmax=0.010, amax=0.004)
    assert config.target_limits == AgentLimits(vmax=0.011, amax=0.005)
    assert len(config.canonical_sha256()) == 64
    assert config.canonical_sha256() == config.canonical_sha256()


def test_action_projection_and_semi_implicit_dynamics_golden_case():
    config = default_environment_config()
    action = canonical_normalized_action([1.0, 1.0])
    np.testing.assert_allclose(action, np.array([1.0, 1.0]) / math.sqrt(2.0))
    transition = integrate_usv(
        [1.0, 1.0],
        [0.0, 0.0],
        [1.0, 1.0],
        config.pursuer_limits,
        config.workspace,
        config.dt,
    )
    expected_velocity = np.array([0.004, 0.004]) / math.sqrt(2.0)
    np.testing.assert_allclose(transition.velocity, expected_velocity)
    np.testing.assert_allclose(transition.position, np.array([1.0, 1.0]) + expected_velocity)
    np.testing.assert_allclose(transition.executed_normalized_action, action)
    assert np.linalg.norm(transition.physical_acceleration) <= 0.004 + 1e-15


def test_speed_projection_and_unclipped_boundary_audit():
    config = default_environment_config()
    transition = integrate_usv(
        [1.999, 1.0],
        [0.010, 0.0],
        [1.0, 0.0],
        config.pursuer_limits,
        config.workspace,
        1.0,
    )
    np.testing.assert_allclose(transition.velocity, [0.010, 0.0])
    np.testing.assert_allclose(transition.proposed_position, [2.009, 1.0])
    np.testing.assert_allclose(transition.position, [2.0, 1.0])
    assert transition.boundary_crossing


def test_lidar_boundary_circle_tangent_and_no_hit_cases():
    config = default_environment_config()
    no_hit = cast_lidar(
        [1.0, 1.0], 0.0, [], config.workspace, config.lidar
    )
    np.testing.assert_allclose(no_hit, np.ones(16))
    near_left = cast_lidar(
        [0.1, 1.0], math.pi, [], config.workspace, config.lidar
    )
    assert near_left[0] == pytest.approx(0.5)
    circle = cast_lidar(
        [1.0, 1.0],
        0.0,
        [(np.array([1.15, 1.0]), 0.05)],
        config.workspace,
        config.lidar,
    )
    assert circle[0] == pytest.approx(0.5)
    tangent = ray_circle_distance([0.0, 0.0], [1.0, 0.0], [0.1, 0.05], 0.05)
    assert tangent == pytest.approx(0.1)


def test_eq18_success_and_clause_failures_are_visible():
    config = default_environment_config()
    target = np.array([1.0, 1.0])
    pursuers = _ring(target, 0.10)
    result = evaluate_capture(
        pursuers, target, [], config.workspace, config.capture
    )
    assert result.success
    assert all(result.clauses().values())
    assert result.maximum_angular_gap == pytest.approx(2.0 * math.pi / 3.0)

    too_far = pursuers.copy()
    too_far[0] = target + np.array([0.151, 0.0])
    radial = evaluate_capture(too_far, target, [], config.workspace, config.capture)
    assert not radial.radial_band_ok
    assert not radial.success

    bad_gap = _ring(target, 0.10, (0.0, 100.0, 200.0))
    angular = evaluate_capture(bad_gap, target, [], config.workspace, config.capture)
    assert angular.target_in_hull
    assert angular.maximum_angular_gap == pytest.approx(math.radians(160.0))
    assert not angular.angular_gap_ok
    assert not angular.success

    obstacle = [(pursuers[0] + np.array([0.025, 0.0]), 0.01)]
    clearance = evaluate_capture(
        pursuers, target, obstacle, config.workspace, config.capture
    )
    assert clearance.minimum_obstacle_clearance == pytest.approx(0.015)
    assert not clearance.obstacle_clearance_ok
    assert not clearance.success


def test_geometry_rejects_degenerate_hull_and_reports_wrapped_gap():
    assert not point_in_triangle([0.5, 0.0], [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    gap = maximum_angular_gap(_ring(np.zeros(2), 1.0, (350.0, 10.0, 180.0)), [0.0, 0.0])
    assert gap == pytest.approx(math.radians(170.0))


def test_reward_terms_are_individually_inspectable_and_sum_exactly():
    config = default_environment_config()
    target = np.array([1.0, 1.0])
    pursuers = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    velocities = np.array([[0.01, 0.0], [0.0, 0.01], [-0.01, 0.0]])
    previous = np.linalg.norm(pursuers - target, axis=1) + 0.01
    terms = compute_reward_terms(
        pursuers,
        velocities,
        target,
        previous,
        [0.1, 0.2, 0.05],
        [False, False, True],
        False,
        config.pursuer_limits,
        config.lidar.max_range,
        config.reward,
    )
    np.testing.assert_allclose(terms.progress, np.ones(3))
    np.testing.assert_allclose(terms.safety, [-0.5, 0.0, -10.0])
    assert terms.stage_branch == "approach"
    reconstructed = (
        terms.progress
        + terms.safety
        + terms.stage
        + terms.separation
        + terms.target_hit
        + terms.terminal
    )
    np.testing.assert_allclose(terms.total, reconstructed)


def test_observation_field_order_is_eq24_formula_order():
    env = MAADPGPursuitEnv(target_policy=StationaryTargetPolicy())
    options = {
        "pursuer_positions": np.array([[1.0, 1.0], [0.5, 1.0], [1.5, 1.0]]),
        "pursuer_velocities": np.array([[0.005, -0.005], [0.0, 0.0], [0.0, 0.0]]),
        "target_position": np.array([1.0, 1.5]),
        "obstacles": [],
    }
    observation, _ = env.reset(seed=5, options=options)
    expected_prefix = np.array(
        [0.5, 0.5, 0.5, -0.5, 0.25, 0.5, 0.75, 0.5]
    )
    np.testing.assert_allclose(observation[0, :8], expected_prefix)
    np.testing.assert_allclose(observation[0, 8:24], np.ones(16))
    assert observation[0, 24] == pytest.approx(0.5 / 4.0)
    assert observation[0, 25] == pytest.approx(math.pi / 2.0)


def test_seeded_reset_and_target_policy_are_reproducible():
    left = MAADPGPursuitEnv()
    right = MAADPGPursuitEnv()
    left_observation, _ = left.reset(seed=20260823)
    right_observation, _ = right.reset(seed=20260823)
    np.testing.assert_array_equal(left_observation, right_observation)
    assert len(left.obstacle_pairs) == len(right.obstacle_pairs)
    for left_obstacle, right_obstacle in zip(
        left.obstacle_pairs, right.obstacle_pairs, strict=True
    ):
        np.testing.assert_array_equal(left_obstacle[0], right_obstacle[0])
        assert left_obstacle[1] == right_obstacle[1]
    action = np.zeros((3, 2))
    left_step = left.step(action)
    right_step = right.step(action)
    np.testing.assert_array_equal(left_step[0], right_step[0])
    np.testing.assert_array_equal(left_step[1], right_step[1])
    np.testing.assert_array_equal(
        left_step[4]["target_action"], right_step[4]["target_action"]
    )


def test_snapshot_restore_replays_step_without_state_or_rng_drift():
    env = MAADPGPursuitEnv()
    env.reset(seed=71, options=_fixed_options())
    snapshot = env.snapshot()
    actions = np.array([[0.3, -0.4], [1.0, 1.0], [-0.2, 0.5]])
    first = env.step(actions)
    first_state = env.snapshot()
    env.restore(snapshot)
    second = env.step(actions)
    second_state = env.snapshot()
    for left, right in zip(first[:2], second[:2], strict=True):
        np.testing.assert_array_equal(left, right)
    assert first[2:4] == second[2:4]
    np.testing.assert_array_equal(
        first[4]["executed_action"], second[4]["executed_action"]
    )
    np.testing.assert_array_equal(
        first[4]["target_action"], second[4]["target_action"]
    )
    np.testing.assert_array_equal(
        first_state.state.target_position, second_state.state.target_position
    )
    assert first_state.target_policy_state["rng_state"] == second_state.target_policy_state["rng_state"]


def test_executed_action_is_canonical_and_available_for_replay():
    env = MAADPGPursuitEnv(target_policy=StationaryTargetPolicy())
    env.reset(seed=1, options=_fixed_options())
    _, _, _, _, info = env.step([[2.0, 2.0], [-2.0, 0.0], [0.25, -0.5]])
    expected = np.stack(
        [
            canonical_normalized_action([2.0, 2.0]),
            canonical_normalized_action([-2.0, 0.0]),
            canonical_normalized_action([0.25, -0.5]),
        ]
    )
    np.testing.assert_allclose(info["executed_action"], expected)
    assert np.all(np.linalg.norm(info["executed_action"], axis=1) <= 1.0 + 1e-15)


def test_termination_and_truncation_are_disjoint():
    base = default_environment_config()
    one_step = replace(base, horizon_steps=1)
    truncated_env = MAADPGPursuitEnv(
        config=one_step, target_policy=StationaryTargetPolicy()
    )
    truncated_env.reset(seed=1, options=_fixed_options())
    _, _, terminated, truncated, _ = truncated_env.step(np.zeros((3, 2)))
    assert not terminated and truncated

    boundary_env = MAADPGPursuitEnv(target_policy=StationaryTargetPolicy())
    options = _fixed_options()
    options["pursuer_positions"] = np.array(
        [[1.999, 1.0], [0.4, 1.6], [0.4, 0.4]]
    )
    options["pursuer_velocities"] = np.array(
        [[0.010, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )
    boundary_env.reset(seed=2, options=options)
    _, _, terminated, truncated, info = boundary_env.step(np.zeros((3, 2)))
    assert terminated and not truncated
    assert info["collision"]["boundary"][0]


def test_strict_capture_terminates_with_terminal_bonus():
    env = MAADPGPursuitEnv(target_policy=StationaryTargetPolicy())
    target = np.array([1.0, 1.0])
    env.reset(
        seed=4,
        options={
            "pursuer_positions": _ring(target, 0.10),
            "target_position": target,
            "obstacles": [],
        },
    )
    _, rewards, terminated, truncated, info = env.step(np.zeros((3, 2)))
    assert terminated and not truncated
    assert info["capture"].success
    np.testing.assert_allclose(
        info["reward_terms"]["terminal"], np.full(3, 100.0)
    )
    assert np.all(rewards > 90.0)


def test_stationary_target_config_can_be_constructed_without_legacy_imports():
    config = default_environment_config()
    stationary_config = replace(
        config,
        target_policy=TargetPolicyConfig(policy_id="stationary_target_v1"),
    )
    stationary_config.validate()
    env = MAADPGPursuitEnv(config=stationary_config)
    env.reset(seed=3, options=_fixed_options())
    before = env.state.target_position.copy()
    env.step(np.zeros((3, 2)))
    np.testing.assert_array_equal(env.state.target_position, before)
