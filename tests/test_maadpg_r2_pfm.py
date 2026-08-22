import numpy as np
import pytest

from maadpg_reproduction.guidance_config import PotentialFieldConfig
from maadpg_reproduction.pfm import potential_field_actions


def test_target_attraction_has_paper_inverse_square_magnitude_and_direction():
    pursuers = np.array([[0.0, 0.0], [0.0, 10.0], [0.0, -10.0]])
    result = potential_field_actions(
        pursuers,
        [2.0, 0.0],
        [],
        PotentialFieldConfig(teammate_gain=0.0),
    )
    np.testing.assert_allclose(result.target_components[0], [0.25, 0.0])
    np.testing.assert_allclose(result.actions[0], [1.0, 0.0])


def test_near_obstacle_repels_by_signed_surface_clearance_and_cutoff():
    pursuers = np.array([[0.0, 0.0], [0.0, 2.0], [0.0, -2.0]])
    obstacle = [(np.array([0.15, 0.0]), 0.10)]
    config = PotentialFieldConfig(target_gain=0.0, teammate_gain=0.0)
    result = potential_field_actions(pursuers, [2.0, 0.0], obstacle, config)
    np.testing.assert_allclose(result.obstacle_components[0], [-400.0, 0.0])
    np.testing.assert_allclose(result.actions[0], [-1.0, 0.0])
    np.testing.assert_array_equal(result.obstacle_components[1:], np.zeros((2, 2)))


def test_symmetric_teammate_repulsion_cancels_exactly_for_middle_agent():
    pursuers = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    config = PotentialFieldConfig(target_gain=0.0, obstacle_gain=0.0)
    result = potential_field_actions(pursuers, [0.0, 5.0], [], config)
    np.testing.assert_allclose(result.teammate_components[1], [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(result.actions[1], [0.0, 0.0], atol=1e-15)
    assert result.actions[0, 0] < 0.0
    assert result.actions[2, 0] > 0.0


def test_all_pfm_outputs_are_finite_and_in_canonical_unit_disk():
    pursuers = np.array([[0.0, 0.0], [0.0, 0.0], [1e-12, 0.0]])
    result = potential_field_actions(
        pursuers,
        [0.0, 0.0],
        [([0.0, 0.0], 0.1)],
    )
    assert np.all(np.isfinite(result.resultant_forces))
    assert np.all(np.isfinite(result.actions))
    assert np.all(np.linalg.norm(result.actions, axis=1) <= 1.0 + 1e-15)


def test_zero_force_uses_epsilon_guard_without_inventing_a_direction():
    positions = np.zeros((3, 2))
    config = PotentialFieldConfig(
        target_gain=0.0, obstacle_gain=0.0, teammate_gain=0.0
    )
    result = potential_field_actions(positions, [0.0, 0.0], [], config)
    np.testing.assert_array_equal(result.actions, np.zeros((3, 2)))


def test_invalid_pfm_config_fails_before_action_generation():
    config = PotentialFieldConfig(target_gain=-1.0)
    with pytest.raises(ValueError, match="gains"):
        potential_field_actions(np.zeros((3, 2)), np.zeros(2), [], config)
