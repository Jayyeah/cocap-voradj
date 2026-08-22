from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from maadpg_reproduction.env import MAADPGPursuitEnv
from maadpg_reproduction.gate import AdaptiveDifferenceGate
from maadpg_reproduction.target_policy import StationaryTargetPolicy


class LinearRewardEnv:
    def __init__(self):
        self.state = 0
        self.external_action_log = []

    def snapshot(self):
        return self.state

    def restore(self, snapshot):
        self.state = snapshot

    def step(self, actions):
        action = np.asarray(actions, dtype=np.float64)
        self.external_action_log.append(action.copy())
        self.state += 1
        return (
            np.zeros((3, 26)),
            action[:, 0].copy(),
            False,
            False,
            {"executed_action": action.copy()},
        )


def _real_options():
    return {
        "pursuer_positions": np.array([[0.4, 0.4], [0.4, 1.6], [1.6, 0.4]]),
        "target_position": np.array([1.4, 1.4]),
        "obstacles": [],
    }


def _assert_environment_snapshots_equal(left, right):
    np.testing.assert_array_equal(
        left.state.pursuer_positions, right.state.pursuer_positions
    )
    np.testing.assert_array_equal(
        left.state.pursuer_velocities, right.state.pursuer_velocities
    )
    np.testing.assert_array_equal(left.state.target_position, right.state.target_position)
    np.testing.assert_array_equal(left.state.target_velocity, right.state.target_velocity)
    assert left.state.step_count == right.state.step_count
    assert left.scenario_rng_state == right.scenario_rng_state
    assert left.target_policy_state["rng_state"] == right.target_policy_state["rng_state"]
    np.testing.assert_array_equal(
        left.target_policy_state["ou_state"], right.target_policy_state["ou_state"]
    )
    assert left.episode_done == right.episode_done


def test_gate_uses_eq21_sign_ratio_threshold_and_same_teammate_actions():
    env = LinearRewardEnv()
    gate = AdaptiveDifferenceGate()
    actor = np.array([[0.2, 0.0], [0.2, 0.0], [0.2, 0.0]])
    pfm = np.array([[0.4, 0.0], [0.21, 0.0], [-0.2, 0.0]])
    decision = gate.evaluate(env, actor, pfm)
    np.testing.assert_allclose(
        decision.delta_ratios,
        [
            0.2 / (0.4 + 1e-6),
            0.01 / (0.21 + 1e-6),
            -0.4 / (0.2 + 1e-6),
        ],
    )
    assert decision.adopted_pfm.tolist() == [True, False, False]
    np.testing.assert_array_equal(decision.adopted_actions[0], pfm[0])
    np.testing.assert_array_equal(decision.adopted_actions[1:], actor[1:])
    assert env.state == 0
    assert len(env.external_action_log) == 4
    np.testing.assert_array_equal(env.external_action_log[0], actor)
    for index, candidate in enumerate(env.external_action_log[1:]):
        for other in range(3):
            expected = pfm[other] if other == index else actor[other]
            np.testing.assert_array_equal(candidate[other], expected)


def test_zero_pfm_reward_uses_lambda_and_no_division_by_zero():
    env = LinearRewardEnv()
    gate = AdaptiveDifferenceGate()
    actor = np.array([[-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    pfm = np.zeros((3, 2))
    decision = gate.evaluate(env, actor, pfm)
    assert decision.delta_ratios[0] == pytest.approx(1.0e6)
    assert decision.adopted_pfm[0]
    assert np.all(np.isfinite(decision.delta_ratios))


def test_candidate_order_is_exactly_invariant():
    actor = np.array([[0.2, 0.1], [-0.3, 0.1], [0.4, -0.2]])
    pfm = np.array([[0.5, 0.1], [-0.2, 0.1], [0.1, -0.2]])
    gate = AdaptiveDifferenceGate()
    forward = gate.evaluate(LinearRewardEnv(), actor, pfm, candidate_order=(0, 1, 2))
    reverse = gate.evaluate(LinearRewardEnv(), actor, pfm, candidate_order=(2, 1, 0))
    np.testing.assert_array_equal(forward.adopted_actions, reverse.adopted_actions)
    np.testing.assert_array_equal(forward.adopted_pfm, reverse.adopted_pfm)
    np.testing.assert_array_equal(forward.actor_rewards, reverse.actor_rewards)
    np.testing.assert_array_equal(forward.pfm_rewards, reverse.pfm_rewards)
    np.testing.assert_array_equal(forward.delta_ratios, reverse.delta_ratios)
    with pytest.raises(ValueError, match="permutation"):
        gate.evaluate(LinearRewardEnv(), actor, pfm, candidate_order=(0, 0, 2))


def test_real_counterfactuals_restore_environment_target_and_rng_exactly():
    env = MAADPGPursuitEnv()
    env.reset(seed=81, options=_real_options())
    before = env.snapshot()
    actor = np.array([[0.1, 0.2], [-0.2, 0.1], [0.3, -0.1]])
    pfm = np.array([[0.9, 0.0], [0.0, 0.9], [-0.9, 0.0]])
    decision = AdaptiveDifferenceGate().evaluate(env, actor, pfm)
    after = env.snapshot()
    _assert_environment_snapshots_equal(before, after)

    reference = MAADPGPursuitEnv()
    reference.reset(seed=81, options=_real_options())
    left = env.step(decision.adopted_actions)
    right = reference.step(decision.adopted_actions)
    np.testing.assert_array_equal(left[0], right[0])
    np.testing.assert_array_equal(left[1], right[1])
    np.testing.assert_array_equal(left[4]["target_action"], right[4]["target_action"])


def test_gated_real_step_executes_adopted_action_once_after_restore():
    env = MAADPGPursuitEnv(target_policy=StationaryTargetPolicy())
    env.reset(seed=82, options=_real_options())
    actor = np.zeros((3, 2))
    pfm = np.array([[0.5, 0.0], [0.0, 0.5], [-0.5, 0.0]])
    result = AdaptiveDifferenceGate().step_environment(env, actor, pfm)
    assert env.state.step_count == 1
    np.testing.assert_array_equal(
        result.info["executed_action"], result.decision.adopted_actions
    )
    assert result.reward.shape == (3,)
