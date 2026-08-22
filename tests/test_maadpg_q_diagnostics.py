import math

from test_maadpg_r1_maddpg import _batch, _small_config

from maadpg_reproduction.maddpg import MADDPGLearner


def test_q_ranking_diagnostics_compare_policy_replay_and_shuffled_actions():
    learner = MADDPGLearner(_small_config(), initialization_seed=123)
    diagnostics = learner.update(_batch())
    for agent in diagnostics.agents:
        assert math.isfinite(agent.policy_q_mean)
        assert math.isfinite(agent.shuffled_action_q_mean)
        assert agent.policy_minus_replay_q == agent.policy_q_mean - agent.q_mean
        assert (
            agent.policy_minus_shuffled_q
            == agent.policy_q_mean - agent.shuffled_action_q_mean
        )
