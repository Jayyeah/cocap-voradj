from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from maadpg_reproduction.maddpg import MADDPGLearner
from maadpg_reproduction.networks import Actor, JointCritic
from maadpg_reproduction.replay import JointBatch, JointReplayBuffer
from maadpg_reproduction.training_config import default_maddpg_config


def _small_config():
    return replace(
        default_maddpg_config(),
        hidden_dims=(32, 32),
        replay_capacity=128,
        batch_size=8,
        replay_warmup=8,
    )


def _batch(batch_size: int = 8) -> JointBatch:
    generator = torch.Generator().manual_seed(51)
    actions = torch.randn(batch_size, 3, 2, generator=generator)
    actions = actions / torch.clamp(
        torch.linalg.vector_norm(actions, dim=-1, keepdim=True), min=1.0
    )
    return JointBatch(
        observations=torch.randn(batch_size, 3, 26, generator=generator),
        actions=actions,
        rewards=torch.randn(batch_size, 3, generator=generator),
        next_observations=torch.randn(batch_size, 3, 26, generator=generator),
        terminated=torch.tensor([False, True] * (batch_size // 2)),
        truncated=torch.tensor([True, False] * (batch_size // 2)),
    )


def _parameters(module):
    return [value.detach().clone() for value in module.parameters()]


def test_paper_network_shapes_and_actor_unit_disk_output():
    actor = Actor()
    critic = JointCritic()
    assert actor.net[0].in_features == 26
    assert actor.net[0].out_features == 128
    assert actor.net[2].in_features == actor.net[2].out_features == 128
    assert actor.net[4].out_features == 2
    assert critic.net[0].in_features == 84
    assert critic.net[0].out_features == 128
    actions = actor(torch.randn(64, 26) * 100.0)
    assert actions.shape == (64, 2)
    assert torch.all(torch.linalg.vector_norm(actions, dim=-1) <= 1.0 + 1e-6)


def test_three_actors_and_critics_do_not_share_parameters():
    learner = MADDPGLearner(_small_config(), initialization_seed=7)
    actor_ids = [{id(parameter) for parameter in actor.parameters()} for actor in learner.actors]
    critic_ids = [{id(parameter) for parameter in critic.parameters()} for critic in learner.critics]
    assert all(actor_ids[left].isdisjoint(actor_ids[right]) for left in range(3) for right in range(left + 1, 3))
    assert all(critic_ids[left].isdisjoint(critic_ids[right]) for left in range(3) for right in range(left + 1, 3))


def test_actor_execution_is_strictly_local_to_its_fixed_observation_slot():
    learner = MADDPGLearner(_small_config(), initialization_seed=8)
    observations = np.zeros((3, 26), dtype=np.float32)
    baseline = learner.select_actions(observations)
    changed = observations.copy()
    changed[1] = 100.0
    result = learner.select_actions(changed)
    np.testing.assert_array_equal(result[0], baseline[0])
    np.testing.assert_array_equal(result[2], baseline[2])
    assert not np.array_equal(result[1], baseline[1])


def test_joint_critic_depends_differentiably_on_other_agent_actions():
    critic = JointCritic(hidden_dims=(32, 32))
    observations = torch.randn(5, 3, 26, requires_grad=True)
    actions = torch.randn(5, 3, 2, requires_grad=True)
    critic(observations, actions).sum().backward()
    assert actions.grad is not None
    assert torch.sum(torch.abs(actions.grad[:, 1:, :])).item() > 0.0
    assert observations.grad is not None
    assert torch.sum(torch.abs(observations.grad[:, 1:, :])).item() > 0.0


def test_true_termination_zeros_bootstrap_but_truncation_does_not():
    learner = MADDPGLearner(_small_config(), initialization_seed=9)
    with torch.no_grad():
        for critic in learner.target_critics:
            for parameter in critic.parameters():
                parameter.zero_()
            critic.net[4].bias.fill_(1.0)
    batch = JointBatch(
        observations=torch.zeros(2, 3, 26),
        actions=torch.zeros(2, 3, 2),
        rewards=torch.zeros(2, 3),
        next_observations=torch.zeros(2, 3, 26),
        terminated=torch.tensor([True, False]),
        truncated=torch.tensor([False, True]),
    )
    targets = learner.compute_critic_targets(batch)
    torch.testing.assert_close(targets[0], torch.zeros(3))
    torch.testing.assert_close(targets[1], torch.full((3,), 0.95))
    torch.testing.assert_close(batch.bootstrap_mask, torch.tensor([0.0, 1.0]))


def test_standard_maddpg_update_changes_all_online_networks_and_logs_health():
    learner = MADDPGLearner(_small_config(), initialization_seed=10)
    actor_before = [_parameters(module) for module in learner.actors]
    critic_before = [_parameters(module) for module in learner.critics]
    diagnostics = learner.update(_batch())
    assert diagnostics.update_step == 1
    assert len(diagnostics.agents) == 3
    for index, values in enumerate(diagnostics.agents):
        assert math.isfinite(values.critic_loss)
        assert math.isfinite(values.actor_loss)
        assert math.isfinite(values.td_abs_mean)
        assert math.isfinite(values.critic_gradient_norm)
        assert math.isfinite(values.actor_gradient_norm)
        assert values.action_norm_max <= 1.0 + 1e-6
        assert any(
            not torch.equal(before, after)
            for before, after in zip(actor_before[index], learner.actors[index].parameters(), strict=True)
        )
        assert any(
            not torch.equal(before, after)
            for before, after in zip(critic_before[index], learner.critics[index].parameters(), strict=True)
        )
    assert all(parameter.grad is None for critic in learner.critics for parameter in critic.parameters())


def test_soft_target_update_uses_frozen_tau_without_hard_copy():
    learner = MADDPGLearner(_small_config(), initialization_seed=11)
    online = next(learner.actors[0].parameters())
    target = next(learner.target_actors[0].parameters())
    with torch.no_grad():
        target.zero_()
        online.fill_(1.0)
    learner.soft_update_targets()
    torch.testing.assert_close(target, torch.full_like(target, learner.config.tau))


def test_replay_preserves_fixed_slots_terminal_transition_and_exact_action():
    replay = JointReplayBuffer(3, seed=12)
    recorded_actions = []
    for step in range(4):
        observation = np.full((3, 26), step, dtype=np.float32)
        action = np.full((3, 2), 0.1 * step, dtype=np.float32)
        recorded_actions.append(action)
        replay.add(
            observation,
            action,
            np.full(3, step, dtype=np.float32),
            observation + 1,
            terminated=step == 2,
            truncated=step == 3,
        )
    assert len(replay) == 3 and replay.cursor == 1
    np.testing.assert_array_equal(replay.actions[0], recorded_actions[3])
    np.testing.assert_array_equal(replay.actions[2], recorded_actions[2])
    assert replay.terminated[2]
    terminal_batch = replay.batch_from_indices([2])
    assert terminal_batch.terminated.item()
    assert terminal_batch.bootstrap_mask.item() == 0.0


def test_replay_rejects_noncanonical_actions_and_restores_sampling_rng():
    replay = JointReplayBuffer(8, seed=13)
    observation = np.zeros((3, 26), dtype=np.float32)
    with pytest.raises(ValueError, match="canonical"):
        replay.add(
            observation,
            np.full((3, 2), 1.0, dtype=np.float32),
            np.zeros(3),
            observation,
            terminated=False,
            truncated=False,
        )
    for step in range(8):
        replay.add(
            observation + step,
            np.zeros((3, 2)),
            np.full(3, step),
            observation + step + 1,
            terminated=False,
            truncated=False,
        )
    state = replay.state_dict()
    expected_indices = replay.sample_indices(6)
    restored = JointReplayBuffer(8, seed=999)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(restored.sample_indices(6), expected_indices)


def test_learner_checkpoint_roundtrip_preserves_next_update_exactly():
    config = _small_config()
    batch = _batch()
    left = MADDPGLearner(config, initialization_seed=14)
    left.update(batch)
    state = left.state_dict()
    right = MADDPGLearner(config, initialization_seed=999)
    right.load_state_dict(state)
    observations = np.arange(78, dtype=np.float32).reshape(3, 26) / 78.0
    np.testing.assert_array_equal(left.select_actions(observations), right.select_actions(observations))
    left_diagnostics = left.update(batch)
    right_diagnostics = right.update(batch)
    assert left_diagnostics == right_diagnostics
    for left_module, right_module in zip(left.actors, right.actors, strict=True):
        for left_value, right_value in zip(left_module.parameters(), right_module.parameters(), strict=True):
            torch.testing.assert_close(left_value, right_value, rtol=0.0, atol=0.0)
