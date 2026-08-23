from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from open_encirclement_ctde.environment import CorrectedRoundupEnv
from open_encirclement_ctde.maddpg import MADDPG, MADDPGConfig
from open_encirclement_ctde.mappo import MAPPOConfig, MAPPOLearner


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_maddpg_actor_local_and_critic_joint_contract() -> None:
    config = MADDPGConfig(replay_capacity=64, batch_size=8, warmup_steps=8)
    learner = MADDPG(config, torch.device("cpu"), seed=1)
    assert learner.actors[0].fc1.in_features == 26
    assert learner.critics[0].fc1.in_features == 3 * 26 + 3 * 2
    joint_obs = torch.zeros((1, config.joint_obs_dim))
    joint_action = torch.zeros((1, config.joint_action_dim))
    base = learner.critics[0](joint_obs, joint_action)
    changed_obs = joint_obs.clone()
    changed_obs[0, 30] = 1.0
    changed_action = joint_action.clone()
    changed_action[0, 4] = 0.04
    assert not torch.equal(base, learner.critics[0](changed_obs, joint_action))
    assert not torch.equal(base, learner.critics[0](joint_obs, changed_action))


def test_maddpg_replay_and_full_state_roundtrip() -> None:
    config = MADDPGConfig(replay_capacity=64, batch_size=8, warmup_steps=8)
    learner = MADDPG(config, torch.device("cpu"), seed=2)
    rng = np.random.default_rng(10)
    for index in range(8):
        obs = rng.normal(size=(3, 26)).astype(np.float32)
        action = rng.uniform(-0.04, 0.04, size=(3, 2)).astype(np.float32)
        reward = rng.normal(size=3).astype(np.float32)
        learner.replay.add(obs, action, reward, obs + 0.01, index == 7, False)
    metrics = learner.train_step()
    assert all(np.isfinite(value) for value in metrics.values())
    state = learner.full_state_dict()
    restored = MADDPG(config, torch.device("cpu"), seed=999)
    restored.load_state_dict(state, full=True)
    observation = rng.normal(size=(3, 26)).astype(np.float32)
    assert np.array_equal(
        learner.actions(observation, deterministic=True, env_steps=100),
        restored.actions(observation, deterministic=True, env_steps=100),
    )
    assert len(restored.replay) == 8
    assert restored.update_count == learner.update_count
    assert np.array_equal(restored.replay.actions[:8], learner.replay.actions[:8])


def test_mappo_uses_local_actor_joint_value_and_bounded_actions() -> None:
    config = MAPPOConfig(n_rollout_threads=2, rollout_length=2, ppo_epoch=1)
    learner = MAPPOLearner(config, torch.device("cpu"), REPO_ROOT)
    assert learner.policy.actor.base.mlp.fc1[0].in_features == 26
    assert learner.policy.critic.base.mlp.fc1[0].in_features == 78
    observations = np.stack([CorrectedRoundupEnv(seed=i, num_obstacles=0).reset()[0] for i in range(2)])
    learner.initialize(observations)
    collected = learner.collect(0)
    assert collected["actions"].shape == (2, 3, 2)
    assert np.max(np.abs(collected["actions"])) <= 0.04


def test_mappo_full_state_roundtrip_preserves_policy_optimizer_and_valuenorm() -> None:
    config = MAPPOConfig(n_rollout_threads=1, rollout_length=2, ppo_epoch=1)
    learner = MAPPOLearner(config, torch.device("cpu"), REPO_ROOT)
    obs = CorrectedRoundupEnv(seed=33, num_obstacles=0).reset()[0][None, ...]
    learner.initialize(obs)
    expected = learner.deterministic_actions(obs)
    state = learner.full_state_dict()
    restored = MAPPOLearner(config, torch.device("cpu"), REPO_ROOT)
    restored.load_state_dict(state, full=True)
    actual = restored.deterministic_actions(obs)
    assert np.array_equal(expected, actual)
    assert restored.policy.actor_optimizer.state_dict()["param_groups"] == learner.policy.actor_optimizer.state_dict()["param_groups"]
    assert restored.trainer.value_normalizer.state_dict().keys() == learner.trainer.value_normalizer.state_dict().keys()
