from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LocalEntityTokenEncoder,
    LocalEntityTokenEncoderConfig,
)
from cocap_voradj.models.small_step_ac import (
    CategoricalGridActor,
    CentralCounterfactualQNetwork,
)
from cocap_voradj.training.continuous.formal_config import resolve_ladder_config
from cocap_voradj.training.small_step_ac import (
    CounterfactualPPOTrainer,
    MAPPOConfig,
    compute_q_lambda_returns,
    counterfactual_advantage,
)
from tools.run_small_step_ac_migration import configure_environment


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "configs/experiments/mappo9_v2_20260830/seed1.yaml"
CF_ROOT = ROOT / "configs/experiments/ppo_counterfactual_q_20260901"


def _global_obs(steps: int, agents: int = 2) -> dict[str, torch.Tensor]:
    active = torch.ones(steps, agents, dtype=torch.bool)
    return {
        "self": torch.randn(steps, agents, 9),
        "pursuers": torch.randn(steps, agents, agents, 7),
        "evaders": torch.randn(steps, 1, 7),
        "obstacles": torch.randn(steps, 1, 5),
        "pursuer_mask": active[:, :, None] & active[:, None, :],
        "evader_mask": torch.ones(steps, 1, dtype=torch.bool),
        "obstacle_mask": torch.ones(steps, 1, dtype=torch.bool),
        "active_mask": active,
    }


def _local_obs(batch: int) -> dict[str, torch.Tensor]:
    return {
        "self": torch.randn(batch, 9),
        "pursuers": torch.randn(batch, 2, 7),
        "evaders": torch.randn(batch, 1, 7),
        "obstacles": torch.randn(batch, 1, 5),
        "masks": torch.ones(batch, 5, dtype=torch.bool),
        "types": torch.tensor([0, 1, 1, 2, 3], dtype=torch.long).repeat(batch, 1),
    }


def _critic() -> CentralCounterfactualQNetwork:
    return CentralCounterfactualQNetwork(
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        max_agents=2,
        max_evaders=1,
        max_obstacles=1,
    )


def test_cf_configs_are_strict_mappo9_v2_task_actor_and_ppo_children() -> None:
    parent_root = resolve_ladder_config(PARENT)
    parent = configure_environment(parent_root, "mappo9_v2")
    for path in sorted(CF_ROOT.glob("seed*.yaml")):
        root = resolve_ladder_config(path)
        configured = configure_environment(root, "ppo_cf")
        assert root["small_step_ac"]["algorithm"] == "ppo_cf"
        assert root["small_step_ac"]["mappo"] == parent_root["small_step_ac"]["mappo"]
        assert configured["actor"] == parent["actor"]
        assert configured["iqn"] == parent["iqn"]
        assert configured["reward"] == parent["reward"]
        assert configured["env"] == parent["env"]
        assert configured["action"] == parent["action"]
        assert configured["voradj"] == parent["voradj"]
        assert root["seed"] == parent_root["seed"] + int(root["experiment_metadata"]["seed_index"]) - 1


def test_exact_counterfactual_baseline_and_focal_action_indexing() -> None:
    q = torch.tensor([[[0.0, 2.0, 8.0], [4.0, 5.0, 6.0]]])
    probabilities = torch.tensor([[[0.25, 0.25, 0.50], [0.2, 0.3, 0.5]]])
    actions = torch.tensor([[1, 0]])
    advantage, chosen, baseline = counterfactual_advantage(q, probabilities, actions)
    assert torch.allclose(chosen, torch.tensor([[2.0, 4.0]]))
    assert torch.allclose(baseline, torch.tensor([[4.5, 5.3]]))
    assert torch.allclose(advantage, torch.tensor([[-2.5, -1.3]]))


def test_q_lambda_terminal_truncation_active_mask_and_rollout_bootstrap() -> None:
    rewards = torch.tensor([[1.0, 9.0], [2.0, 9.0], [3.0, 9.0]])
    next_baselines = torch.tensor([[10.0, 7.0], [20.0, 7.0], [30.0, 7.0]])
    terminated = torch.tensor([[False, False], [False, False], [True, False]])
    truncated = torch.tensor([[False, False], [True, False], [False, False]])
    episode_end = terminated | truncated
    active = torch.tensor([[True, False], [True, False], [True, False]])
    target = compute_q_lambda_returns(
        rewards,
        next_baselines,
        terminated,
        active,
        gamma=0.9,
        trace_lambda=1.0,
        truncated=truncated,
        episode_end=episode_end,
    )
    assert torch.allclose(target[:, 0], torch.tensor([19.0, 20.0, 3.0]))
    assert torch.all(target[:, 1] == 0)

    boundary = compute_q_lambda_returns(
        torch.tensor([[1.0]]),
        torch.tensor([[10.0]]),
        torch.tensor([[False]]),
        torch.tensor([[True]]),
        gamma=0.9,
        trace_lambda=0.95,
        episode_end=torch.tensor([[False]]),
    )
    assert torch.allclose(boundary, torch.tensor([[10.0]]))


def test_central_q_excludes_focal_action_but_conditions_on_teammates() -> None:
    torch.manual_seed(20260901)
    critic = _critic().eval()
    obs = _global_obs(1)
    actions = torch.zeros(1, 2, dtype=torch.long)
    with torch.no_grad():
        baseline = critic(obs, actions)
        focal_changed = actions.clone()
        focal_changed[:, 0] = 8
        focal = critic(obs, focal_changed)
        teammate_changed = actions.clone()
        teammate_changed[:, 1] = 8
        teammate = critic(obs, teammate_changed)
    assert baseline.shape == (1, 2, 9)
    assert torch.allclose(baseline[:, 0], focal[:, 0], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(baseline[:, 0], teammate[:, 0])


def test_central_q_masks_inactive_padding_and_rejects_bad_indices() -> None:
    torch.manual_seed(20260903)
    critic = _critic().eval()
    obs = _global_obs(1)
    obs["active_mask"][:, 1] = False
    obs["pursuer_mask"] = obs["active_mask"][:, :, None] & obs["active_mask"][:, None, :]
    actions = torch.zeros(1, 2, dtype=torch.long)
    with torch.no_grad():
        baseline = critic(obs, actions)
        padded_changed = actions.clone()
        padded_changed[:, 1] = 8
        changed = critic(obs, padded_changed)
    assert torch.all(baseline[:, 1] == 0)
    assert torch.allclose(baseline[:, 0], changed[:, 0], atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="outside"):
        critic(obs, torch.tensor([[0, 9]]))


def test_counterfactual_trainer_updates_actor_critic_and_roundtrips() -> None:
    torch.manual_seed(20260902)
    encoder = LocalEntityTokenEncoder(
        LocalEntityTokenEncoderConfig(
            hidden_dim=32,
            num_heads=4,
            num_layers=1,
            self_feature_dim=9,
            max_pursuers=2,
            max_evaders=1,
            max_obstacles=1,
            dropout=0.0,
        )
    )
    grid = np.asarray(
        [(a, w) for a in (-0.4, 0.0, 0.4) for w in (-0.5, 0.0, 0.5)],
        dtype=np.float32,
    )
    actor = CategoricalGridActor(encoder, grid, orthogonal_policy_head=True)
    config = MAPPOConfig(
        ppo_epochs=1,
        minibatches=1,
        actor_lr=3e-5,
        critic_lr=1e-4,
        target_kl=0.02,
        value_norm=True,
    )
    trainer = CounterfactualPPOTrainer(actor, _critic(), config, "cpu")
    steps, agents = 4, 2
    local_flat = _local_obs(steps * agents)
    local = {
        key: value.reshape(steps, agents, *value.shape[1:])
        for key, value in local_flat.items()
    }
    next_local = {key: value.clone() for key, value in local.items()}
    with torch.no_grad():
        distribution = actor.distribution(local_flat)
        action = distribution.sample().reshape(steps, agents)
        probabilities = distribution.probs.reshape(steps, agents, 9)
        log_prob = distribution.log_prob(action.reshape(-1)).reshape(steps, agents)
    global_obs = _global_obs(steps, agents)
    batch = {
        "local_obs": local,
        "next_local_obs": next_local,
        "global_obs": global_obs,
        "next_global_obs": copy.deepcopy(global_obs),
        "physical_actions": torch.zeros(steps, agents, 2),
        "action_indices": action,
        "next_action_indices": action.clone(),
        "log_prob": log_prob,
        "behavior_action_probs": probabilities,
        "next_behavior_action_probs": probabilities.clone(),
        "rewards": torch.randn(steps, agents),
        "active_mask": torch.ones(steps, agents, dtype=torch.bool),
        "terminated": torch.tensor([[False, False]] * 3 + [[True, True]]),
        "truncated": torch.zeros(steps, agents, dtype=torch.bool),
        "episode_end": torch.tensor([[False, False]] * 3 + [[True, True]]),
    }
    metrics = trainer.update(batch)
    required = {
        "actor_loss",
        "critic_loss",
        "approx_kl",
        "q_chosen_mean",
        "q_baseline_mean",
        "counterfactual_advantage_mean",
        "counterfactual_advantage_std",
        "q_span_mean",
        "chosen_action_q_rank_mean",
        "chosen_action_q_argmax_rate",
        "critic_grad_norm",
    }
    assert required <= metrics.keys()
    assert all(np.isfinite(metrics[key]) for key in required)
    assert metrics["actor_update_l2"] > 0
    assert metrics["critic_grad_norm"] > 0

    clone = CounterfactualPPOTrainer(
        CategoricalGridActor(copy.deepcopy(encoder), grid, orthogonal_policy_head=True),
        _critic(),
        config,
        "cpu",
    )
    clone.load_state_dict(trainer.state_dict())
    assert clone.update_count == trainer.update_count
    for left, right in zip(trainer.actor.parameters(), clone.actor.parameters()):
        assert torch.equal(left, right)
    for left, right in zip(trainer.critic.parameters(), clone.critic.parameters()):
        assert torch.equal(left, right)
