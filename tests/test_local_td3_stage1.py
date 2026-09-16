from __future__ import annotations

import copy

import numpy as np
import torch

from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LegacyVorAdjFeatureBackbone,
    LegacyVorAdjFeatureBackboneConfig,
)
from cocap_voradj.models.continuous.local_td3 import (
    LocalTD3Actor,
    LocalTD3Critic,
    assert_no_parameter_sharing,
)
from cocap_voradj.training.continuous.local_td3 import (
    LocalTD3Config,
    LocalTD3Trainer,
    td3_bootstrap_mask,
)


def backbone() -> LegacyVorAdjFeatureBackbone:
    return LegacyVorAdjFeatureBackbone(
        LegacyVorAdjFeatureBackboneConfig(
            hidden_dim=32,
            num_heads=4,
            num_layers=1,
            self_feature_dim=9,
            max_pursuers=2,
            max_evaders=1,
            max_obstacles=1,
            pursuing_embed_dim=4,
            dropout=0.1,
        )
    )


def local_obs(batch_size: int, agents: int) -> dict[str, torch.Tensor]:
    tokens = 1 + 2 + 1 + 1
    return {
        "self": torch.randn(batch_size, agents, 9),
        "pursuers": torch.randn(batch_size, agents, 2, 7),
        "evaders": torch.randn(batch_size, agents, 1, 7),
        "obstacles": torch.randn(batch_size, agents, 1, 5),
        "masks": torch.ones(batch_size, agents, tokens),
        "types": torch.tensor([0, 1, 1, 2, 3]).view(1, 1, tokens).expand(batch_size, agents, -1),
    }


def trainer(policy_delay: int = 2) -> LocalTD3Trainer:
    torch.manual_seed(7)
    actor = LocalTD3Actor(backbone(), a_max=0.4, w_max=float(np.pi / 6.0))
    q1 = LocalTD3Critic(backbone())
    q2 = LocalTD3Critic(backbone())
    return LocalTD3Trainer(
        actor,
        q1,
        q2,
        LocalTD3Config(policy_delay=policy_delay, policy_noise=0.2, noise_clip=0.5),
        "cpu",
    )


def batch() -> dict[str, object]:
    obs = local_obs(3, 2)
    next_obs = {key: value.clone() for key, value in obs.items()}
    return {
        "local_obs": obs,
        "next_local_obs": next_obs,
        "actions": torch.zeros(3, 2, 2),
        "rewards": torch.randn(3, 2),
        "active_mask": torch.tensor([[True, True], [True, False], [True, True]]),
        "terminated": torch.tensor([[False, False], [True, False], [False, True]]),
        "truncated": torch.tensor([[False, True], [False, False], [False, False]]),
    }


def test_actor_q_backbones_and_optimizers_are_independent() -> None:
    value = trainer()
    assert_no_parameter_sharing(value.actor, value.critic1, value.critic2)
    ids = [
        {id(parameter) for parameter in module.parameters()}
        for module in (value.actor, value.critic1, value.critic2)
    ]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert value.actor_optimizer is not value.critic1_optimizer
    assert value.critic1_optimizer is not value.critic2_optimizer


def test_targets_are_hard_copies_without_parameter_aliases() -> None:
    value = trainer()
    for online, target in (
        (value.actor, value.target_actor),
        (value.critic1, value.target_critic1),
        (value.critic2, value.target_critic2),
    ):
        assert online.state_dict().keys() == target.state_dict().keys()
        for left, right in zip(online.parameters(), target.parameters()):
            assert torch.equal(left, right)
            assert left.data_ptr() != right.data_ptr()
        assert not any(parameter.requires_grad for parameter in target.parameters())


def test_target_smoothing_is_normalized_then_physically_clipped() -> None:
    value = trainer()
    obs = {key: tensor[:, 0] for key, tensor in local_obs(2, 1).items()}
    with torch.no_grad():
        for parameter in value.target_actor.head.parameters():
            parameter.zero_()
    action, normalized_noise = value.smoothed_target_action(
        obs, noise=torch.tensor([[0.9, -0.9], [-0.1, 0.1]])
    )
    expected_noise = torch.tensor([[0.5, -0.5], [-0.1, 0.1]])
    assert torch.allclose(normalized_noise, expected_noise)
    assert torch.allclose(action, expected_noise * value.target_actor.scale)
    assert bool((action.abs() <= value.target_actor.scale + 1e-7).all())


def test_delayed_actor_update_counts_and_all_metrics_are_finite() -> None:
    value = trainer(policy_delay=2)
    before = copy.deepcopy(value.actor.state_dict())
    first = value.update(batch())
    assert first["actor_updated_this_step"] == 0.0
    assert first["policy_update_count"] == 0.0
    assert all(torch.equal(before[key], value.actor.state_dict()[key]) for key in before)
    second = value.update(batch())
    assert second["actor_updated_this_step"] == 1.0
    assert second["policy_update_count"] == 1.0
    assert second["update_count"] == 2.0
    assert all(np.isfinite(number) for number in second.values())


def test_bootstrap_stops_on_terminal_but_not_truncation() -> None:
    terminated = torch.tensor([[True, False, False]])
    truncated = torch.tensor([[False, True, False]])
    assert torch.equal(
        td3_bootstrap_mask(terminated, truncated),
        torch.tensor([[False, True, True]]),
    )


def test_critic_contract_has_no_global_or_teammate_action_input() -> None:
    critic = LocalTD3Critic(backbone())
    obs = {key: tensor[:, 0] for key, tensor in local_obs(4, 1).items()}
    output = critic(obs, torch.zeros(4, 2))
    assert output.shape == (4,)
