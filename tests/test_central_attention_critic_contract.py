from __future__ import annotations

import pytest
import torch

from cocap_voradj.models.continuous.central_attention_critic import (
    CentralAttentionCritic,
    CentralCriticConfig,
    CentralTwinCritics,
)


def make_global_obs(
    batch: int = 2,
    agents: int = 12,
    evaders: int = 8,
    obstacles: int = 5,
    active_count: int = 8,
) -> dict[str, torch.Tensor]:
    active = torch.zeros(batch, agents, dtype=torch.bool)
    active[:, :active_count] = True
    pursuer_mask = active[:, :, None] & active[:, None, :]
    return {
        "self": torch.randn(batch, agents, 9),
        "pursuers": torch.randn(batch, agents, agents, 7),
        "evaders": torch.randn(batch, evaders, 7),
        "obstacles": torch.randn(batch, obstacles, 5),
        "pursuer_mask": pursuer_mask,
        "evader_mask": torch.ones(batch, evaders, dtype=torch.bool),
        "obstacle_mask": torch.ones(batch, obstacles, dtype=torch.bool),
        "active_mask": active,
    }


def config(agents: int = 12) -> CentralCriticConfig:
    return CentralCriticConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        max_agents=agents,
        max_evaders=8,
        max_obstacles=5,
        dropout=0.0,
    )


def test_central_twin_shapes_masks_and_independent_parameters() -> None:
    torch.manual_seed(2026080405)
    model = CentralTwinCritics(config()).eval()
    obs = make_global_obs()
    commands = torch.randn(2, 12, 2)
    with torch.no_grad():
        q1, q2 = model(obs, commands)
    assert q1.shape == (2, 12)
    assert q2.shape == (2, 12)
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all()
    assert torch.all(q1[:, 8:] == 0) and torch.all(q2[:, 8:] == 0)
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(model.critic1.parameters(), model.critic2.parameters()))


def test_central_critic_entity_and_focal_permutation_equivariance() -> None:
    torch.manual_seed(2026080406)
    model = CentralAttentionCritic(config()).eval()
    obs = make_global_obs(batch=1, active_count=4)
    commands = torch.randn(1, 12, 2)
    perm = torch.tensor([2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11])
    permuted = {key: value.clone() for key, value in obs.items()}
    permuted["self"] = obs["self"][:, perm]
    permuted["pursuers"] = obs["pursuers"][:, perm][:, :, perm]
    permuted["pursuer_mask"] = obs["pursuer_mask"][:, perm][:, :, perm]
    permuted["active_mask"] = obs["active_mask"][:, perm]
    permuted_commands = commands[:, perm]
    with torch.no_grad():
        original = model(obs, commands)
        reordered = model(permuted, permuted_commands)
    inverse = torch.argsort(perm)
    assert torch.allclose(original, reordered[:, inverse], atol=1e-5, rtol=1e-5)


def test_central_critic_supports_4_8_12_agent_padded_shapes() -> None:
    for count in (4, 8, 12):
        model = CentralAttentionCritic(config(count)).eval()
        obs = make_global_obs(agents=count, active_count=count)
        commands = torch.randn(2, count, 2)
        with torch.no_grad():
            q = model(obs, commands)
        assert q.shape == (2, count)
        assert torch.isfinite(q).all()


def test_central_critic_rejects_all_inactive_batch() -> None:
    model = CentralAttentionCritic(config())
    obs = make_global_obs(active_count=0)
    with pytest.raises(ValueError, match="at least one active"):
        model(obs, torch.zeros(2, 12, 2))


def test_central_critic_rejects_joint_command_shape() -> None:
    model = CentralAttentionCritic(config())
    obs = make_global_obs()
    with pytest.raises(ValueError, match="joint_actions"):
        model(obs, torch.zeros(2, 12, 3))


def test_central_focal_q_responds_to_teammate_command_but_not_padding() -> None:
    torch.manual_seed(2026080408)
    model = CentralAttentionCritic(config()).eval()
    obs = make_global_obs(batch=1, active_count=4)
    commands = torch.zeros(1, 12, 2)
    with torch.no_grad():
        baseline = model(obs, commands)
        teammate_changed = commands.clone()
        teammate_changed[:, 1] = torch.tensor([2.0, -1.0])
        changed = model(obs, teammate_changed)
        padded_changed = commands.clone()
        padded_changed[:, 8] = torch.tensor([2.0, -1.0])
        padded = model(obs, padded_changed)
    assert not torch.allclose(baseline[:, 0], changed[:, 0])
    assert torch.allclose(baseline[:, :4], padded[:, :4], atol=1e-6, rtol=1e-6)
