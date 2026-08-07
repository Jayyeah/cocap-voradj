from __future__ import annotations

import numpy as np
import torch

from cocap_voradj.models.continuous.central_attention_critic import CentralCriticConfig
from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoderConfig
from cocap_voradj.models.continuous.radial_actor import RadialActorConfig
from cocap_voradj.training.continuous.central_sac import CentralSACConfig, CentralSACTrainer
from cocap_voradj.training.continuous.local_sac import LocalSACConfig, LocalSACTrainer


def make_batch(batch_size: int = 2, agents: int = 4) -> dict[str, object]:
    torch.manual_seed(2026080407)
    local = {
        "self": torch.randn(batch_size, agents, 9),
        "pursuers": torch.randn(batch_size, agents, 8, 7),
        "evaders": torch.randn(batch_size, agents, 8, 7),
        "obstacles": torch.randn(batch_size, agents, 5, 5),
        "masks": torch.ones(batch_size, agents, 22),
        "types": torch.zeros(batch_size, agents, 22),
    }
    local["types"][:, :, 1:9] = 1
    local["types"][:, :, 9:17] = 2
    local["types"][:, :, 17:] = 3
    active = torch.tensor([[True, True, True, False], [True, True, True, True]])[:batch_size, :agents]
    central = {
        "self": torch.randn(batch_size, agents, 9),
        "pursuers": torch.randn(batch_size, agents, agents, 7),
        "evaders": torch.randn(batch_size, 8, 7),
        "obstacles": torch.randn(batch_size, 5, 5),
        "pursuer_mask": active[:, :, None] & active[:, None, :],
        "evader_mask": torch.ones(batch_size, 8, dtype=torch.bool),
        "obstacle_mask": torch.ones(batch_size, 5, dtype=torch.bool),
        "active_mask": active,
    }
    return {
        "local_obs": local,
        "next_local_obs": {key: value + 0.01 for key, value in local.items()},
        "global_state": central,
        "next_global_state": {key: value + 0.01 if value.is_floating_point() else value.clone() for key, value in central.items()},
        "actions": torch.randn(batch_size, agents, 2).clamp(-1.0, 1.0),
        "rewards": torch.randn(batch_size, agents),
        "active_mask": active,
        "terminated": torch.zeros(batch_size, agents, dtype=torch.bool),
        "truncated": torch.zeros(batch_size, agents, dtype=torch.bool),
    }


def test_central_sac_update_is_finite_and_changes_actor_and_critics() -> None:
    batch = make_batch()
    encoder = LocalEntityTokenEncoderConfig(hidden_dim=16, num_heads=4, num_layers=1)
    actor = RadialActorConfig(hidden_dim=16, a_max=0.8, decision_dt=0.5)
    critic = CentralCriticConfig(hidden_dim=16, num_heads=4, num_layers=1, max_agents=4)
    trainer = CentralSACTrainer(
        encoder_config=encoder,
        actor_config=actor,
        critic_config=critic,
        config=CentralSACConfig(hidden_dim=16),
        device="cpu",
    )
    actor_before = next(trainer.actor.parameters()).detach().clone()
    critic_before = next(trainer.critic1.parameters()).detach().clone()
    metrics = trainer.update(batch)
    assert metrics["finite"] == 1.0
    assert metrics["active_count"] == 7.0
    assert np.isfinite(metrics["twin_q_gap"])
    assert not torch.equal(actor_before, next(trainer.actor.parameters()).detach())
    assert not torch.equal(critic_before, next(trainer.critic1.parameters()).detach())


def test_local_and_central_modes_accept_the_same_joint_batch() -> None:
    batch = make_batch()
    encoder = LocalEntityTokenEncoderConfig(hidden_dim=16, num_heads=4, num_layers=1)
    actor = RadialActorConfig(hidden_dim=16)
    local = LocalSACTrainer(encoder_config=encoder, actor_config=actor, config=LocalSACConfig(hidden_dim=16))
    central = CentralSACTrainer(
        encoder_config=encoder,
        actor_config=actor,
        critic_config=CentralCriticConfig(hidden_dim=16, num_heads=4, num_layers=1, max_agents=4),
        config=CentralSACConfig(hidden_dim=16),
    )
    assert local.update(batch)["finite"] == 1.0
    assert central.update(batch)["finite"] == 1.0
