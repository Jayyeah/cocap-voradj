from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from cocap_voradj.models.continuous.box_actor import (
    BoxActorConfig,
    SquashedGaussianAccelerationAngularVelocityActor,
)
from cocap_voradj.models.continuous.local_entity_token_encoder import LegacyVorAdjFeatureBackbone
from cocap_voradj.models.small_step_ac import CentralValueNetwork
from cocap_voradj.training.small_step_ac import MAPPOConfig, MAPPOTrainer
from cocap_voradj.training.continuous.formal_config import resolve_ladder_config
from tools.run_small_step_ac_migration import configure_environment, make_components


ROOT = Path(__file__).resolve().parents[1]


class _FeatureEncoder(nn.Module):
    """Small tensor-returning encoder for action-head math tests."""

    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_dim=hidden_dim)
        self.decision_feature_dim = hidden_dim

    def forward(self, obs):
        return obs["feature"]


def _actor() -> SquashedGaussianAccelerationAngularVelocityActor:
    return SquashedGaussianAccelerationAngularVelocityActor(
        _FeatureEncoder(),
        BoxActorConfig(
            hidden_dim=8,
            a_max=0.4,
            w_max=math.pi / 6.0,
            dropout=0.0,
            orthogonal_policy_head=True,
            mean_output_gain=0.01,
            initial_log_std=0.0,
            saturation_threshold=0.99,
        ),
    )


def _global_obs(steps: int, agents: int = 2):
    return {
        "self": torch.randn(steps, agents, 9),
        "pursuers": torch.randn(steps, agents, agents, 7),
        "evaders": torch.randn(steps, 1, 7),
        "obstacles": torch.randn(steps, 1, 5),
        "pursuer_mask": torch.ones(steps, agents, agents, dtype=torch.bool),
        "evader_mask": torch.ones(steps, 1, dtype=torch.bool),
        "obstacle_mask": torch.ones(steps, 1, dtype=torch.bool),
        "active_mask": torch.ones(steps, agents, dtype=torch.bool),
    }


def test_aw_v2_runner_reuses_parent_legacy_backbone_and_ppo_recipe() -> None:
    root = resolve_ladder_config(
        ROOT / "configs/experiments/mappo_aw_v2_20260831/seed1.yaml"
    )
    config = configure_environment(root, "mappo_aw_v2")
    config["seed"] = root["seed"]
    trainer, replay = make_components(config, "mappo_aw_v2", "cpu")
    assert replay is None
    assert isinstance(trainer.actor, SquashedGaussianAccelerationAngularVelocityActor)
    assert isinstance(trainer.actor.encoder, LegacyVorAdjFeatureBackbone)
    assert trainer.actor.encoder.config.hidden_dim == 256
    assert trainer.actor.encoder.config.num_heads == 8
    assert trainer.actor.encoder.config.num_layers == 4
    assert trainer.actor.encoder.config.max_pursuers == 8
    assert trainer.actor.config.a_max == 0.4
    assert trainer.actor.config.w_max == pytest.approx(math.pi / 6.0)
    assert trainer.actor.config.initial_log_std == 0.0
    assert trainer.config.ppo_epochs == 3
    assert trainer.config.minibatches == 2
    assert trainer.config.actor_lr == 3e-5
    assert trainer.config.critic_lr == 1e-4
    assert trainer.config.target_kl == 0.02
    assert trainer.value_normalizer is not None


def test_aw_v2_head_initialization_sampling_determinism_and_bounds() -> None:
    torch.manual_seed(2026083101)
    actor = _actor().eval()
    obs = {"feature": torch.randn(64, 8)}
    distribution, log_std = actor.distribution(obs)

    assert float(distribution.loc.detach().abs().max()) < 0.05
    torch.testing.assert_close(log_std, torch.zeros_like(log_std), rtol=0.0, atol=0.0)
    deterministic, deterministic_logp, deterministic_latent = actor.sample(obs, deterministic=True)
    expected_scale = torch.tensor([0.4, math.pi / 6.0])
    torch.testing.assert_close(deterministic_latent, distribution.loc)
    torch.testing.assert_close(deterministic, torch.tanh(distribution.loc) * expected_scale)
    assert torch.isfinite(deterministic_logp).all()

    stochastic_1, logp_1, latent_1 = actor.sample(obs, deterministic=False)
    stochastic_2, logp_2, latent_2 = actor.sample(obs, deterministic=False)
    assert not torch.equal(latent_1, latent_2)
    for action, logp in ((stochastic_1, logp_1), (stochastic_2, logp_2)):
        assert torch.isfinite(action).all() and torch.isfinite(logp).all()
        detached = action.detach()
        assert float(detached[:, 0].abs().max()) <= 0.4
        assert float(detached[:, 1].abs().max()) <= math.pi / 6.0
    initial_saturation = (torch.tanh(latent_1.detach()).abs() >= 0.99).float().mean()
    assert float(initial_saturation) < 0.1


def test_aw_v2_log_prob_uses_exact_affine_tanh_jacobian() -> None:
    torch.manual_seed(2026083102)
    actor = _actor().eval()
    obs = {"feature": torch.randn(5, 8)}
    latent = torch.tensor([[-5.0, 5.0], [-2.0, 2.0], [0.0, 0.0], [2.0, -2.0], [5.0, -5.0]])
    distribution, _ = actor.distribution(obs)
    actual = actor.log_prob_from_distribution(distribution, latent)
    scale = torch.tensor([0.4, math.pi / 6.0])
    manual_log_det = (
        torch.log(scale) + torch.log1p(-torch.tanh(latent).square())
    ).sum(dim=-1)
    expected = distribution.log_prob(latent).sum(dim=-1) - manual_log_det
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)

    action = torch.tanh(latent) * scale
    # The inverse path is valid away from float32 saturation and remains an
    # explicitly validated convenience for external physical actions.
    middle = slice(1, 4)
    torch.testing.assert_close(actor.log_prob(obs, action)[middle], actual[middle], rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError):
        actor.log_prob({"feature": obs["feature"][:1]}, torch.tensor([[0.41, 0.0]]))
    with pytest.raises(ValueError):
        actor.log_prob({"feature": obs["feature"][:1]}, torch.tensor([[float("nan"), 0.0]]))


def test_saturated_rollout_latent_is_authoritative_for_ppo_ratio() -> None:
    torch.manual_seed(2026083103)
    actor = _actor().eval()
    steps, agents = 2, 2
    flat_obs = {"feature": torch.randn(steps * agents, 8)}
    latent = torch.tensor([[8.0, -8.0], [10.0, -10.0], [0.2, -0.3], [-0.4, 0.5]])
    action = actor._squash(latent)
    exact_logp = actor.log_prob_from_latent(flat_obs, latent)
    inverse_logp = actor.log_prob(flat_obs, action)
    assert float((exact_logp[:2] - inverse_logp[:2]).detach().abs().max()) > 5.0

    value = CentralValueNetwork(
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        max_agents=agents,
        max_evaders=1,
        max_obstacles=1,
    )
    trainer = MAPPOTrainer(
        actor,
        value,
        MAPPOConfig(
            ppo_epochs=1,
            minibatches=1,
            actor_lr=0.0,
            critic_lr=0.0,
            entropy_coef=0.0,
            target_kl=0.02,
            value_norm=True,
        ),
        "cpu",
    )
    batch = {
        "local_obs": {"feature": flat_obs["feature"].reshape(steps, agents, 8)},
        "global_obs": _global_obs(steps, agents),
        "actions": action.reshape(steps, agents, 2),
        "latent": latent.reshape(steps, agents, 2),
        "log_prob": exact_logp.reshape(steps, agents),
        "values": torch.zeros(steps, agents),
        "next_values": torch.zeros(steps, agents),
        "rewards": torch.tensor([[0.0, 0.2], [0.4, 0.6]]),
        "active_mask": torch.ones(steps, agents, dtype=torch.bool),
        "terminated": torch.zeros(steps, agents, dtype=torch.bool),
        "truncated": torch.zeros(steps, agents, dtype=torch.bool),
        "episode_end": torch.zeros(steps, agents, dtype=torch.bool),
    }
    metrics = trainer.update(batch, categorical=False)
    assert metrics["approx_kl_max"] < 1e-6
    assert metrics["clip_fraction"] == 0.0
    assert metrics["post_tanh_saturation_ratio"] > 0.0
    assert metrics["pre_tanh_mean_abs_max"] < 0.05
    assert metrics["log_std_a_mean"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["log_std_w_mean"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["critic_grad_norm"] == metrics["value_grad_norm"]
    for key in (
        "entropy",
        "entropy_base_gaussian",
        "entropy_squashed_physical_mc",
        "action_magnitude_mean",
        "actor_grad_norm",
        "value_loss",
        "explained_variance",
    ):
        assert np.isfinite(metrics[key]), key
