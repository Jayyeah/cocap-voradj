from __future__ import annotations

import torch
import pytest

from cocap_voradj.models.iqn import CoCapIQN, CoCapNetConfig
from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LocalEntityTokenEncoder,
    LocalEntityTokenEncoderConfig,
)
from cocap_voradj.models.continuous.radial_actor import RadialActorConfig, RadialSquashedGaussianActor


def make_obs(batch: int = 3, config: LocalEntityTokenEncoderConfig | None = None) -> dict[str, torch.Tensor]:
    config = config or LocalEntityTokenEncoderConfig(hidden_dim=32, num_heads=4, num_layers=2)
    tokens = config.max_pursuers + config.max_evaders + config.max_obstacles + 1
    types = torch.zeros(batch, tokens, dtype=torch.long)
    types[:, 1 : 1 + config.max_pursuers] = 1
    types[:, 1 + config.max_pursuers : 1 + config.max_pursuers + config.max_evaders] = 2
    types[:, 1 + config.max_pursuers + config.max_evaders :] = 3
    masks = torch.ones(batch, tokens, dtype=torch.bool)
    masks[:, -2:] = False
    return {
        "self": torch.randn(batch, config.self_feature_dim),
        "pursuers": torch.randn(batch, config.max_pursuers, 7),
        "evaders": torch.randn(batch, config.max_evaders, 7),
        "obstacles": torch.randn(batch, config.max_obstacles, 5),
        "masks": masks,
        "types": types,
    }


def test_encoder_maps_legacy_iqn_frontend_without_touching_old_model() -> None:
    torch.manual_seed(20260804)
    config = LocalEntityTokenEncoderConfig(hidden_dim=32, num_heads=4, num_layers=2)
    legacy_config = CoCapNetConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        self_feature_dim=config.self_feature_dim,
        max_pursuers=config.max_pursuers,
        max_evaders=config.max_evaders,
        max_obstacles=config.max_obstacles,
    )
    legacy = CoCapIQN(legacy_config).eval()
    encoder = LocalEntityTokenEncoder(config).eval()
    mapped = encoder.map_legacy_iqn_encoder_keys(legacy.state_dict())
    assert set(mapped) == set(encoder.state_dict())
    encoder.load_legacy_iqn_state_dict(legacy.state_dict())
    obs = make_obs()
    with torch.no_grad():
        old_features = legacy.features(obs)
        new_features = encoder(obs)
    for key in ("self_token", "mean_context", "max_context", "evader_features", "evader_mask"):
        assert torch.allclose(old_features[key], new_features[key], atol=1e-6, rtol=1e-6)


def test_encoder_entity_permutation_preserves_pooled_context() -> None:
    torch.manual_seed(20260805)
    encoder = LocalEntityTokenEncoder(LocalEntityTokenEncoderConfig(hidden_dim=32, num_heads=4, num_layers=2)).eval()
    obs = make_obs(batch=1)
    perm = torch.tensor([2, 0, 7, 1, 3, 4, 5, 6])
    permuted = {key: value.clone() for key, value in obs.items()}
    permuted["pursuers"] = obs["pursuers"][:, perm]
    start = 1
    permuted["types"][:, start : start + 8] = obs["types"][:, start : start + 8][:, perm]
    permuted["masks"][:, start : start + 8] = obs["masks"][:, start : start + 8][:, perm]
    with torch.no_grad():
        first = encoder(obs)
        second = encoder(permuted)
    assert torch.allclose(first["self_token"], second["self_token"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(first["mean_context"], second["mean_context"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(first["max_context"], second["max_context"], atol=1e-5, rtol=1e-5)


def test_encoder_supports_4_8_12_pursuer_token_shapes() -> None:
    for count in (4, 8, 12):
        config = LocalEntityTokenEncoderConfig(hidden_dim=16, num_heads=4, num_layers=1, max_pursuers=count)
        encoder = LocalEntityTokenEncoder(config).eval()
        obs = make_obs(batch=2, config=config)
        with torch.no_grad():
            features = encoder(obs)
        assert features["tokens"].shape == (2, encoder.token_count, config.hidden_dim)
        assert features["self_token"].shape == (2, config.hidden_dim)


def test_radial_acceleration_actor_disk_logprob_gradient_and_contract() -> None:
    torch.manual_seed(20260806)
    encoder = LocalEntityTokenEncoder(LocalEntityTokenEncoderConfig(hidden_dim=32, num_heads=4, num_layers=2))
    actor = RadialSquashedGaussianActor(
        encoder,
        RadialActorConfig(hidden_dim=32, a_max=0.8, decision_dt=0.5, dropout=0.0),
    )
    actor.eval()
    obs = make_obs()
    action, log_prob, latent = actor.sample(obs)
    assert action.shape == (3, 2)
    assert latent.shape == (3, 2)
    assert torch.all(torch.linalg.vector_norm(action, dim=-1) <= 0.8 + 1e-6)
    assert torch.isfinite(log_prob).all()
    inverse_log_prob = actor.log_prob(obs, action)
    assert torch.isfinite(inverse_log_prob).all()
    (-log_prob.mean()).backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in actor.parameters())
    action, diagnostics = actor.act_with_acceleration_contract(obs, deterministic=True)
    assert torch.all(torch.linalg.vector_norm(action, dim=-1) <= 0.8 + 1e-6)
    assert len(diagnostics) == 3
    assert all("validation_delta" in item and "jerk" in item for item in diagnostics)


def test_acceleration_sampling_matches_a_max_contract() -> None:
    torch.manual_seed(2026080611)
    config = LocalEntityTokenEncoderConfig(hidden_dim=32, num_heads=4, num_layers=2)
    actor = RadialSquashedGaussianActor(
        LocalEntityTokenEncoder(config),
        RadialActorConfig(hidden_dim=32, a_max=0.8, decision_dt=0.5, dropout=0.0),
    )
    obs = make_obs(batch=4, config=config)
    command, log_prob, _ = actor.sample(obs)
    assert torch.isfinite(command).all()
    assert torch.isfinite(log_prob).all()
    assert torch.linalg.vector_norm(command, dim=-1).max() <= 0.8 + 1e-5
    (-log_prob.mean()).backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in actor.parameters())

    adapter = actor.adapter
    for command_i in command.detach().numpy():
        _, diagnostics = adapter.validate_with_diagnostics(command_i)
    assert diagnostics.validation_rate == 0.0


def test_radial_actor_rejects_outside_disk_logprob() -> None:
    actor = RadialSquashedGaussianActor(
        LocalEntityTokenEncoder(LocalEntityTokenEncoderConfig(hidden_dim=32, num_heads=4, num_layers=2)),
        RadialActorConfig(hidden_dim=32, a_max=0.8),
    )
    with pytest.raises(ValueError):
        actor.log_prob(make_obs(batch=1), torch.tensor([[0.81, 0.0]]))
