from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LocalEntityTokenEncoder,
    LocalEntityTokenEncoderConfig,
)
from cocap_voradj.training.continuous.formal_config import resolve_formal_config
from tools.run_continuous_ctde_training import _make_trainer


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "configs/experiments/continuous_marl_20260804/p6_formal_central_masac_4v1.yaml"
LEGACY = ROOT / "artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage1_4v1_step_2000000.pt"


def test_legacy_encoder_exact_key_and_tensor_transfer() -> None:
    payload = torch.load(LEGACY, map_location="cpu", weights_only=False)
    source = payload["state_dict"]
    config = resolve_formal_config(FORMAL)
    actor = config["actor"]
    encoder = LocalEntityTokenEncoder(
        LocalEntityTokenEncoderConfig(
            hidden_dim=actor["hidden_dim"],
            num_heads=actor["num_heads"],
            num_layers=actor["num_layers"],
            self_feature_dim=actor["self_feature_dim"],
            max_pursuers=actor["max_pursuers"],
            max_evaders=actor["max_evaders"],
            max_obstacles=actor["max_obstacles"],
            dropout=0.0,
        )
    )
    encoder.load_legacy_iqn_state_dict(source, strict=True)
    prefixes = ("encoders.", "type_embedding.", "transformer.")
    expected = {key: value for key, value in source.items() if key.startswith(prefixes)}
    actual = encoder.state_dict()
    assert set(expected) == set(actual)
    for key in expected:
        assert torch.equal(expected[key], actual[key])
    assert not any(key.startswith(("pis", "head", "quantile", "mean_head", "log_std_head")) for key in expected)


def test_encoder_shape_mismatch_fails() -> None:
    payload = torch.load(LEGACY, map_location="cpu", weights_only=False)
    encoder = LocalEntityTokenEncoder(
        LocalEntityTokenEncoderConfig(hidden_dim=128, num_heads=8, num_layers=4, self_feature_dim=9)
    )
    with pytest.raises(RuntimeError):
        encoder.load_legacy_iqn_state_dict(payload["state_dict"], strict=True)


def test_runner_legacy_init_keeps_policy_and_critic_random() -> None:
    config = resolve_formal_config(FORMAL)
    config["initialization"]["actor_encoder"] = {
        "mode": "legacy_iqn",
        "checkpoint": str(LEGACY),
        "strict_shape_match": True,
        "freeze_env_steps": 0,
    }
    trainer = _make_trainer(config, "cpu")
    payload = torch.load(LEGACY, map_location="cpu", weights_only=False)
    source = payload["state_dict"]
    prefixes = ("encoders.", "type_embedding.", "transformer.")
    for name, parameter in trainer.actor.encoder.named_parameters():
        key = name
        assert key in source
        assert torch.equal(parameter.detach().cpu(), source[key])
    assert not any(name.startswith(("mean_head", "log_std_head", "policy")) for name in source)
    critic_keys = set(trainer.critic1.state_dict())
    assert not any(key in critic_keys for key in source if key.startswith(("encoders.", "transformer.")))
    if "type_embedding.weight" in critic_keys:
        assert not torch.equal(trainer.critic1.state_dict()["type_embedding.weight"], source["type_embedding.weight"])
