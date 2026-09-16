from __future__ import annotations

import copy

import numpy as np
import torch

from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LegacyVorAdjFeatureBackbone,
)
from cocap_voradj.models.continuous.local_td3 import LocalTD3Actor, LocalTD3Critic
from cocap_voradj.models.iqn import CoCapIQN, CoCapNetConfig
from cocap_voradj.training.td3_warmstart import (
    backbone_config_from_iqn,
    nearest_aw9_index,
    tensor_state_sha256,
    transfer_iqn_backbone,
)


def _teacher() -> CoCapIQN:
    return CoCapIQN(
        CoCapNetConfig(
            hidden_dim=32,
            num_heads=4,
            num_layers=1,
            action_size=9,
            self_feature_dim=9,
            max_pursuers=8,
            max_evaders=8,
            max_obstacles=5,
            num_quantiles=8,
            num_cosine_features=16,
            architecture="voradj_single_head",
            pursuing_embed_dim=8,
        )
    ).eval()


def test_iqn_transfer_is_actor_backbone_only_and_strict() -> None:
    teacher = _teacher()
    config = backbone_config_from_iqn(teacher.config)
    actor = LocalTD3Actor(LegacyVorAdjFeatureBackbone(config))
    critic1 = LocalTD3Critic(LegacyVorAdjFeatureBackbone(config))
    critic2 = LocalTD3Critic(LegacyVorAdjFeatureBackbone(config))
    actor_head_before = tensor_state_sha256(actor.head.state_dict())
    q1_before = tensor_state_sha256(critic1.state_dict())
    q2_before = tensor_state_sha256(critic2.state_dict())
    report = transfer_iqn_backbone(actor, teacher)
    assert report["mapped_key_count"] == len(actor.backbone.state_dict())
    # The production four-layer checkpoint maps 86 keys; this deliberately
    # smaller one-layer fixture has fewer, and strict equality to the target
    # backbone is the invariant that matters.
    assert report["mapped_key_count"] > 0
    assert report["critic_transfer"] == "none"
    assert not any(key.startswith("single_head.") for key in report["mapped_keys"])
    assert tensor_state_sha256(actor.head.state_dict()) == actor_head_before
    assert tensor_state_sha256(critic1.state_dict()) == q1_before
    assert tensor_state_sha256(critic2.state_dict()) == q2_before


def test_transferred_feature_is_exact_in_eval_mode() -> None:
    torch.manual_seed(7)
    teacher = _teacher()
    actor = LocalTD3Actor(
        LegacyVorAdjFeatureBackbone(backbone_config_from_iqn(teacher.config))
    ).eval()
    transfer_iqn_backbone(actor, teacher)
    batch = 3
    tokens = 1 + 8 + 8 + 5
    obs = {
        "self": torch.randn(batch, 9),
        "pursuers": torch.randn(batch, 8, 7),
        "evaders": torch.randn(batch, 8, 7),
        "obstacles": torch.randn(batch, 5, 5),
        "masks": torch.ones(batch, tokens, dtype=torch.bool),
        "types": torch.cat(
            [
                torch.zeros(batch, 1, dtype=torch.long),
                torch.ones(batch, 8, dtype=torch.long),
                torch.full((batch, 8), 2, dtype=torch.long),
                torch.full((batch, 5), 3, dtype=torch.long),
            ],
            dim=1,
        ),
    }
    with torch.no_grad():
        legacy = teacher.voradj_single_feature(obs, teacher.features(obs))
        transferred = actor.backbone(obs)
    torch.testing.assert_close(legacy, transferred, atol=1e-6, rtol=1e-6)


def test_nearest_aw9_index_uses_component_normalization() -> None:
    grid = torch.tensor(
        [(a, w) for a in (-0.4, 0.0, 0.4) for w in (-np.pi / 6, 0.0, np.pi / 6)],
        dtype=torch.float32,
    )
    assert torch.equal(nearest_aw9_index(grid), torch.arange(9))
