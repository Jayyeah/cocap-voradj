"""MAPPO regression for zero-filled terminal all-masked local rows."""

import numpy as np
import torch

from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LegacyVorAdjFeatureBackbone,
    LegacyVorAdjFeatureBackboneConfig,
)
from cocap_voradj.models.small_step_ac import CategoricalGridActor


def _observations(batch: int = 3) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260923)
    return {
        "self": torch.randn(batch, 9, generator=generator),
        "pursuers": torch.randn(batch, 8, 7, generator=generator),
        "evaders": torch.randn(batch, 8, 7, generator=generator),
        "obstacles": torch.randn(batch, 5, 5, generator=generator),
        "types": torch.tensor([0] + [1] * 8 + [2] * 8 + [3] * 5).expand(batch, -1).clone(),
        "masks": torch.zeros(batch, 22, dtype=torch.bool),
    }


def test_legacy_mappo_backbone_all_masked_terminal_row_is_finite_and_immutable():
    torch.manual_seed(7)
    config = LegacyVorAdjFeatureBackboneConfig(
        hidden_dim=32, num_heads=4, num_layers=1, dropout=0.0
    )
    actor = CategoricalGridActor(
        LegacyVorAdjFeatureBackbone(config),
        np.asarray([(a, w) for a in (-0.4, 0.0, 0.4) for w in (-1.0, 0.0, 1.0)], dtype=np.float32),
    ).eval()
    observations = _observations()
    original = {key: value.clone() for key, value in observations.items()}

    with torch.no_grad():
        logits = actor.distribution(observations).logits

    assert torch.isfinite(logits).all()
    assert all(torch.equal(value, original[key]) for key, value in observations.items())


def test_all_masked_fallback_matches_explicit_self_valid_reference():
    torch.manual_seed(11)
    config = LegacyVorAdjFeatureBackboneConfig(
        hidden_dim=32, num_heads=4, num_layers=1, dropout=0.0
    )
    actor = CategoricalGridActor(
        LegacyVorAdjFeatureBackbone(config),
        np.asarray([(a, w) for a in (-0.4, 0.0, 0.4) for w in (-1.0, 0.0, 1.0)], dtype=np.float32),
    ).eval()
    actual = _observations(batch=1)
    reference = {key: value.clone() for key, value in actual.items()}
    reference["masks"][0, 0] = True

    with torch.no_grad():
        actual_logits = actor.distribution(actual).logits
        reference_logits = actor.distribution(reference).logits

    torch.testing.assert_close(actual_logits, reference_logits, rtol=0.0, atol=0.0)
