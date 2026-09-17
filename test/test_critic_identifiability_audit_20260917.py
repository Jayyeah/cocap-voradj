from __future__ import annotations

import numpy as np
import torch

from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LegacyVorAdjFeatureBackboneConfig,
)
from cocap_voradj.models.critic_identifiability import LocalQ, NeighborQ
from tools.run_critic_identifiability_audit_20260917 import (
    EVENTS,
    EVENT_TO_ID,
    event_flags,
    split_assignment,
    verify_mc,
)


def _observation(batch: int, neighbors: int | None = None):
    prefix = (batch,) if neighbors is None else (batch, neighbors)
    return {
        "self": torch.randn(*prefix, 9),
        "pursuers": torch.randn(*prefix, 8, 7),
        "evaders": torch.randn(*prefix, 8, 7),
        "obstacles": torch.randn(*prefix, 5, 5),
        "masks": torch.ones(*prefix, 22, dtype=torch.bool),
        "types": torch.cat((
            torch.zeros(*prefix, 1, dtype=torch.long),
            torch.ones(*prefix, 8, dtype=torch.long),
            torch.full((*prefix, 8), 2, dtype=torch.long),
            torch.full((*prefix, 5), 3, dtype=torch.long),
        ), dim=-1),
    }


def test_capture_transition_and_recovery_are_not_conflated():
    terminated = np.asarray([True, True, True, True])
    flags, phase, _ = event_flags(
        "mixed", {"targets": {}}, [{"capture_type": "loose", "participants": [0, 1, 2]}],
        False, terminated, 10, None, 0,
    )
    assert phase == "pre_capture"
    assert flags[EVENT_TO_ID["normal_capture"]]
    assert flags[EVENT_TO_ID["capture_terminal_transition"]]
    assert not flags[EVENT_TO_ID["early_recovery"]]

    flags, phase, _ = event_flags(
        "mixed", {"targets": {}}, [], False,
        np.asarray([False, False, False, False]), 11, 10, 1,
    )
    assert phase == "post_capture"
    assert flags[EVENT_TO_ID["early_recovery"]]
    assert not flags[EVENT_TO_ID["normal_capture"]]


def test_mc_recompute_and_episode_split_contract():
    rewards = np.asarray([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float32).reshape(3, 1)
    expected = np.asarray([[1 + 0.9 * 2 + 0.9**2 * 3], [2 + 0.9 * 3], [3]], dtype=np.float32)
    bank = {
        "episode_id": np.asarray([7, 7, 7]),
        "reward": rewards,
        "mc_return": expected,
    }
    assert verify_mc(bank, 0.9) < 1e-5
    labels = split_assignment(0, 18)
    assert len(labels) == 18
    assert set(labels) == {0, 1, 2}
    assert labels == sorted(labels)


def test_neighbor_q_is_permutation_invariant_and_differentiable():
    torch.manual_seed(3)
    cfg = LegacyVorAdjFeatureBackboneConfig(
        hidden_dim=32, num_heads=4, num_layers=1, dropout=0.0,
    )
    model = NeighborQ(cfg, max_neighbors=3).eval()
    obs = _observation(2)
    action = torch.randn(2, 2)
    neighbor_obs = _observation(2, 3)
    neighbor_action = torch.randn(2, 3, 2)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    first = model(obs, action, neighbor_obs, neighbor_action, mask)
    permutation = torch.tensor([2, 0, 1])
    permuted_obs = {key: value[:, permutation] for key, value in neighbor_obs.items()}
    second = model(
        obs, action, permuted_obs, neighbor_action[:, permutation], mask[:, permutation],
    )
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)
    first.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_local_q_shape_and_event_enum_are_stable():
    cfg = LegacyVorAdjFeatureBackboneConfig(
        hidden_dim=32, num_heads=4, num_layers=1, dropout=0.0,
    )
    model = LocalQ(cfg)
    output = model(_observation(5), torch.randn(5, 2))
    assert output.shape == (5,)
    assert len(EVENTS) == len(EVENT_TO_ID) == 12
