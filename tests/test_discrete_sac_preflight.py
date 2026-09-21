from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from cocap_voradj.models.shared_local_ac import (
    EXPECTED_AW9,
    SharedLocalACNetworkConfig,
    canonical_aw9_grid,
)
from cocap_voradj.training.discrete_sac import (
    DiscreteSACConfig,
    DiscreteSACTrainer,
    atomic_torch_save,
    categorical_target_value,
)
from cocap_voradj.training.shared_local_ac import state_hash


def observation(seed: int = 0, *, with_evader: bool = False) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    masks = np.zeros(1 + 8 + 8 + 5, dtype=bool)
    masks[0] = True
    masks[1:3] = True
    if with_evader:
        masks[9] = True
    return {
        "self": rng.normal(size=(9,)).astype(np.float32),
        "pursuers": rng.normal(size=(8, 7)).astype(np.float32),
        "evaders": rng.normal(size=(8, 7)).astype(np.float32),
        "obstacles": rng.normal(size=(5, 5)).astype(np.float32),
        "masks": masks,
        "types": np.asarray([0] + [1] * 8 + [2] * 8 + [3] * 5, dtype=np.int64),
    }


def batch(size: int = 8) -> dict[str, object]:
    obs = [observation(index, with_evader=index % 2 == 0) for index in range(size)]
    next_obs = [observation(index + 100, with_evader=index % 2 == 0) for index in range(size)]
    stack = lambda items: {
        key: torch.as_tensor(
            np.stack([item[key] for item in items]),
            dtype=torch.long if key == "types" else torch.bool if key == "masks" else torch.float32,
        )
        for key in items[0]
    }
    return {
        "obs": stack(obs),
        "next_obs": stack(next_obs),
        "actions": torch.arange(size, dtype=torch.long) % 9,
        "rewards": torch.linspace(-1.0, 1.0, size),
        "terminated": torch.zeros(size, dtype=torch.bool),
    }


def test_categorical_expectation_uses_all_actions_and_twin_minimum() -> None:
    logits = torch.log(torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float32))
    q1 = torch.tensor([[10.0, 20.0, 30.0]])
    q2 = torch.tensor([[1.0, 25.0, 5.0]])
    alpha = 0.5
    expected = (torch.tensor([0.1, 0.2, 0.7]) * (torch.tensor([1.0, 20.0, 5.0]) - alpha * logits[0])).sum()
    actual = categorical_target_value(logits, q1, q2, alpha)
    assert torch.allclose(actual, expected[None], atol=1e-6)


def test_aw9_mapping_is_exact_existing_order() -> None:
    np.testing.assert_allclose(canonical_aw9_grid(), np.asarray(EXPECTED_AW9), atol=1e-7)
    assert canonical_aw9_grid().shape == (9, 2)
    np.testing.assert_allclose(canonical_aw9_grid()[0], (-0.4, -np.pi / 6.0), atol=1e-7)
    np.testing.assert_allclose(canonical_aw9_grid()[-1], (0.4, np.pi / 6.0), atol=1e-7)


def test_twin_q_discrete_sac_update_is_finite_and_changes_actor() -> None:
    torch.manual_seed(2026092102)
    network = SharedLocalACNetworkConfig(hidden_dim=16, num_heads=4, num_layers=1, action_size=9)
    trainer = DiscreteSACTrainer(
        network,
        DiscreteSACConfig(alpha_init=0.2, max_grad_norm=0.5),
        device="cpu",
    )
    before = state_hash(trainer.actor)
    metrics = trainer.update(batch())
    assert metrics["finite"] == 1.0
    assert trainer.update_count == 1
    assert state_hash(trainer.actor) != before
    assert np.isfinite(metrics["td_target_mean"])


def test_atomic_finite_checkpoint_save_load_round_trip() -> None:
    network = SharedLocalACNetworkConfig(hidden_dim=16, num_heads=4, num_layers=1, action_size=9)
    config = DiscreteSACConfig(alpha_init=0.2, max_grad_norm=0.5)
    trainer = DiscreteSACTrainer(network, config, device="cpu")
    trainer.update(batch())
    contract = {"schema": "test-aw9", "action_size": 9, "task": "coverage"}
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "latest.pt"
        atomic_torch_save(trainer.checkpoint_payload(contract=contract, global_step=8), checkpoint)
        restored = DiscreteSACTrainer(network, config, device="cpu")
        restored.load_payload(torch.load(checkpoint, map_location="cpu", weights_only=False), contract)
        assert state_hash(restored.actor) == state_hash(trainer.actor)
        assert state_hash(restored.critic1) == state_hash(trainer.critic1)
        assert restored.update_count == trainer.update_count
