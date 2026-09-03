import json

import numpy as np
import pytest
import torch

from tools import collect_iqn_aw_teacher_dataset_20260903 as collector
from tools import distill_mappo_actor_from_iqn_20260903 as distill


def test_fixed_midpoint_q_uses_deterministic_midpoints(monkeypatch):
    monkeypatch.setattr(
        collector,
        "stack_obs",
        lambda observations, device: {"dummy": torch.zeros(len(observations), 1, device=device)},
    )

    class Teacher:
        captured_tau = None

        def __call__(self, batch, *, num_tau, mode, tau):
            assert num_tau == 32
            assert mode == "voradj"
            self.captured_tau = tau.detach().cpu()
            q = torch.arange(9, device=tau.device, dtype=torch.float32)
            return {"q_values": q.view(1, 1, 9).expand(len(batch["dummy"]), 32, 9)}

    teacher = Teacher()
    q, actions = collector.fixed_midpoint_q(teacher, [{}, {}], "cpu")
    expected_tau = (torch.arange(32, dtype=torch.float32) + 0.5) / 32.0
    assert torch.equal(teacher.captured_tau, expected_tau)
    assert q.shape == (2, 9)
    assert actions.tolist() == [8, 8]


def test_dataset_loader_rejects_non_greedy_labels(tmp_path):
    shard_root = tmp_path / "shards"
    shard_root.mkdir()
    shard = shard_root / "shard_0000.npz"
    q = np.asarray([[0.0, 1.0] + [0.0] * 7], dtype=np.float32)
    np.savez_compressed(shard, teacher_q=q, greedy_action=np.asarray([0], dtype=np.int64))
    manifest = {
        "schema": distill.DATASET_SCHEMA,
        "status": "complete",
        "row_count": 1,
        "shards": [{"path": "shards/shard_0000.npz", "sha256": distill.sha256_file(shard)}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="greedy_action"):
        distill.load_dataset(tmp_path)


def test_distilled_actor_checkpoint_is_weights_only_safe(tmp_path):
    actor = torch.nn.Linear(3, 9)
    destination = tmp_path / "actor.pt"
    distill.save_actor_checkpoint(
        destination,
        actor,
        {
            "teacher_checkpoint_sha256": "a" * 64,
            "epoch": 1,
            "validation": {"action_agreement": 1.0},
        },
    )
    payload = torch.load(destination, map_location="cpu", weights_only=True)
    assert payload["schema"] == distill.CHECKPOINT_SCHEMA
    clone = torch.nn.Linear(3, 9)
    clone.load_state_dict(payload["actor_state_dict"], strict=True)
    for expected, actual in zip(actor.parameters(), clone.parameters()):
        assert torch.equal(expected, actual)
