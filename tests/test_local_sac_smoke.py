from __future__ import annotations

import numpy as np
import pytest
import torch

from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoderConfig
from cocap_voradj.models.continuous.radial_actor import RadialActorConfig
from cocap_voradj.training.continuous.joint_replay import JointReplayBuffer
from cocap_voradj.training.continuous.local_sac import (
    LocalSACConfig,
    LocalSACTrainer,
    sac_bootstrap_mask,
)


def add_transition(replay: JointReplayBuffer, value: float) -> None:
    n = replay.max_agents
    obs = {
        "self": np.full((n, 9), value, dtype=np.float32),
        "pursuers": np.full((n, 8, 7), value, dtype=np.float32),
        "evaders": np.full((n, 8, 7), value, dtype=np.float32),
        "obstacles": np.full((n, 5, 5), value, dtype=np.float32),
        "masks": np.ones((n, 22), dtype=np.float32),
        "types": np.ones((n, 22), dtype=np.float32),
    }
    replay.add(
        local_obs=obs,
        next_local_obs={key: item + 0.01 for key, item in obs.items()},
        global_state=np.zeros((n, 8), dtype=np.float32),
        next_global_state=np.ones((n, 8), dtype=np.float32),
        actions=np.zeros((n, 2), dtype=np.float32),
        rewards=np.full(n, 0.1, dtype=np.float32),
        active_mask=np.asarray([True, True, True, False]),
        terminated=np.zeros(n, dtype=bool),
        truncated=np.zeros(n, dtype=bool),
        metadata={"regime": "active_target", "event_ids": []},
    )


def test_local_sac_updates_shared_parameters_on_joint_batch() -> None:
    torch.manual_seed(20260804)
    replay = JointReplayBuffer(capacity=16, max_agents=4, seed=5)
    for value in range(8):
        add_transition(replay, float(value))
    batch = replay.sample(4)
    trainer = LocalSACTrainer(
        encoder_config=LocalEntityTokenEncoderConfig(hidden_dim=16, num_heads=4, num_layers=1),
        actor_config=RadialActorConfig(hidden_dim=16, a_max=0.8, decision_dt=0.5),
        config=LocalSACConfig(hidden_dim=16),
        device="cpu",
    )
    before = next(trainer.actor.parameters()).detach().clone()
    metrics = trainer.update(batch)
    after = next(trainer.actor.parameters()).detach()
    assert metrics["finite"] == 1.0
    assert metrics["active_count"] == 12.0
    assert np.isfinite(metrics["critic_loss"])
    assert not torch.equal(before, after)


def test_local_sac_checkpoint_strict_contract(tmp_path) -> None:
    trainer = LocalSACTrainer(
        encoder_config=LocalEntityTokenEncoderConfig(hidden_dim=16, num_heads=4, num_layers=1),
        actor_config=RadialActorConfig(hidden_dim=16),
        config=LocalSACConfig(hidden_dim=16),
        device="cpu",
    )
    contract = {"action_mode": "acceleration_2d_body", "a_max": 0.8, "v_max": 3.0, "critic_mode": "local"}
    path = tmp_path / "local_sac.pt"
    trainer.save_checkpoint(path, contract, runtime_state={"next_scene_index": 3})
    restored = LocalSACTrainer(
        encoder_config=LocalEntityTokenEncoderConfig(hidden_dim=16, num_heads=4, num_layers=1),
        actor_config=RadialActorConfig(hidden_dim=16),
        config=LocalSACConfig(hidden_dim=16),
        device="cpu",
    )
    restored.load_checkpoint(path, contract)
    assert restored.resume_runtime_state["next_scene_index"] == 3
    with pytest.raises(ValueError, match="contract mismatch"):
        restored.load_checkpoint(path, {**contract, "a_max": 1.6})


def test_sac_bootstraps_through_time_limit_but_not_true_terminal() -> None:
    terminated = torch.tensor([[False, True, False]])
    truncated = torch.tensor([[True, False, False]])
    mask = sac_bootstrap_mask(terminated, truncated)
    torch.testing.assert_close(mask, torch.tensor([[1.0, 0.0, 1.0]]))
