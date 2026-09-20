from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from cocap_voradj.models.shared_local_ac import (
    EXPECTED_AW9,
    SharedLocalACNetworkConfig,
    SharedLocalActor,
    SharedLocalQ,
    canonical_aw9_grid,
    parameter_count,
)
from cocap_voradj.training.shared_local_ac import (
    RecoveryInitPool,
    SharedLocalACConfig,
    SharedLocalACTrainer,
    SharedLocalReplay,
    Transition,
    atomic_torch_save,
    state_hash,
)


def observation(seed: int = 0, *, with_evader: bool = True) -> dict[str, np.ndarray]:
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


def transition(index: int, replay_class: str) -> Transition:
    obs = observation(index, with_evader=replay_class != "recovery_pure")
    return Transition(
        obs=obs,
        action=index % 9,
        reward=float(index % 3),
        next_obs=observation(index + 100, with_evader=replay_class != "recovery_pure"),
        done=False,
        terminated=False,
        truncated=False,
        phase="pre_capture",
        replay_class=replay_class,
        behavior_probability=1.0 / 9.0,
        epsilon=0.6,
        actor_probability=1.0 / 9.0,
        episode_id=index,
        agent_id=index % 4,
        metadata={},
    )


def test_aw9_mapping_and_shared_network_contract() -> None:
    assert np.allclose(canonical_aw9_grid(), np.asarray(EXPECTED_AW9), atol=1e-7)
    config = SharedLocalACNetworkConfig(hidden_dim=32, num_heads=4, num_layers=1)
    actor = SharedLocalActor(config).eval()
    critic = SharedLocalQ(config).eval()
    first = observation(1)
    batch = {key: torch.as_tensor(np.stack([value, value])) for key, value in first.items()}
    batch["masks"] = batch["masks"].bool()
    batch["types"] = batch["types"].long()
    logits = actor.logits(batch)
    q_values = critic(batch)
    assert logits.shape == (2, 9)
    assert q_values.shape == (2, 9)
    assert torch.allclose(logits[0], logits[1])
    assert torch.allclose(q_values[0], q_values[1])
    assert parameter_count(actor) > 0 and parameter_count(critic) > 0
    actor_ptrs = {id(param) for param in actor.parameters()}
    assert not actor_ptrs.intersection({id(param) for param in critic.parameters()})


def test_behavior_mixture_has_floor_and_normalizes() -> None:
    config = SharedLocalACNetworkConfig(hidden_dim=32, num_heads=4, num_layers=1)
    actor = SharedLocalActor(config).eval()
    obs = observation(2)
    batch = {key: torch.as_tensor(np.stack([obs, obs])) for key, obs in obs.items()}
    batch["masks"] = batch["masks"].bool()
    batch["types"] = batch["types"].long()
    epsilon = 0.6
    probs = actor.behavior_probabilities(batch, epsilon)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert float(probs.min()) >= epsilon / 9.0 - 1e-6


def test_shared_replay_exact_64_16_32_16() -> None:
    replay = SharedLocalReplay(1000)
    counts = {"pursuing": 64, "pre_capture_cover": 16, "post_capture_real": 32, "recovery_pure": 16}
    index = 0
    for replay_class, count in [(name, 64) for name in counts]:
        for _ in range(count):
            replay.add(transition(index, replay_class))
            index += 1
    batch, report = replay.sample_stratified(counts, np.random.RandomState(4), fallback="defer")
    assert len(batch) == 128
    assert report["sampler"] == "semantic_stratified"
    assert report["actual_counts"] == counts


def test_single_q_target_actor_update_and_atomic_resume() -> None:
    network = SharedLocalACNetworkConfig(hidden_dim=32, num_heads=4, num_layers=1)
    trainer = SharedLocalACTrainer(
        network,
        SharedLocalACConfig(batch_size=8, min_replay_size=8, train_freq=1),
        "cpu",
    )
    transitions = [transition(i, "pursuing") for i in range(8)]
    before = state_hash(trainer.actor)
    stats = trainer.update(transitions)
    assert stats["finite"] == 1.0
    assert state_hash(trainer.actor) != before
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "resume_latest.pt"
        payload = trainer.checkpoint_payload(config={}, global_step=1, episode=1, include_runtime=True)
        atomic_torch_save(payload, path)
        restored = SharedLocalACTrainer(network, trainer.config, "cpu")
        restored.load_payload(torch.load(path, map_location="cpu", weights_only=False))
        assert state_hash(restored.actor) == state_hash(trainer.actor)
        model_only = trainer.checkpoint_payload(config={}, global_step=1, episode=1)
        assert "replay" not in model_only
        assert "actor_optimizer" not in model_only


def test_recovery_pool_is_bounded_and_has_three_source_contracts() -> None:
    pool = RecoveryInitPool(capacity=1, captured_ratio=0.75, map_random_ratio=0.5)
    pool.add({"step": 3, "positions": [[1.0, 1.0]] * 4, "active_mask": [True] * 4})
    assert len(pool) == 1

    class FixedRandom:
        def __init__(self, values):
            self.values = iter(values)

        def random(self):
            return next(self.values)

        def choice(self, values):
            return values[0]

    source, snapshot = pool.choose(FixedRandom([0.1]))
    assert source == "capture_snapshot" and snapshot is not None
    source, snapshot = pool.choose(FixedRandom([0.99, 0.1]))
    assert source == "ordinary_map_random" and snapshot is None
    source, snapshot = pool.choose(FixedRandom([0.99, 0.99]))
    assert source == "synthetic_cluster" and snapshot is None
