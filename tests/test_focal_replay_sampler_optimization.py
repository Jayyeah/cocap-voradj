from __future__ import annotations

from pathlib import Path

import numpy as np

from cocap_voradj.training.continuous.joint_replay import (
    FocalReplaySampler,
    JointReplayBuffer,
)


def _add_pursuing(replay: JointReplayBuffer, value: float) -> None:
    obs = {"self": np.asarray([[value]], dtype=np.float32)}
    replay.add(
        local_obs=obs,
        next_local_obs={"self": obs["self"] + 1.0},
        global_state=np.asarray([[value]], dtype=np.float32),
        next_global_state=np.asarray([[value + 1.0]], dtype=np.float32),
        actions=np.zeros((1, 2), dtype=np.float32),
        rewards=np.zeros(1, dtype=np.float32),
        active_mask=np.ones(1, dtype=bool),
        terminated=np.zeros(1, dtype=bool),
        truncated=np.zeros(1, dtype=bool),
        metadata={
            "phase": "pre_capture",
            "scene": "capture",
            "origin": "test",
            "event_ids": [],
        },
        agent_role_id=np.ones(1, dtype=np.uint8),
    )


def test_dense_incremental_role_pool_sparse_sampling_is_uniform_and_valid() -> None:
    replay = JointReplayBuffer(capacity=5000, max_agents=1, seed=1)
    for index in range(5000):
        _add_pursuing(replay, float(index))
    assert replay.focal_index_sizes["pre_capture_pursuing"] == 5000

    sampler = FocalReplaySampler({"pre_capture_pursuing": 64}, seed=2)
    quartiles = np.zeros(4, dtype=np.int64)
    total = 0
    for _ in range(200):
        items = sampler.sample_items(replay)
        pairs = [(item.slot_id, item.agent_id) for item in items]
        assert len(items) == 64
        assert len(pairs) == len(set(pairs))
        for item in items:
            transition = replay.get_item_transition(item)
            quartiles[min(3, transition.transition_id // 1250)] += 1
            total += 1

    expected = total / 4.0
    assert np.all(np.abs(quartiles - expected) < 0.08 * expected)

    # Ring overwrite uses swap-delete in the dense pool and must not leave a
    # stale generation reference behind.
    for index in range(10):
        _add_pursuing(replay, float(5000 + index))
    assert len(replay) == 5000
    assert replay.focal_index_sizes["pre_capture_pursuing"] == 5000
    for item in sampler.sample_items(replay):
        replay.get_item_transition(item)


def test_focal_sampler_rng_state_round_trip_reproduces_next_batch() -> None:
    replay = JointReplayBuffer(capacity=5000, max_agents=1, seed=3)
    for index in range(5000):
        _add_pursuing(replay, float(index))
    sampler = FocalReplaySampler({"pre_capture_pursuing": 64}, seed=4)
    sampler.sample_items(replay)
    state = sampler.state_dict()
    expected = [
        (item.slot_id, item.generation_id, item.agent_id, item.bucket_id)
        for item in sampler.sample_items(replay)
    ]

    restored = FocalReplaySampler({"pre_capture_pursuing": 64}, seed=999)
    restored.load_state_dict(state)
    actual = [
        (item.slot_id, item.generation_id, item.agent_id, item.bucket_id)
        for item in restored.sample_items(replay)
    ]
    assert actual == expected


def test_replay_reload_preserves_dense_pool_order_after_ring_overwrite(tmp_path: Path) -> None:
    replay = JointReplayBuffer(capacity=5000, max_agents=1, seed=5)
    for index in range(5017):
        _add_pursuing(replay, float(index))
    sampler = FocalReplaySampler({"pre_capture_pursuing": 64}, seed=6)
    sampler.sample_items(replay)
    sampler_state = sampler.state_dict()
    expected = [
        (item.slot_id, item.generation_id, item.agent_id, item.bucket_id)
        for item in sampler.sample_items(replay)
    ]

    path = tmp_path / "replay.pkl"
    manifest = {"test": "dense-pool-order"}
    replay.save(path, manifest)
    restored_replay = JointReplayBuffer.load(path, manifest)
    restored_sampler = FocalReplaySampler({"pre_capture_pursuing": 64}, seed=999)
    restored_sampler.load_state_dict(sampler_state)
    actual = [
        (item.slot_id, item.generation_id, item.agent_id, item.bucket_id)
        for item in restored_sampler.sample_items(restored_replay)
    ]

    assert actual == expected
