from __future__ import annotations

import numpy as np
import pytest

from cocap_voradj.training.continuous.joint_replay import JointReplayBuffer, JointReplaySampler


def add_transition(replay: JointReplayBuffer, index: int, regime: str = "uniform", event_ids=None, coverage_only: bool = False) -> int:
    obs = {
        "self": np.full((replay.max_agents, 9), index, dtype=np.float32),
        "pursuers": np.full((replay.max_agents, 8, 7), index, dtype=np.float32),
        "evaders": np.full((replay.max_agents, 8, 7), index, dtype=np.float32),
        "obstacles": np.full((replay.max_agents, 5, 5), index, dtype=np.float32),
        "masks": np.ones((replay.max_agents, 22), dtype=np.float32),
        "types": np.ones((replay.max_agents, 22), dtype=np.float32),
    }
    return replay.add(
        local_obs=obs,
        next_local_obs={key: value + 1.0 for key, value in obs.items()},
        global_state=np.full((replay.max_agents, 4), index, dtype=np.float32),
        next_global_state=np.full((replay.max_agents, 4), index + 1, dtype=np.float32),
        actions=np.full((replay.max_agents, 2), 0.1, dtype=np.float32),
        rewards=np.full(replay.max_agents, 1.0, dtype=np.float32),
        active_mask=np.ones(replay.max_agents, dtype=bool),
        terminated=np.zeros(replay.max_agents, dtype=bool),
        truncated=np.zeros(replay.max_agents, dtype=bool),
        metadata={
            "regime": regime,
            "event_ids": event_ids or [],
            "coverage_only": coverage_only,
            "phase": "pure_coverage" if coverage_only else "pre_capture",
        },
    )


def test_joint_schema_stacks_acceleration_actions_and_masks() -> None:
    replay = JointReplayBuffer(capacity=4, max_agents=4, seed=1)
    transition_id = add_transition(replay, 0, regime="active_target", event_ids=["discovery"])
    batch = replay.sample(batch_size=1)
    assert batch["transition_ids"].shape == (1,)
    assert batch["actions"].shape == (1, 4, 2)
    assert batch["rewards"].shape == (1, 4)
    assert batch["active_mask"].shape == (1, 4)
    assert batch["metadata"][0]["event_ids"] == ["discovery"]
    assert "v_raw" not in batch
    assert replay.get(transition_id).transition_id == transition_id


def test_ring_overwrite_removes_stale_event_indices_atomically() -> None:
    replay = JointReplayBuffer(capacity=2, max_agents=4, seed=2)
    first = add_transition(replay, 0, regime="active_target", event_ids=["old"])
    add_transition(replay, 1, regime="coverage_only", coverage_only=True)
    third = add_transition(replay, 2, regime="active_target", event_ids=["new"])
    assert len(replay) == 2
    assert first not in replay.transition_ids
    assert replay._pool("event", "old") == []
    assert replay._pool("event", "new") == [third]


def test_sampler_records_ratios_and_enforces_coverage_floor() -> None:
    replay = JointReplayBuffer(capacity=128, max_agents=4, seed=3)
    for index in range(80):
        add_transition(replay, index, regime="uniform")
    for index in range(20):
        add_transition(replay, 100 + index, regime="active_target", event_ids=["capture"])
    for index in range(28):
        add_transition(replay, 200 + index, regime="coverage_only", coverage_only=True)
    sampler = JointReplaySampler()
    batch = replay.sample(64, sampler=sampler)
    stats = batch["sampling_stats"]
    assert len(batch["transition_ids"]) == 64
    assert stats["coverage_only_count"] >= 16
    assert stats["event_count"] > 0
    assert sum(stats["source_slot_counts"].values()) == 64
    assert stats["actual"] == 64


def test_sampler_records_empty_event_pool_fallback() -> None:
    replay = JointReplayBuffer(capacity=16, max_agents=4, seed=8)
    for index in range(8):
        add_transition(replay, index, regime="coverage_only", coverage_only=True)
    batch = replay.sample(10, sampler=JointReplaySampler())
    stats = batch["sampling_stats"]
    assert stats["event_pool_available"] is False
    assert stats["event_fallback_count"] == 1
    assert stats["source_slot_counts"]["fallback_uniform"] == 1
    assert stats["actual"] == 10


def test_joint_replay_strict_manifest_resume(tmp_path) -> None:
    replay = JointReplayBuffer(capacity=4, max_agents=4, seed=4)
    add_transition(replay, 0)
    path = tmp_path / "joint_replay.pkl"
    manifest = {"action_mode": "acceleration_2d_body", "v_max": 3.0, "a_max": 0.8}
    replay.save(path, manifest, runtime_state={"next_scene_index": 2})
    expected_ids = replay.sample_ids(3)
    restored = JointReplayBuffer.load(path, manifest)
    assert restored.transition_ids == replay.transition_ids
    assert restored.runtime_state["next_scene_index"] == 2
    assert restored.sample_ids(3) == expected_ids
    with pytest.raises(ValueError, match="manifest mismatch"):
        JointReplayBuffer.load(path, {"action_mode": "acceleration_2d_body", "v_max": 3.0, "a_max": 1.6})


def test_joint_replay_rejects_candidate_velocity_metadata() -> None:
    replay = JointReplayBuffer(capacity=2, max_agents=4)
    with pytest.raises(ValueError, match="action candidates"):
        add_transition(replay, 0, regime="uniform", event_ids=[])
        replay.add(
            local_obs={"self": np.zeros((4, 9), dtype=np.float32)},
            next_local_obs={"self": np.zeros((4, 9), dtype=np.float32)},
            global_state=np.zeros((4, 4), dtype=np.float32),
            next_global_state=np.zeros((4, 4), dtype=np.float32),
        actions=np.zeros((4, 2), dtype=np.float32),
            rewards=np.zeros(4, dtype=np.float32),
            active_mask=np.ones(4, dtype=bool),
            terminated=np.zeros(4, dtype=bool),
            truncated=np.zeros(4, dtype=bool),
            metadata={"v_raw": [[0.0, 0.0]]},
        )
