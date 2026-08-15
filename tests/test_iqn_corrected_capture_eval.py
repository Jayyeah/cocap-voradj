from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tools.evaluate_iqn_corrected_capture as iqn_eval
from tools.evaluate_iqn_corrected_capture import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CHECKPOINT_SHA256,
    TRACE_SCHEMA,
    action_grid_from_config,
    capture_flags,
    checkpoint_sha256,
    current_apf_actions,
    episode_seeds,
    last50_trace_payload,
    map_action_index,
    near_capture_action_modes,
    ring_metrics,
    summarize,
    verify_checkpoint_hash,
)


def _snapshot(ring_count: int, distances: list[float]):
    return {
        "ring_count": ring_count,
        "pursuers": [
            {
                "id": index,
                "distance_to_enemy": distance,
                "in_ring_8_10_5": 8.0 <= distance < 10.5,
            }
            for index, distance in enumerate(distances)
        ],
        "angular": {
            "largest_angular_gap_rad": 2.5 if ring_count >= 2 else None,
            "minimum_pairwise_separation_rad": 0.8 if ring_count >= 2 else None,
        },
        "clearance": {
            "min_agent_agent_surface_clearance": 1.0,
            "min_obstacle_surface_clearance": 2.0,
            "min_boundary_center_clearance": 3.0,
        },
    }


def _action(index: int, *, in_ring: bool):
    grid = [
        (-0.4, -np.pi / 6), (-0.4, 0.0), (-0.4, np.pi / 6),
        (0.0, -np.pi / 6), (0.0, 0.0), (0.0, np.pi / 6),
        (0.4, -np.pi / 6), (0.4, 0.0), (0.4, np.pi / 6),
    ]
    a, omega = grid[index]
    return {
        "pursuer_id": index % 4,
        "action_index": index,
        "a": a,
        "omega": omega,
        "a_saturated": abs(a) == 0.4,
        "omega_saturated": abs(omega) > 0.5,
        "omega_sign_flip": False,
        "action_repeat": False,
        "pre_in_ring": in_ring,
    }


def test_historical_checkpoint_hash_constant_and_file_when_available() -> None:
    assert DEFAULT_CHECKPOINT_SHA256 == (
        "5a0ad1c1400d0004334669908c85db6f7b8496fb2987f36f992040c26bd0344d"
    )
    if DEFAULT_CHECKPOINT.exists():
        assert checkpoint_sha256(DEFAULT_CHECKPOINT) == DEFAULT_CHECKPOINT_SHA256


def test_checkpoint_hash_verification_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"known checkpoint bytes")
    expected = hashlib.sha256(b"known checkpoint bytes").hexdigest()
    assert verify_checkpoint_hash(checkpoint, expected) == expected
    with pytest.raises(ValueError, match="mismatch"):
        verify_checkpoint_hash(checkpoint, "0" * 64)
    with pytest.raises(ValueError, match="64 lowercase hex"):
        verify_checkpoint_hash(checkpoint, "not-a-sha")


def test_exact_nine_grid_index_to_aw_mapping() -> None:
    config = {
        "pursuer": {
            "a": [-0.4, 0.0, 0.4],
            "w": [-np.pi / 6, 0.0, np.pi / 6],
        }
    }
    grid = action_grid_from_config(config)
    assert grid == [
        (-0.4, -np.pi / 6), (-0.4, 0.0), (-0.4, np.pi / 6),
        (0.0, -np.pi / 6), (0.0, 0.0), (0.0, np.pi / 6),
        (0.4, -np.pi / 6), (0.4, 0.0), (0.4, np.pi / 6),
    ]
    for index, expected in enumerate(grid):
        np.testing.assert_allclose(map_action_index(index, grid), expected, atol=1e-7)
    with pytest.raises(ValueError, match="outside"):
        map_action_index(9, grid)


def test_episode_seeds_have_no_legacy_plus_43_offset() -> None:
    assert episode_seeds(2026081201, 4) == [
        2026081201, 2026081202, 2026081203, 2026081204,
    ]


def test_current_apf_path_is_mandatory_and_fail_closed(monkeypatch) -> None:
    inactive_contract = SimpleNamespace(
        config={"evader": {"autonomous": False}}, evaders=[]
    )
    with pytest.raises(ValueError, match="autonomous=true"):
        current_apf_actions(inactive_contract, [])

    active_evader = SimpleNamespace(deactivated=False)
    env = SimpleNamespace(
        config={"evader": {"autonomous": True}}, evaders=[active_evader]
    )
    calls = []

    def fake_current_helper(received_env, received_agents):
        calls.append((received_env, received_agents))
        return [7]

    monkeypatch.setattr(iqn_eval, "_evader_actions_for_env", fake_current_helper)
    assert current_apf_actions(env, [SimpleNamespace()]) == [7]
    assert calls and calls[0][0] is env


def test_capture_type_and_collision_summary_remain_separate() -> None:
    assert capture_flags(["loose"], {"captured": True}) == {
        "normal_capture": True,
        "stationary_capture": False,
        "captured": True,
    }
    assert capture_flags(["stationary"], {"captured": True}) == {
        "normal_capture": False,
        "stationary_capture": True,
        "captured": True,
    }
    with pytest.raises(RuntimeError, match="mismatch"):
        capture_flags([], {"captured": True})
    result = summarize([
        {
            "normal_capture": True, "stationary_capture": False, "captured": True,
            "collision": False, "visited_2plus_ring": True, "visited_3plus_ring": True,
            "fraction_steps_2plus_in_ring": 0.2, "fraction_steps_3plus_in_ring": 0.1,
            "max_3plus_ring_hold_steps": 2, "length": 30,
            "collision_type_counts": {}, "action_modes_all_steps": {},
            "near_capture_action_modes": {},
        },
        {
            "normal_capture": False, "stationary_capture": True, "captured": True,
            "collision": True, "visited_2plus_ring": True, "visited_3plus_ring": False,
            "fraction_steps_2plus_in_ring": 0.1, "fraction_steps_3plus_in_ring": 0.0,
            "max_3plus_ring_hold_steps": 0, "length": 20,
            "collision_type_counts": {"agent_agent": 1},
            "action_modes_all_steps": {}, "near_capture_action_modes": {},
        },
    ])
    assert result["normal_capture_count"] == 1
    assert result["stationary_capture_count"] == 1
    assert result["collision_count"] == 1
    assert result["collision_type_counts"] == {"agent_agent": 1}


def test_ring_two_three_fraction_and_hold_metrics() -> None:
    result = ring_metrics([0, 2, 2, 3, 3, 1])
    assert result["max_num_in_ring"] == 3
    assert result["visited_2plus_ring"] is True
    assert result["visited_3plus_ring"] is True
    assert result["max_2plus_ring_hold_steps"] == 4
    assert result["max_3plus_ring_hold_steps"] == 2
    assert np.isclose(result["fraction_steps_2plus_in_ring"], 4 / 6)


def test_near_capture_action_histograms_split_ring_members() -> None:
    far = {
        "ring_count_pre": 0,
        "ring_count_post_was_active": 0,
        "pre": _snapshot(0, [20.0, 30.0, 40.0, 50.0]),
        "post_was_active": _snapshot(0, [19.0, 29.0, 39.0, 49.0]),
        "actions": [_action(4, in_ring=False)],
    }
    near = {
        "ring_count_pre": 2,
        "ring_count_post_was_active": 2,
        "pre": _snapshot(2, [9.0, 10.0, 11.0, 20.0]),
        "post_was_active": _snapshot(2, [9.0, 10.0, 10.4, 19.0]),
        "actions": [_action(6, in_ring=True), _action(8, in_ring=False)],
    }
    result = near_capture_action_modes([far, near])
    assert result["near_capture_steps"] == 1
    assert result["ring_2plus_steps"] == 1
    assert result["d3_le_10_5_steps"] == 1
    assert result["all"]["action_index_histogram"]["6"] == 1
    assert result["all"]["action_index_histogram"]["8"] == 1
    assert result["pre_ring_members"]["agent_steps"] == 1
    assert result["pre_non_ring_agents"]["agent_steps"] == 1


def test_last50_trace_is_a4_summary_compatible() -> None:
    frames = []
    for step in range(1, 56):
        snapshot = _snapshot(2 if step < 55 else 3, [9.0, 10.0, 10.4, 20.0])
        snapshot["pursuers"] = [{
            **row,
            "radial_velocity_outward": -0.2,
            "tangential_velocity_ccw": 0.3,
        } for row in snapshot["pursuers"]]
        frames.append({
            "episode_step": step,
            "ring_count_post_alive": snapshot["ring_count"],
            "post": snapshot,
            "actions": [_action(7, in_ring=True)],
            "collision_events": [],
        })
    payload = last50_trace_payload(frames, requested_tail_steps=50)
    assert payload["schema"] == TRACE_SCHEMA
    assert len(payload["frames"]) == 50
    assert payload["frames"][0]["episode_step"] == 6
    assert payload["trace_summary"]["steps_saved"] == 50
    assert payload["trace_summary"]["max_num_in_ring"] == 3

