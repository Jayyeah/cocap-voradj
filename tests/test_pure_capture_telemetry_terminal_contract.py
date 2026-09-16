from __future__ import annotations

import copy

import pytest

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.forward_final_v2 import new_pure_capture_config
from cocap_voradj.training.trainer import set_global_config
from tools.forward_final_single_task_20260915 import Telemetry


def _env(*, horizon: int = 3000) -> VorAdjEnv:
    config = new_pure_capture_config()
    config["env"]["episode_max_length"] = horizon
    config["env"]["pre_capture_max_length"] = horizon
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=2026091511)
    env.reset()
    return env


def _capture_event(capture_type: str) -> dict:
    return {
        "evader_id": 0,
        "participants": [0, 1, 2],
        "angles": [0.0, 2.0, 4.0],
        "capture_type": capture_type,
    }


@pytest.mark.parametrize("capture_type", ["loose", "stationary"])
def test_capture_terminal_is_accepted_and_has_no_following_transition(
    monkeypatch: pytest.MonkeyPatch,
    capture_type: str,
) -> None:
    env = _env()
    telemetry = Telemetry(env, "capture")
    monkeypatch.setattr(env, "_loose_capture_events", lambda: [_capture_event(capture_type)])

    result = env.step([4] * 4, [4])
    telemetry.observe(env, result, [0, 1, 2, 3])

    assert all(result.dones)
    assert all(info["terminated"] and not info["truncated"] for info in result.infos)
    assert telemetry.terminal_transition_seen is True
    assert telemetry.finish(env)["post_capture_transitions"] == 0
    with pytest.raises(AssertionError, match="after the episode terminal"):
        telemetry.observe(env, result, [0, 1, 2, 3])


def test_no_capture_transition_stays_pre_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _env()
    telemetry = Telemetry(env, "capture")
    monkeypatch.setattr(env, "_loose_capture_events", lambda: [])

    result = env.step([4] * 4, [4])
    telemetry.observe(env, result, [0, 1, 2, 3])

    assert not all(result.dones)
    assert all(info["replay_metadata"]["phase"] == "pre_capture" for info in result.infos)
    assert telemetry.terminal_transition_seen is False


def test_timeout_is_truncated_without_post_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _env(horizon=1)
    telemetry = Telemetry(env, "capture")
    monkeypatch.setattr(env, "_loose_capture_events", lambda: [])

    result = env.step([4] * 4, [4])
    telemetry.observe(env, result, [0, 1, 2, 3])

    assert all(result.dones)
    assert all(info["truncated"] and not info["terminated"] for info in result.infos)
    assert all(info["replay_metadata"]["phase"] == "pre_capture" for info in result.infos)
    assert telemetry.terminal_transition_seen is True


def test_post_capture_metadata_is_only_allowed_on_terminal_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _env()
    telemetry = Telemetry(env, "capture")
    monkeypatch.setattr(env, "_loose_capture_events", lambda: [])
    result = env.step([4] * 4, [4])
    for info in result.infos:
        info["replay_metadata"]["phase"] = "post_capture"

    with pytest.raises(AssertionError):
        telemetry.observe(env, result, [0, 1, 2, 3])


def test_terminal_enemy_loss_post_capture_metadata_is_not_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _env()
    telemetry = Telemetry(env, "capture")
    monkeypatch.setattr(env, "_loose_capture_events", lambda: [])
    env.evaders[0].deactivated = True
    env.evaders[0].collision = True

    result = env.step([4] * 4, [None])
    telemetry.observe(env, result, [0, 1, 2, 3])

    assert all(result.dones)
    assert all(info["terminated"] and not info["truncated"] for info in result.infos)
    assert all(info["replay_metadata"]["phase"] == "post_capture" for info in result.infos)
    assert telemetry.post_capture_terminal_metadata_transitions == 1
    assert telemetry.finish(env)["post_capture_transitions"] == 0


def test_legacy_checkpoint_telemetry_without_new_fields_is_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _env()
    telemetry = Telemetry(env, "capture")
    del telemetry.terminal_transition_seen
    del telemetry.post_capture_terminal_metadata_transitions
    monkeypatch.setattr(env, "_loose_capture_events", lambda: [])

    result = env.step([4] * 4, [4])
    telemetry.observe(env, result, [0, 1, 2, 3])

    assert telemetry.finish(env)["post_capture_terminal_metadata_transitions"] == 0
