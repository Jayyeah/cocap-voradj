from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cocap_voradj.training.continuous.formal_config import resolve_formal_config
from cocap_voradj.training.continuous.joint_replay import JointReplayBuffer
from tools.run_continuous_ctde_training import (
    _make_trainer,
    _save_checkpoint_bundle,
    _verify_resume_steps,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "configs/experiments/continuous_marl_20260804/p6_formal_central_masac_4v1.yaml"


def make_replay(max_agents: int = 12) -> JointReplayBuffer:
    replay = JointReplayBuffer(capacity=8, max_agents=max_agents, seed=1)
    roles = np.zeros(max_agents, dtype=np.uint8)
    roles[0] = 1
    replay.add(
        local_obs={
            "self": np.zeros((max_agents, 9), dtype=np.float32),
            "pursuers": np.zeros((max_agents, 12, 7), dtype=np.float32),
            "evaders": np.zeros((max_agents, 8, 7), dtype=np.float32),
            "obstacles": np.zeros((max_agents, 5, 5), dtype=np.float32),
            "masks": np.ones((max_agents, 26), dtype=bool),
            "types": np.zeros((max_agents, 26), dtype=np.int64),
        },
        next_local_obs={
            "self": np.zeros((max_agents, 9), dtype=np.float32),
            "pursuers": np.zeros((max_agents, 12, 7), dtype=np.float32),
            "evaders": np.zeros((max_agents, 8, 7), dtype=np.float32),
            "obstacles": np.zeros((max_agents, 5, 5), dtype=np.float32),
            "masks": np.ones((max_agents, 26), dtype=bool),
            "types": np.zeros((max_agents, 26), dtype=np.int64),
        },
        global_state={
            "self": np.zeros((max_agents, 9), dtype=np.float32),
            "pursuers": np.zeros((max_agents, max_agents, 7), dtype=np.float32),
            "evaders": np.zeros((8, 7), dtype=np.float32),
            "obstacles": np.zeros((5, 5), dtype=np.float32),
            "pursuer_mask": np.zeros((max_agents, max_agents), dtype=bool),
            "evader_mask": np.zeros(8, dtype=bool),
            "obstacle_mask": np.zeros(5, dtype=bool),
            "active_mask": roles > 0,
        },
        next_global_state={
            "self": np.zeros((max_agents, 9), dtype=np.float32),
            "pursuers": np.zeros((max_agents, max_agents, 7), dtype=np.float32),
            "evaders": np.zeros((8, 7), dtype=np.float32),
            "obstacles": np.zeros((5, 5), dtype=np.float32),
            "pursuer_mask": np.zeros((max_agents, max_agents), dtype=bool),
            "evader_mask": np.zeros(8, dtype=bool),
            "obstacle_mask": np.zeros(5, dtype=bool),
            "active_mask": roles > 0,
        },
        actions=np.zeros((max_agents, 2), dtype=np.float32),
        rewards=np.ones(max_agents, dtype=np.float32),
        active_mask=roles > 0,
        terminated=np.zeros(max_agents, dtype=bool),
        truncated=np.zeros(max_agents, dtype=bool),
        metadata={"phase": "pre_capture", "scene": "capture", "origin": "map_random", "event_ids": []},
        agent_role_id=roles,
    )
    return replay


def test_25k_bundle_contains_all_required_files(tmp_path: Path) -> None:
    config = resolve_formal_config(FORMAL)
    trainer = _make_trainer(config, "cpu")
    replay = make_replay()
    manifest = {"action_mode": "acceleration_2d_world", "max_agents": 12, "step": 25000}
    runtime = {
        "transition_count": 25000,
        "update_count": 3,
        "next_scene_index": 5,
        "recovery_pool": [],
        "metrics_history": [{"step": 1000}],
    }
    bundle = _save_checkpoint_bundle(
        Path(tmp_path),
        25000,
        config,
        manifest,
        trainer,
        replay,
        runtime,
        [{"step": 1000}],
        {"capture": {"episodes": 0}},
    )
    assert bundle.is_dir()
    for name in (
        "trainer.pt",
        "replay.pkl",
        "runtime_state.pkl",
        "effective_config.yaml",
        "manifest.json",
        "metrics.jsonl",
        "diagnostic_eval.json",
    ):
        assert (bundle / name).is_file(), name
    assert "step_000025000" in bundle.name


def test_bundle_failure_does_not_destroy_previous_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = resolve_formal_config(FORMAL)
    trainer = _make_trainer(config, "cpu")
    replay = make_replay()
    manifest = {"step": 0}
    runtime = {"transition_count": 0}
    first = _save_checkpoint_bundle(
        Path(tmp_path), 0, config, manifest, trainer, replay, runtime, [], {}
    )
    before = (first / "trainer.pt").read_bytes()

    def fail_save(*args, **kwargs):
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr(trainer, "save_checkpoint", fail_save)
    with pytest.raises(RuntimeError, match="simulated save failure"):
        _save_checkpoint_bundle(
            Path(tmp_path), 25000, config, manifest, trainer, replay, runtime, [], {}
        )
    assert (first / "trainer.pt").read_bytes() == before
    assert not (Path(tmp_path) / "checkpoints" / "step_000025000").exists()


def test_resume_step_mismatch_rejected() -> None:
    trainer = _make_trainer(resolve_formal_config(FORMAL), "cpu")
    replay = make_replay()
    trainer.resume_runtime_state = {"transition_count": 25000}
    replay.runtime_state = {"transition_count": 24000}
    with pytest.raises(ValueError, match="step mismatch"):
        _verify_resume_steps(trainer, replay)
    replay.runtime_state = {"transition_count": 25000}
    _verify_resume_steps(trainer, replay)


def test_diagnostic_rollout_cap_is_400() -> None:
    config = resolve_formal_config(FORMAL)
    assert config["evaluation"]["diagnostic_rollout_cap"] == 400
    assert config["training"]["checkpoint_interval_env_steps"] == 25000
    assert config["training"]["metrics_flush_interval_env_steps"] == 1000
    assert config["training"]["diagnostic_eval_interval_env_steps"] == 25000
