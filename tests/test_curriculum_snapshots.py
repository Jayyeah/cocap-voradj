from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.continuous.curriculum_snapshots import build_snapshot, restore_snapshot
from cocap_voradj.training.continuous.formal_config import resolve_formal_config, scene_config
from cocap_voradj.training.trainer import set_global_config


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "configs/experiments/continuous_marl_20260804/p6_formal_central_masac_4v1.yaml"


def make_env(scene: str = "capture", seed: int = 7) -> VorAdjEnv:
    config = scene_config(resolve_formal_config(FORMAL), scene)
    set_global_config(config)
    env = VorAdjEnv(config, seed=seed)
    env.reset()
    return env


def test_snapshot_geometry_reset_restores_positions_active_obstacles_and_phase() -> None:
    env = make_env("capture")
    for i, pursuer in enumerate(env.pursuers):
        pursuer.x = 20.0 + i
        pursuer.y = 30.0 + i
        pursuer.velocity = np.asarray([0.5, -0.2])
        pursuer.speed = float(np.linalg.norm(pursuer.velocity))
    snapshot = build_snapshot(
        env,
        scene="capture",
        phase="pre_capture",
        step=42,
        source={"branch": "main", "commit": "test", "seed": 1},
    )
    payload = json.dumps(snapshot)
    restored, diagnostics = restore_snapshot(
        json.loads(payload),
        scene_config(resolve_formal_config(FORMAL), "capture"),
        seed=8,
        mode="geometry_reset",
    )
    for original, new in zip(env.pursuers, restored.pursuers):
        assert np.isclose(new.x, original.x, atol=1e-6)
        assert np.isclose(new.y, original.y, atol=1e-6)
        assert np.linalg.norm(new.velocity) == pytest.approx(0.0, abs=1e-6)
    assert len(restored.obstacles) == len(env.obstacles)
    assert restored.last_task_labels
    assert restored.last_reward_terms == {}


def test_snapshot_full_state_restores_velocity_and_theta() -> None:
    env = make_env("pure_ce")
    for pursuer in env.pursuers:
        pursuer.velocity = np.asarray([1.0, -0.5])
        pursuer.speed = float(np.linalg.norm(pursuer.velocity))
        pursuer.theta = 0.7
    snapshot = build_snapshot(env, scene="pure_ce", phase="pure_recovery", step=1, source={"commit": "test"})
    restored, _ = restore_snapshot(
        snapshot,
        scene_config(resolve_formal_config(FORMAL), "pure_ce"),
        seed=9,
        mode="full_state",
    )
    for original, new in zip(env.pursuers, restored.pursuers):
        assert np.allclose(new.velocity, original.velocity, atol=1e-6)
        assert np.isclose(new.speed, original.speed, atol=1e-6)
        assert np.isclose(new.theta, original.theta, atol=1e-6)


def test_snapshot_schema_version_mismatch_rejected() -> None:
    env = make_env("pure_ce")
    snapshot = build_snapshot(env, scene="pure_ce", phase="pure_recovery", step=0, source={})
    snapshot["schema_version"] = 999
    with pytest.raises(ValueError, match="schema version"):
        restore_snapshot(snapshot, scene_config(resolve_formal_config(FORMAL), "pure_ce"), seed=1)
