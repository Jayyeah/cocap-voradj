from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.trainer import set_global_config
from tools import evaluate_iqn_z05_independent_20260923 as formal
from tools.run_iqn_repeated_arrival_demo_20261006 import (
    choose_worker_count, spawn_target_generation, stable_hash,
    next_spawn_boundary, planned_spawn_boundary,
)


CONFIG = ROOT / "configs/experiments/iqn_z_unified_decay_curriculum_20260919/z05_stage3_12p3e3obs_700k.yaml"


def _world():
    cfg = formal.formal.resolved(CONFIG)
    cfg = formal.configure_large_case(cfg, 12, 3, 3, 700)
    cfg = formal.scene_config(cfg, "mixed")
    set_global_config(cfg)
    env = VorAdjEnv(copy.deepcopy(cfg), seed=2026100601)
    observations = list(env.reset())
    return env, observations


def test_arrival_delay_uses_next_decision_boundary():
    assert next_spawn_boundary(100, 0) == 101
    assert next_spawn_boundary(100, 1) == 102
    assert next_spawn_boundary(100, 10) == 111
    assert planned_spawn_boundary("PERSIST-A", None, 100, 0) is None
    assert planned_spawn_boundary("PERSIST-A", 120, 100, 0) == 121
    assert planned_spawn_boundary("PERSIST-B", None, 100, 0) == 101
    assert planned_spawn_boundary("PERSIST-C", None, 100, 10) == 111


def test_random_delay_uses_one_paired_stream_per_episode():
    seed = 2026800601 ^ 0x5A05
    stream = np.random.default_rng(seed)
    delays = [int(stream.integers(0, 11)) for _ in range(3)]
    reset_each_time = [int(np.random.default_rng(seed).integers(0, 11)) for _ in range(3)]
    assert all(0 <= value <= 10 for value in delays)
    assert delays != reset_each_time


def test_resource_aware_worker_count_is_bounded_and_single_threaded():
    automatic = choose_worker_count()
    assert 1 <= automatic["selected_workers"] <= automatic["safe_worker_cap"] <= 4
    assert automatic["thread_limit_per_worker"] == 1
    assert choose_worker_count(2)["selected_workers"] <= 2


def test_target_generation_reuses_slots_and_preserves_persistent_state():
    env, _ = _world()
    for target in env.evaders:
        target.deactivated = True
        target.collision = False
        target.wave_id = 1
        target.target_generation = 1
    env.pursuers[0].deactivated = True
    positions = [[float(p.x), float(p.y)] for p in env.pursuers]
    velocities = [[float(v) for v in p.velocity] for p in env.pursuers]
    active_mask = [not p.deactivated for p in env.pursuers]
    obstacles = [[float(o.x), float(o.y), float(o.r)] for o in env.obstacles]
    z = np.linspace(0.0, 0.9, len(env.pursuers), dtype=np.float32)
    env.z_state = z.copy()
    env._z_update_count = 47
    env.post_capture_started = True
    env.post_capture_step = 44
    env.post_capture_coverage_success = True
    env.post_capture_coverage_step = 31
    env._stationary_capture_counters = {"stale": 9}

    row = spawn_target_generation(env, np.random.default_rng(2026800601), wave_id=2, generation=2)
    observations = env.get_observations()

    assert row["target_slots"] == [0, 1, 2]
    assert row["wave_id"] == 2 and row["target_generation"] == 2
    assert [not p.deactivated for p in env.pursuers] == active_mask
    assert [[float(p.x), float(p.y)] for p in env.pursuers] == positions
    assert [[float(v) for v in p.velocity] for p in env.pursuers] == velocities
    assert stable_hash([[float(o.x), float(o.y), float(o.r)] for o in env.obstacles]) == stable_hash(obstacles)
    np.testing.assert_array_equal(env.z_state, z)
    assert env._z_update_count == 47
    assert env.post_capture_step == 0
    assert not env.post_capture_started
    assert not env.post_capture_coverage_success
    assert env.post_capture_coverage_step is None
    assert env._stationary_capture_counters == {}
    assert all(not target.deactivated and not target.collision for target in env.evaders)
    assert all(target.wave_id == 2 and target.target_generation == 2 for target in env.evaders)
    assert sum(not target.deactivated for target in env.evaders) == 3
    assert observations[0] is None and all(obs is not None for obs in observations[1:])

    # Reused targets satisfy the canonical obstacle, target-target, and
    # pursuer-target clearances after the sampler's documented relax fallback.
    obstacle_rows = [(np.array([x, y]), r) for x, y, r in obstacles]
    min_sep = float(env.env_cfg.get("evader_spawn_min_sep", 8.0))
    min_pe = float(env.env_cfg.get("min_pursuer_evader_init_dis", 13.0))
    for index, target in enumerate(env.evaders):
        pos = np.array([target.x, target.y], dtype=float)
        for other_pos, other_r in obstacle_rows:
            assert np.linalg.norm(pos - other_pos) >= target.r + other_r + min_sep
        for pursuer in env.pursuers:
            if not pursuer.deactivated:
                assert np.linalg.norm(pos - np.array([pursuer.x, pursuer.y])) >= min_pe
        for other in env.evaders[index + 1:]:
            assert np.linalg.norm(pos - np.array([other.x, other.y])) >= target.r + other.r + min_sep
