from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.continuous.formal_config import (
    resolve_ladder_config,
    scene_config,
)
from cocap_voradj.training.trainer import set_global_config


ROOT = Path(__file__).resolve().parents[1]
CF3 = (
    ROOT
    / "configs/experiments/parallel_ce_legacy_voradj_20260809"
    / "legacy_voradj_cf3_capture_first_local_support_full_4p1e1obs_100k_aw.yaml"
)


def _env(semantics: str = "synchronized_swept_v1") -> VorAdjEnv:
    config = scene_config(resolve_ladder_config(CF3), "capture")
    config = copy.deepcopy(config)
    config.setdefault("env", {})["collision_semantics"] = semantics
    set_global_config(config)
    env = VorAdjEnv(config, seed=2026081501)
    env.reset()
    positions = [(15.0, 15.0), (35.0, 15.0), (15.0, 35.0), (35.0, 35.0)]
    for pursuer, position in zip(env.pursuers, positions):
        pursuer.x, pursuer.y = position
        pursuer.speed = 0.0
        pursuer.velocity = np.zeros(2, dtype=float)
        pursuer.deactivated = False
        pursuer.collision = False
        pursuer.collision_types = set()
    env.evaders[0].x, env.evaders[0].y = 90.0, 90.0
    env.evaders[0].speed = 0.0
    env.evaders[0].velocity = np.zeros(2, dtype=float)
    env.evaders[0].deactivated = False
    env.evaders[0].collision = False
    env.evaders[0].collision_types = set()
    env.obstacles[0].x, env.obstacles[0].y = 80.0, 20.0
    env.last_collision_events = []
    env.episode_collision_events = []
    env.episode_collision_type_counts = {}
    return env


def _resolve_without_motion(env: VorAdjEnv) -> None:
    env._begin_collision_step()
    try:
        env._refresh_collisions()
    finally:
        env._end_collision_step()


def test_cf3_substep_contract_now_selects_synchronized_swept_default() -> None:
    config = scene_config(resolve_ladder_config(CF3), "capture")
    assert config["dynamics"]["collision_check_each_substep"] is True
    config = copy.deepcopy(config)
    config["env"].pop("collision_semantics", None)
    set_global_config(config)
    env = VorAdjEnv(config, seed=2026081502)
    assert env.collision_semantics == "synchronized_swept_v1"


def test_atomic_pair_collision_marks_both_and_is_list_order_invariant() -> None:
    for reverse in (False, True):
        env = _env()
        first, second = env.pursuers[:2]
        first.x = second.x = 40.0
        first.y = second.y = 40.0
        expected_ids = {int(first.id), int(second.id)}
        if reverse:
            env.pursuers[0], env.pursuers[1] = env.pursuers[1], env.pursuers[0]
        _resolve_without_motion(env)
        collided_ids = {int(p.id) for p in env.pursuers if p.collision}
        assert expected_ids <= collided_ids
        assert all(p.deactivated for p in env.pursuers if int(p.id) in expected_ids)


def test_legacy_endpoint_detector_reproduces_one_sided_order_bug_for_audit() -> None:
    env = _env("legacy_end_step")
    first, second = env.pursuers[:2]
    first.x = second.x = 40.0
    first.y = second.y = 40.0
    _resolve_without_motion(env)
    assert first.collision is True
    assert second.collision is False


def test_three_agent_overlap_is_resolved_atomically() -> None:
    env = _env()
    for pursuer in env.pursuers[:3]:
        pursuer.x, pursuer.y = 42.0, 42.0
    _resolve_without_motion(env)
    assert [p.collision for p in env.pursuers[:3]] == [True, True, True]
    assert env.episode_collision_type_counts["agent_agent"] == 3


def test_exact_agent_and_obstacle_tangency_count_as_contact() -> None:
    env = _env()
    first, second = env.pursuers[:2]
    first.x, first.y = 40.0, 40.0
    second.x = first.x + float(first.r) + float(second.r)
    second.y = first.y
    obstacle = env.obstacles[0]
    third = env.pursuers[2]
    obstacle.x, obstacle.y = 70.0, 70.0
    third.x = obstacle.x + float(third.r) + float(obstacle.r)
    third.y = obstacle.y
    _resolve_without_motion(env)
    assert first.collision and second.collision
    assert "agent_agent" in first.collision_types
    assert third.collision and "obstacle" in third.collision_types


def test_synchronized_aw_swept_contact_catches_safe_endpoint_graze() -> None:
    fixed = _env()
    legacy = _env("legacy_end_step")
    for env in (fixed, legacy):
        first, second = env.pursuers[:2]
        assert float(first.r) + float(second.r) > 2.2
        assert np.hypot(2.2, 1.0) > float(first.r) + float(second.r)
        first.x, first.y, first.theta, first.speed = 52.2, 49.0, np.pi / 2.0, 2.0
        second.x, second.y, second.theta, second.speed = 50.0, 50.0, 3.0 * np.pi / 2.0, 2.0
        first.velocity = np.asarray([0.0, 2.0], dtype=float)
        second.velocity = np.asarray([0.0, -2.0], dtype=float)
        env._begin_collision_step()
        try:
            env._move_robot(first, (0.0, 0.0))
            env._move_robot(second, (0.0, 0.0))
            env._refresh_collisions()
        finally:
            env._end_collision_step()
    assert fixed.pursuers[0].collision and fixed.pursuers[1].collision
    assert not legacy.pursuers[0].collision and not legacy.pursuers[1].collision


def test_safe_parallel_swept_paths_do_not_false_positive() -> None:
    env = _env()
    first, second = env.pursuers[:2]
    separation = float(first.r) + float(second.r) + 0.25
    first.x, first.y, first.theta, first.speed = 45.0, 45.0, 0.0, 2.0
    second.x, second.y, second.theta, second.speed = 45.0, 45.0 + separation, 0.0, 2.0
    first.velocity = second.velocity = np.asarray([2.0, 0.0], dtype=float)
    env._begin_collision_step()
    try:
        env._move_robot(first, (0.0, 0.0))
        env._move_robot(second, (0.0, 0.0))
        env._refresh_collisions()
    finally:
        env._end_collision_step()
    assert not first.collision and not second.collision


def test_boundary_and_evader_collision_provenance_are_episode_visible() -> None:
    env = _env()
    pursuer = env.pursuers[0]
    pursuer.x = float(pursuer.r)
    env._clip_and_kill_boundary(pursuer)
    evader = env.evaders[0]
    obstacle = env.obstacles[0]
    evader.x, evader.y = obstacle.x, obstacle.y
    _resolve_without_motion(env)
    fields = env._collision_episode_fields()
    assert "boundary" in pursuer.collision_types
    assert evader.collision and "obstacle" in evader.collision_types
    assert fields["pursuer_collision_event"] is True
    assert fields["evader_collision_event"] is True
    assert fields["collision_event"] is True
    assert fields["collision_type_counts"]["boundary"] == 1
    assert fields["collision_type_counts"]["obstacle"] == 1
