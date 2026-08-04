from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cocap_voradj.envs.coverage_ce import (
    BLOCKED_OWNER,
    build_masked_voronoi_map,
    ce_transition_reward,
)
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.trainer import deep_update, load_config, set_global_config


ROOT = Path(__file__).resolve().parents[1]
FINAL_CONFIG = ROOT / "configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml"


def _final_ce_config():
    config = load_config(str(FINAL_CONFIG))
    return deep_update(config, config["tasks"]["voradj_coverage"])


def _ce_env_config(**overrides):
    config = _final_ce_config()
    base = {
        "device": "cpu",
        "env": {
            "num_pursuers": 4,
            "num_evaders": 0,
            "num_obstacles": 1,
            "episode_max_length": 30,
            "init_speed": 0.0,
        },
        "reward": {
            "voradj_grid_size": 24,
            "coverage_ce_min_active_pursuers": 4,
            "coverage_ce_pbrs_reset_mode": "phase_and_all_terminal",
            "min_active_pursuers": 4,
        },
        "voradj": {
            "perception_topology_version": "legacy_voradj",
            "capture_adjacency_obstacle_mode": "legacy_assign",
            "voronoi_obstacle_mode": "free_mask_projected",
            "capture_episode_ends_on_capture": False,
        },
        "zone_demo": {"enabled": False},
    }
    return deep_update(deep_update(config, base), overrides)


def _grid(size: int = 41) -> tuple[np.ndarray, int]:
    axis = np.linspace(0.0, 10.0, size)
    mesh_x, mesh_y = np.meshgrid(axis, axis)
    return np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1), size


def test_masked_voronoi_excludes_obstacles_from_all_cells() -> None:
    points, grid_n = _grid()
    data = build_masked_voronoi_map(
        keys=[("pursuer", 0), ("pursuer", 1)],
        sites=np.asarray([[2.0, 5.0], [8.0, 5.0]], dtype=float),
        grid_points=points,
        grid_n=grid_n,
        obstacles=[(5.0, 5.0, 1.0)],
        robot_radius=0.5,
        grid_margin=0.1,
    )

    assert np.all(data["owners"][~data["free_mask"]] == BLOCKED_OWNER)
    assert sum(data["counts"].values()) == data["free_grid_count"]
    assert data["free_grid_count"] + data["blocked_grid_count"] == len(points)
    for centroid in data["centroids"].values():
        assert np.linalg.norm(np.asarray(centroid) - np.asarray([5.0, 5.0])) > 1.6


def test_disconnected_cell_centroid_projects_to_site_component() -> None:
    points, grid_n = _grid()
    wall = [(5.0, y, 0.8) for y in np.linspace(0.0, 10.0, 11)]
    key = ("pursuer", 0)
    disconnected = build_masked_voronoi_map(
        keys=[key],
        sites=np.asarray([[2.0, 5.0]], dtype=float),
        grid_points=points,
        grid_n=grid_n,
        obstacles=wall,
        robot_radius=0.0,
        grid_margin=0.0,
    )
    projected = build_masked_voronoi_map(
        keys=[key],
        sites=np.asarray([[2.0, 5.0]], dtype=float),
        grid_points=points,
        grid_n=grid_n,
        obstacles=[(5.0, 5.0, 1.5)],
        robot_radius=0.0,
        grid_margin=0.0,
    )

    assert disconnected["centroid_component_counts"][key] >= 2
    assert disconnected["centroid_disconnected_ratios"][key] > 0.0
    assert float(disconnected["centroids"][key][0]) < 5.0
    assert projected["centroid_projection_distances"][key] > 0.0
    assert np.linalg.norm(np.asarray(projected["centroids"][key]) - np.asarray([5.0, 5.0])) > 1.5


def test_ce_pbrs_terminal_correction_removes_terminal_potential() -> None:
    nonterminal, terms = ce_transition_reward(
        before_center_cost=0.04,
        after_center_cost=0.01,
        control_cost=0.002,
        reward_scale=10.0,
        gamma=0.99,
        pbrs_enabled=True,
        pbrs_kappa=1.0,
        terminal=False,
    )
    terminal, terminal_terms = ce_transition_reward(
        before_center_cost=0.04,
        after_center_cost=0.01,
        control_cost=0.002,
        reward_scale=10.0,
        gamma=0.99,
        pbrs_enabled=True,
        pbrs_kappa=1.0,
        terminal=True,
    )

    assert np.isclose(terminal - nonterminal, 10.0 * 0.99 * 0.01)
    assert terms["base_center"] == terminal_terms["base_center"]
    assert terms["control"] == terminal_terms["control"]


def test_config_extends_and_ce_success_ignore_speed_and_cv() -> None:
    config = _final_ce_config()
    assert config["reward"]["coverage_objective_version"] == "centroid_energy_v0"
    assert config["reward"]["coverage_ce_speed_weight"] == 0.0
    assert config["env"]["num_evaders"] == 0

    config = deep_update(
        config,
        {
            "reward": {"voradj_grid_size": 16},
            "env": {"episode_max_length": 20},
        },
    )
    set_global_config(config)
    env = VorAdjEnv(config, seed=17)
    env.reset()
    for pursuer in env.pursuers:
        pursuer.speed = float(pursuer.max_speed)
    data = env._voronoi_map()
    fake = dict(data)
    fake["centroids"] = {
        ("pursuer", i): env._position(pursuer).copy()
        for i, pursuer in enumerate(env.pursuers)
        if not pursuer.deactivated
    }
    fake["counts"] = {
        ("pursuer", i): 1 if i == 0 else 100
        for i, pursuer in enumerate(env.pursuers)
        if not pursuer.deactivated
    }

    metrics = env._voradj_coverage_geometry(fake, strict=True)
    assert metrics["area_cv"] > 0.3
    assert metrics["max_speed"] == max(p.max_speed for p in env.pursuers)
    assert metrics["ce_center_rms"] == 0.0
    assert metrics["ce_center_max"] == 0.0
    assert metrics["converged_now"] is True
    assert env._coverage_settle_enabled() is False


def test_legacy_config_keeps_legacy_voronoi_and_objective() -> None:
    base = load_config(
        str(
            ROOT
            / "configs/experiments/voradj_a3_apfnew_sqrtn_20260723"
            / "a3_apfnew_sqrtn_4v1_scratch_mix_2m.yaml"
        )
    )
    base = deep_update(base, {"reward": {"voradj_grid_size": 16}})
    set_global_config(base)
    env = VorAdjEnv(base, seed=19)
    env.reset()
    data = env._voronoi_map()

    assert env._ce_coverage_enabled() is False
    assert env._voronoi_obstacle_mode() == "legacy_assign"
    assert np.all(data["owners"] >= 0)
    assert "blocked_grid_count" not in data


def test_ce_step_success_uses_hold_without_legacy_terminal_bonus() -> None:
    config = _final_ce_config()
    config = deep_update(
        config,
        {
            "reward": {"voradj_grid_size": 24, "coverage_ce_success_hold_steps": 5},
            "env": {"episode_max_length": 30},
        },
    )
    set_global_config(config)
    env = VorAdjEnv(config, seed=29)
    env.reset()
    for _ in range(30):
        data = env._voronoi_map()
        for i, pursuer in enumerate(env.pursuers):
            centroid = np.asarray(data["centroids"][("pursuer", i)], dtype=float)
            pursuer.x, pursuer.y = float(centroid[0]), float(centroid[1])
            pursuer.speed = 0.0
            pursuer.velocity = np.zeros(2, dtype=float)
        env._invalidate_voronoi_cache()
    assert env._voradj_coverage_geometry(env._voronoi_map(), strict=True)["converged_now"] is True

    result = None
    for _ in range(5):
        result = env.step([4] * len(env.pursuers))
    assert result is not None
    assert env.post_capture_coverage_success is True
    assert env.coverage_geometric_success is True
    assert env.coverage_settled_success is True
    assert env.last_reward_terms["coverage_success_bonus"] == 0.0
    assert env.last_reward_terms["reward_terminal_mean"] == 0.0
    assert all(result.dones)


def test_ce_mix_uses_legacy_capture_graph_and_free_mask_coverage_target() -> None:
    config = _ce_env_config(env={"num_evaders": 1})
    set_global_config(config)
    env = VorAdjEnv(config, seed=41)
    observations = env.reset()

    capture_data = env._capture_voronoi_map()
    coverage_data = env._coverage_voronoi_map()
    assert capture_data is env._capture_voronoi_map()
    assert coverage_data is env._coverage_voronoi_map()
    assert capture_data is not coverage_data
    assert capture_data["voronoi_obstacle_mode"] == "legacy_assign"
    assert coverage_data["voronoi_obstacle_mode"] == "free_mask_projected"
    assert "blocked_grid_count" not in capture_data
    assert coverage_data["blocked_grid_count"] > 0

    idx = next(i for i, observation in enumerate(observations) if observation is not None)
    pursuer = env.pursuers[idx]
    centroid = np.asarray(coverage_data["centroids"][("pursuer", idx)], dtype=float)
    center_robot = env._robot_frame(
        pursuer, centroid - env._position(pursuer), True
    )
    expected = (
        center_robot
        / env._distance_scale()
        * env._center_sqrt_n_scale(coverage_data)
    )
    assert np.allclose(observations[idx]["self"][6:8], expected, atol=1e-6)


def test_final_capture_resets_pbrs_and_post_capture_starts_next_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _ce_env_config(env={"num_evaders": 1})
    set_global_config(config)
    env = VorAdjEnv(config, seed=43)
    env.reset()
    env._pursuing_flags_initialized = True
    for i, pursuer in enumerate(env.pursuers):
        pursuer.is_pursuing = i != 0
    env._pursuing_release_counters = [7] * len(env.pursuers)

    calls = 0

    def capture_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            return [
                {
                    "evader_id": 0,
                    "participants": [1],
                    "angles": [0.0],
                    "capture_type": "stationary",
                }
            ]
        return []

    monkeypatch.setattr(env, "_loose_capture_events", capture_once)
    first = env.step([4] * len(env.pursuers), [4])

    assert env.evaders[0].deactivated is True
    assert env.post_capture_started is False
    assert env.post_capture_step == 0
    assert all(not pursuer.is_pursuing for pursuer in env.pursuers if not pursuer.deactivated)
    assert env._pursuing_release_counters == [0] * len(env.pursuers)
    first_meta = first.infos[0]["replay_metadata"]
    assert first_meta["phase"] == "pre_capture"
    assert first_meta["coverage_ce_pbrs_reset_reason"] == "capture_phase_boundary"
    assert first_meta["reward_ce_terminal_correction"] >= 0.0
    assert not all(first.dones)

    second = env.step([4] * len(env.pursuers), [None])
    assert env.post_capture_started is True
    assert env.post_capture_step == 1
    assert second.infos[0]["replay_metadata"]["phase"] == "post_capture"


def test_ce_timeout_applies_generic_terminal_pbrs_correction() -> None:
    config = _ce_env_config(env={"episode_max_length": 1})
    set_global_config(config)
    env = VorAdjEnv(config, seed=47)
    env.reset()
    result = env.step([4] * len(env.pursuers), [])

    assert all(result.dones)
    metadata = [info["replay_metadata"] for info in result.infos]
    assert all(item["coverage_ce_pbrs_reset_reason"] == "terminal" for item in metadata)
    assert all(item["reward_ce_terminal_correction"] >= 0.0 for item in metadata)
