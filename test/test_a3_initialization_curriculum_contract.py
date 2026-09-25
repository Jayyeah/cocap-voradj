from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.trainer import load_config, set_global_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/experiments/a3_capture_initialization_20260925"
CANONICAL = ROOT / "configs/experiments/forward_final_mappo_20260908/stage1_4v1.yaml"


def _load(stage: str):
    config = load_config(str(CONFIG_ROOT / f"stage_{stage.lower()}.yaml"))
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=2026092503)
    env.reset()
    return config, env


def _contract_projection(config):
    return {
        key: copy.deepcopy(config.get(key))
        for key in (
            "env",
            "pursuer",
            "evader",
            "perception",
            "apf",
            "reward",
            "voradj",
            "tasks",
            "action_mode",
            "dynamics",
        )
    }


def test_a3_configs_keep_canonical_final_contract():
    canonical = load_config(str(CANONICAL))
    for stage in ("I0", "I1", "I2", "I3"):
        config, _env = _load(stage)
        assert _contract_projection(config) == _contract_projection(canonical)
        assert config["a3_initialization_curriculum"]["stage"] == stage
        assert config["a3_initialization_curriculum"]["enabled"] is True


def test_a3_ring_stages_have_exact_initial_geometry_and_no_terminal_event():
    for stage, ranges, expected_direct in (
        ("I0", [(12.0, 13.0)] * 4, 4),
        ("I1", [(13.0, 16.0)] * 4, 4),
    ):
        for seed in range(2026092503, 2026092511):
            config = load_config(str(CONFIG_ROOT / f"stage_{stage.lower()}.yaml"))
            set_global_config(config)
            env = VorAdjEnv(copy.deepcopy(config), seed=seed)
            env.reset()
            metadata = env.initialization_metadata()
            assert metadata["stage"] == stage
            assert metadata["source"] == "a3_ring_sampler"
            assert metadata["initial_capture_events"] == 0
            assert metadata["initial_collision_events"] == 0
            assert np.isclose(metadata["target_speed"], 0.0)
            assert metadata["direct_visible_count_by_center_range"] == expected_direct
            assert metadata["pursuer_pairwise_min_center_distance"] >= 15.0
            for distance, (low, high) in zip(metadata["pursuer_target_distances"], ranges):
                assert low <= distance <= high


def test_a3_i2_randomizes_visibility_lanes_but_keeps_two_direct():
    assignments = set()
    config = load_config(str(CONFIG_ROOT / "stage_i2.yaml"))
    for seed in range(2026092503, 2026092527):
        set_global_config(config)
        env = VorAdjEnv(copy.deepcopy(config), seed=seed)
        env.reset()
        metadata = env.initialization_metadata()
        assignments.add(tuple(metadata["pursuer_assignment"]))
        assert metadata["stage"] == "I2"
        assert metadata["direct_visible_count_by_center_range"] == 2
        assert metadata["initial_capture_events"] == 0
        assert metadata["initial_collision_events"] == 0
        assert metadata["pursuer_pairwise_min_center_distance"] >= 15.0
        assert sum(distance <= 20.0 for distance in metadata["pursuer_target_distances"]) == 2
        assert sum(distance >= 23.0 for distance in metadata["pursuer_target_distances"]) == 2
    assert len(assignments) >= 2


def test_a3_i3_uses_canonical_map_random_and_mixed_coverage_skips_capture_sampler():
    config, env = _load("I3")
    metadata = env.initialization_metadata()
    assert metadata["source"] == "canonical_map_random"
    assert metadata["stage"] == "I3"
    assert min(metadata["pursuer_target_distances"]) >= 15.0
    assert metadata["initial_capture_events"] == 0
    coverage = copy.deepcopy(config)
    coverage["env"] = copy.deepcopy(coverage["env"])
    coverage["env"]["num_evaders"] = 0
    set_global_config(coverage)
    coverage_env = VorAdjEnv(coverage, seed=2026092503)
    coverage_env.reset()
    assert coverage_env.initialization_metadata()["source"] == "skipped_no_evader_task"

