from __future__ import annotations

from pathlib import Path

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.continuous.formal_config import resolve_ladder_config, scene_config
from cocap_voradj.training.trainer import set_global_config
from tools.run_continuous_ctde_training import _make_trainer, _screen


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/experiments/parallel_ce_legacy_voradj_20260809"
PURE = CONFIG_DIR / "pure_ce_4p0e1obs_200k_aw.yaml"
LEGACY = CONFIG_DIR / "legacy_voradj_oldmix_4p1e1obs_200k_aw.yaml"


def test_pure_ce_line_contract() -> None:
    config = resolve_ladder_config(PURE)

    assert config["env"]["num_pursuers"] == 4
    assert config["env"]["num_evaders"] == 0
    assert config["env"]["num_obstacles"] == 1
    assert config["training"]["total_env_steps"] == 200000
    assert config["training"]["checkpoint_interval_env_steps"] == 25000
    assert config["training"]["diagnostic_eval_interval_env_steps"] == 25000
    assert config["training"]["periodic_checkpoint_replay_mode"] == "rolling_latest"
    assert config["training"]["scene_cycle"] == ["pure_ce"]
    assert config["training"]["focal_quota"]["pure_recovery_coverage"] == 128
    assert config["reward"]["coverage_objective_version"] == "centroid_energy_v0"
    assert config["reward"]["coverage_ce_speed_weight"] == 0.0001
    assert config["reward"]["coverage_ce_success_rms_threshold"] == 0.05
    assert config["reward"]["coverage_ce_success_max_threshold"] == 0.10
    assert config["reward"]["coverage_ce_success_hold_steps"] == 30
    assert config["reward"]["coverage_cv_loose_area_cv_threshold"] == 0.20
    assert config["evaluation"]["diagnostic_scenes"] == ["pure_ce"]


def test_legacy_voradj_old_mix_contract() -> None:
    config = resolve_ladder_config(LEGACY)
    reward = config["reward"]
    voradj = config["voradj"]
    quota = config["training"]["focal_quota"]

    assert config["training"]["scene_cycle"] == ["mixed_crms", "pure_ce"]
    assert config["evaluation"]["diagnostic_scenes"] == ["capture", "pure_ce", "mixed_crms"]
    assert reward["coverage_objective_version"] == "centroid_energy_v0"
    assert reward["capture_reward_mode"] == "legacy"
    assert reward["voradj_grid_size"] == 60
    assert reward["omega_approach"] == 1.0
    assert reward["omega_mean_shift"] == 2.0
    assert reward["omega_front"] == 0.5
    assert reward["k_required"] == 3
    assert reward["capture_stationary_min_pursuers"] == 2
    assert voradj["perception_topology_version"] == "legacy_voradj"
    assert voradj["capture_adjacency_obstacle_mode"] == "legacy_assign"
    assert voradj["voronoi_obstacle_mode"] == "free_mask_projected"
    assert voradj["is_pursuing_release_delay_steps"] == 10
    assert config["apf"]["version"] == "v2_fixed"
    assert config["perception"]["evader_observation_mode"] == "apf_compatible"
    assert len(config["evader"]["w"]) == 5
    assert quota == {
        "pre_capture_pursuing": 64,
        "pre_capture_support": 0,
        "pre_capture_coverage": 16,
        "post_capture_coverage": 32,
        "pure_recovery_coverage": 16,
    }


def test_legacy_capture_graph_includes_evader_site() -> None:
    config = scene_config(resolve_ladder_config(LEGACY), "capture")
    config["env"]["episode_max_length"] = 2
    set_global_config(config)
    env = VorAdjEnv(config, seed=2026080902)
    env.reset()

    data = env._capture_voronoi_map()

    assert env._vct_ls_enabled() is False
    assert data["site_scope"] == "all_active_entities"
    assert any(key[0] == "evader" for key in data["keys"])
    assert data["voronoi_obstacle_mode"] == "legacy_assign"


def test_all_enemies_captured_switches_directly_to_coverage() -> None:
    config = scene_config(resolve_ladder_config(LEGACY), "mixed_crms")
    set_global_config(config)
    env = VorAdjEnv(config, seed=2026080902)
    env.reset()
    before_capture = env._capture_voronoi_map()

    for pursuer in env.pursuers:
        pursuer.is_pursuing = True
    env._pursuing_flags_initialized = True
    env._pursuing_release_counters = [10] * len(env.pursuers)
    env.evaders[0].deactivated = True
    env._clear_release_delay_after_single_enemy_capture(before_capture, {0})
    env._invalidate_voronoi_cache()

    assert env._pursuing_release_counters == [0] * len(env.pursuers)
    assert all(not pursuer.is_pursuing for pursuer in env.pursuers)
    data = env._capture_voronoi_map()
    switched = env._task_labels_from_map(data, update_effective=True)
    assert all(label == "coverage" for label in switched)


def test_diagnostic_screen_exposes_ce_and_capture_progress_signals() -> None:
    config = resolve_ladder_config(LEGACY)
    trainer = _make_trainer(config, "cpu")

    result = _screen(
        trainer,
        config,
        seed=2026080902,
        episodes=1,
        device="cpu",
        scenes=("capture", "pure_ce"),
        max_steps=2,
    )

    capture = result["capture"]
    pure = result["pure_ce"]
    for key in (
        "capture_rate",
        "detected_rate",
        "mean_discovery_step",
        "mean_initial_min_distance",
        "mean_final_min_distance",
        "mean_min_min_distance",
        "mean_distance_progress",
    ):
        assert key in capture
    for key in (
        "coverage_strict_rate",
        "coverage_cv020_rate",
        "mean_coverage_area_cv",
        "mean_coverage_ce_center_rms",
        "mean_coverage_ce_center_max",
        "mean_ce_energy_progress",
    ):
        assert key in pure
