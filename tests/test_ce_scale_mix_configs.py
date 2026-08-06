from __future__ import annotations

from pathlib import Path

from cocap_voradj.training.trainer import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/experiments/ce_coverage_20260730_scale_mix"


def test_ce10_pure8_config_scales_participant_contract() -> None:
    config = load_config(
        str(CONFIG_ROOT / "ce10_pure8_scratch_delayed_energy0005_300k.yaml")
    )

    assert config["env"]["num_pursuers"] == 8
    assert config["env"]["num_evaders"] == 0
    assert config["reward"]["coverage_ce_min_active_pursuers"] == 8
    assert config["reward"]["min_active_pursuers"] == 8
    assert config["reward"]["coverage_ce_pbrs_reset_mode"] == "phase_and_all_terminal"
    assert config["tasks"]["voradj"]["env"]["num_pursuers"] == 8
    assert config["voradj"]["capture_adjacency_obstacle_mode"] == "free_mask_projected"
    assert config["voradj"]["voronoi_obstacle_mode"] == "free_mask_projected"


def test_ce11_mix8_uses_unified_obstacle_masked_graph() -> None:
    config = load_config(
        str(CONFIG_ROOT / "ce11_mix8_a3warm_delayed_energy0005_500k.yaml")
    )

    assert config["env"]["num_pursuers"] == 8
    assert config["env"]["num_evaders"] == 2
    assert config["reward"]["coverage_objective_version"] == "centroid_energy_v0"
    assert config["reward"]["coverage_settle_enabled"] is False
    assert config["reward"]["coverage_early_speed_shaping_enabled"] is False
    assert config["voradj"]["capture_adjacency_obstacle_mode"] == "free_mask_projected"
    assert config["voradj"]["voronoi_obstacle_mode"] == "free_mask_projected"
    assert config["voradj"]["replay_batch_counts"] == {
        "pursuing": 64,
        "pre_capture_cover": 16,
        "post_capture_real": 24,
        "recovery_pure": 24,
    }
    checkpoint = ROOT / config["pretrained"]["path"]
    assert checkpoint.is_file()


def test_ce12_mix8_scratch_removes_warm_start() -> None:
    config = load_config(
        str(CONFIG_ROOT / "ce12_mix8_scratch_delayed_energy0005_500k.yaml")
    )

    assert config.get("pretrained") is None
    assert config["iqn"]["epsilon_start"] == 0.6
    assert config["iqn"]["learning_rate_schedule"][-1] == {
        "step": 250_000,
        "learning_rate": 0.00003,
    }
    assert config["voradj"]["capture_adjacency_obstacle_mode"] == "free_mask_projected"
    assert config["voradj"]["voronoi_obstacle_mode"] == "free_mask_projected"


def test_ce13_pure8_extends_ce8_schedule_without_legacy_settle() -> None:
    config = load_config(
        str(CONFIG_ROOT / "ce13_pure8_scratch_ce8schedule_500k.yaml")
    )

    assert config["device"] == "cuda:0"
    assert config["total_timesteps"] == 500_000
    assert config["train_mode"] == "voradj"
    assert config["env"]["num_pursuers"] == 8
    assert config["env"]["num_evaders"] == 0
    assert config["reward"]["coverage_objective_version"] == "centroid_energy_v0"
    assert config["reward"]["coverage_ce_min_active_pursuers"] == 8
    assert config["reward"]["coverage_ce_speed_weight_schedule"] == [
        {"step": 0, "value": 0.0},
        {"step": 200_000, "value": 0.0005},
    ]
    assert config["reward"]["coverage_settle_enabled"] is False
    assert config["reward"]["coverage_early_speed_shaping_enabled"] is False
    assert config["voradj"]["voronoi_obstacle_mode"] == "free_mask_projected"


def test_ce14_mix4v1_keeps_a3_capture_and_replaces_only_coverage() -> None:
    config = load_config(
        str(CONFIG_ROOT / "ce14_mix4v1_a3capture_cecoverage_scratch_500k.yaml")
    )

    assert config["device"] == "cuda:0"
    assert config["total_timesteps"] == 500_000
    assert config["train_mode"] == "voradj_mixed_coverage"
    assert config.get("pretrained") is None
    assert config["env"]["num_pursuers"] == 4
    assert config["env"]["num_evaders"] == 1
    assert config["reward"]["coverage_objective_version"] == "centroid_energy_v0"
    assert config["reward"]["coverage_ce_min_active_pursuers"] == 4
    assert config["reward"]["coverage_ce_speed_weight_schedule"] == [
        {"step": 0, "value": 0.0},
        {"step": 200_000, "value": 0.0005},
    ]
    assert config["reward"]["coverage_settle_enabled"] is False
    assert config["reward"]["coverage_early_speed_shaping_enabled"] is False
    assert config["reward"]["coverage_motion_success_enabled"] is False
    assert config["reward"]["coverage_cell_center_speed_penalty_enabled"] is False
    assert config["reward"]["coverage_success_reward"] == 0.0
    assert config["reward"]["post_capture_coverage_success_reward"] == 0.0
    assert config["reward"]["post_capture_coverage_window_steps"] == 300
    assert config["voradj"]["capture_adjacency_obstacle_mode"] == "legacy_assign"
    assert config["voradj"]["voronoi_obstacle_mode"] == "free_mask_projected"
    assert config["voradj"]["replay_batch_counts"] == {
        "pursuing": 64,
        "pre_capture_cover": 16,
        "post_capture_real": 16,
        "recovery_pure": 32,
    }
    assert config["vct_ls"]["enabled"] is False

