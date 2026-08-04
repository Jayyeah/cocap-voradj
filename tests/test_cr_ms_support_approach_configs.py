from __future__ import annotations

from pathlib import Path

from cocap_voradj.training.trainer import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/experiments/cr_ms_support_approach_ce_curriculum_20260802"


def _load(name: str):
    return load_config(str(CONFIG_ROOT / name))


def test_stage1_uses_cr_ms_support_approach_latest_ce_and_full_2m() -> None:
    cfg = _load("stage1_4p1e1obs_scratch2m.yaml")
    reward = cfg["reward"]
    voradj = cfg["voradj"]

    assert cfg["total_timesteps"] == 2_000_000
    assert cfg["device"] == "cuda:0"
    assert cfg["iqn"]["checkpoint_freq"] == 100_000
    assert cfg.get("pretrained", {}).get("path") is None

    assert reward["capture_reward_mode"] == "ring_importance_ms_v0"
    assert reward["capture_timestep_penalty"] == 0.0
    assert reward["omega_ring_ms"] == 2.0
    assert reward["omega_approach"] == 0.0
    assert reward["omega_mean_shift"] == 0.0
    assert reward["omega_front"] == 0.0

    assert voradj["perception_topology_version"] == "friendly_voronoi_comm_v0"
    assert voradj["enemy_sensing_radius"] == 20.0
    assert voradj["local_sensing_uses_surface_distance"] is True
    assert voradj["support_reward_blend_enabled"] is True
    assert voradj["support_reward_capture_weight"] == 0.5
    assert voradj["support_reward_coverage_weight"] == 0.5
    assert voradj["support_reward_capture_target_mode"] == "neighbor_visible"
    assert voradj["support_reward_capture_component_mode"] == "approach_only"
    assert voradj["support_reward_approach_weight"] == 1.0
    assert voradj["support_reward_approach_clip"] == 3.0

    assert reward["coverage_objective_version"] == "centroid_energy_v0"
    assert reward["coverage_ce_speed_weight_schedule"] == [
        {"step": 0, "value": 0.0},
        {"step": 200_000, "value": 0.0005},
    ]
    assert reward["post_capture_coverage_window_steps"] == 500
    assert voradj["replay_batch_counts"] == {
        "pursuing": 64,
        "pre_capture_cover": 16,
        "post_capture_real": 32,
        "recovery_pure": 16,
    }
    assert voradj["recovery"]["captured_state_ratio"] == 0.75
    assert voradj["recovery"]["map_random_ratio_within_non_capture"] == 0.5


def test_stage2_and_stage3_course_shapes_and_windows() -> None:
    stage2 = _load("stage2_8p2e2obs_700k.yaml")
    stage3 = _load("stage3_12p3e3obs_700k.yaml")

    assert stage2["total_timesteps"] == 700_000
    assert stage2["tasks"]["voradj"]["env"]["num_pursuers"] == 8
    assert stage2["tasks"]["voradj"]["env"]["num_evaders"] == 2
    assert stage2["tasks"]["voradj"]["env"]["num_obstacles"] == 2
    assert stage2["reward"]["coverage_ce_min_active_pursuers"] == 8
    assert stage2["reward"]["post_capture_coverage_window_steps"] == 600
    assert stage2["iqn"]["learning_rate_schedule"] == [{"step": 0, "learning_rate": 0.00003}]
    assert stage2["pretrained"].get("path") is None

    assert stage3["total_timesteps"] == 700_000
    assert stage3["tasks"]["voradj"]["env"]["num_pursuers"] == 12
    assert stage3["tasks"]["voradj"]["env"]["num_evaders"] == 3
    assert stage3["tasks"]["voradj"]["env"]["num_obstacles"] == 3
    assert stage3["reward"]["coverage_ce_min_active_pursuers"] == 12
    assert stage3["reward"]["post_capture_coverage_window_steps"] == 700
    assert stage3["pretrained"].get("path") is None
