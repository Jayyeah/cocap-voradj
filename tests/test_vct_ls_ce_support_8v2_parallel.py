from __future__ import annotations

from pathlib import Path

from cocap_voradj.training.trainer import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = (
    ROOT
    / "configs"
    / "experiments"
    / "vct_ls_ce_support_8v2_parallel_20260802"
)


def _load(name: str):
    return load_config(str(CONFIG_ROOT / name))


def _assert_shared_8v2_contract(cfg) -> None:
    reward = cfg["reward"]
    voradj = cfg["voradj"]
    assert cfg["total_timesteps"] == 700_000
    assert cfg["iqn"]["checkpoint_freq"] == 100_000
    assert cfg["env"]["episode_max_length"] == 3000
    assert cfg["tasks"]["voradj"]["env"]["num_pursuers"] == 8
    assert cfg["tasks"]["voradj"]["env"]["num_evaders"] == 2
    assert cfg["tasks"]["voradj"]["env"]["num_obstacles"] == 2
    assert cfg["tasks"]["voradj_coverage"]["env"]["num_pursuers"] == 8
    assert cfg["tasks"]["voradj_coverage"]["env"]["num_evaders"] == 0

    assert reward.get("capture_reward_mode", "legacy") == "legacy"
    assert reward["omega_approach"] == 1.0
    assert reward["coverage_objective_version"] == "centroid_energy_v0"
    assert reward["coverage_ce_min_active_pursuers"] == 8
    assert reward["post_capture_coverage_window_steps"] == 600
    assert reward["coverage_ce_speed_weight_schedule"] == [
        {"step": 0, "value": 0.0},
        {"step": 200_000, "value": 0.0005},
    ]

    assert voradj["perception_topology_version"] == "friendly_voronoi_comm_v0"
    assert voradj["support_reward_blend_enabled"] is True
    assert voradj["support_reward_capture_weight"] == 0.5
    assert voradj["support_reward_coverage_weight"] == 0.5
    assert voradj["support_reward_capture_target_mode"] == "neighbor_visible"
    assert voradj["support_reward_capture_component_mode"] == "capture_task"
    assert voradj["replay_batch_counts"] == {
        "pursuing": 64,
        "pre_capture_cover": 16,
        "post_capture_real": 32,
        "recovery_pure": 16,
    }
    assert voradj["recovery"]["captured_state_ratio"] == 0.75
    assert voradj["recovery"]["map_random_ratio_within_non_capture"] == 0.5


def test_warm_8v2_uses_solid_4v1_support_checkpoint() -> None:
    cfg = _load("warm_from_4v1_support_700k.yaml")
    _assert_shared_8v2_contract(cfg)
    assert cfg["device"] == "cuda:0"
    assert cfg["iqn"]["learning_rate"] == 0.00003
    assert cfg["iqn"]["epsilon_start"] == 0.28
    assert cfg["iqn"]["learning_rate_schedule"] == [
        {"step": 0, "learning_rate": 0.00003}
    ]
    checkpoint = ROOT / cfg["pretrained"]["path"]
    assert checkpoint.is_file()


def test_scratch_8v2_has_no_pretrained_state() -> None:
    cfg = _load("scratch_700k.yaml")
    _assert_shared_8v2_contract(cfg)
    assert cfg["device"] == "cuda:1"
    assert cfg.get("pretrained") is None
    assert cfg["iqn"]["epsilon_start"] == 0.6
    assert cfg["iqn"]["learning_rate_schedule"] == [
        {"step": 0, "learning_rate": 0.0001},
        {"step": 200_000, "learning_rate": 0.00003},
    ]
