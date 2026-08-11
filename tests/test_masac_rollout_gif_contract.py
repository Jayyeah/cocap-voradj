from __future__ import annotations

import argparse
from pathlib import Path

from cocap_voradj.training.continuous.formal_config import scene_config
from cocap_voradj.training.trainer import load_config
from tools.run_masac_rollout_gifs import apply_overrides, load_preset, split_episode_indices


ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "configs/evaluation/masac_rollout_gif_20260811"


def _args(**overrides):
    values = {
        "episodes": None,
        "gif_count": None,
        "seed": None,
        "workers": None,
        "horizon_cap": None,
        "frame_duration_ms": None,
        "max_gif_frames": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_presets_freeze_20_rollout_5_gif_and_10_fps() -> None:
    for filename in ("pure_coverage.yaml", "old_mix.yaml", "vct_ls_capture.yaml"):
        preset = load_preset(PRESETS / filename)
        assert preset["episodes"] == 20
        assert preset["gif_count"] == 5
        assert preset["workers"] == 4
        assert preset["display"]["frame_duration_ms"] == 100
        assert preset["display"]["max_gif_frames"] == 300
        assert preset["display"]["draw_trails"] is False


def test_scene_specific_display_and_horizon_contracts() -> None:
    pure = load_preset(PRESETS / "pure_coverage.yaml")
    assert [(item["scene"], item["max_steps"]) for item in pure["scenarios"]] == [("pure_ce", 1500)]
    assert pure["display"]["draw_neighbor_edges"] is False
    assert pure["display"]["draw_sensing_circles"] is False
    assert pure["scenarios"][0]["draw_ce_targets"] is True

    old_mix = load_preset(PRESETS / "old_mix.yaml")
    assert [(item["scene"], item["max_steps"]) for item in old_mix["scenarios"]] == [
        ("capture", 1000),
        ("pure_ce", 1500),
        ("mixed_crms", 1500),
    ]
    assert old_mix["display"]["draw_neighbor_edges"] is True
    assert old_mix["display"]["draw_sensing_circles"] is False

    vct_ls = load_preset(PRESETS / "vct_ls_capture.yaml")
    assert [(item["scene"], item["max_steps"]) for item in vct_ls["scenarios"]] == [("capture", 1000)]
    assert vct_ls["display"]["draw_neighbor_edges"] is True
    assert vct_ls["display"]["draw_sensing_circles"] is True
    assert vct_ls["scenarios"][0]["draw_ce_targets"] is False


def test_presets_match_their_source_task_horizons_and_topologies() -> None:
    cases = [
        (
            "pure_coverage.yaml",
            ROOT / "configs/experiments/parallel_ce_legacy_voradj_20260809/pure_ce_4p0e1obs_200k_aw.yaml",
            "friendly_voronoi_comm_v0",
        ),
        (
            "old_mix.yaml",
            ROOT / "configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_oldmix_4p1e1obs_200k_aw.yaml",
            "legacy_voradj",
        ),
        (
            "vct_ls_capture.yaml",
            ROOT / "configs/experiments/positive_feedback_ladder_20260807/stage4a_capture_aw.yaml",
            "friendly_voronoi_comm_v0",
        ),
    ]
    for preset_name, config_path, topology in cases:
        preset = load_preset(PRESETS / preset_name)
        config = load_config(str(config_path))
        assert config["voradj"]["perception_topology_version"] == topology
        for item in preset["scenarios"]:
            effective = scene_config(config, item["scene"])
            assert int(item["max_steps"]) == int(effective["env"]["episode_max_length"])


def test_cli_smoke_overrides_do_not_mutate_source_preset() -> None:
    source = load_preset(PRESETS / "old_mix.yaml")
    resolved = apply_overrides(
        source,
        _args(episodes=1, gif_count=1, workers=1, horizon_cap=2, max_gif_frames=2),
    )
    assert source["episodes"] == 20
    assert [item["max_steps"] for item in source["scenarios"]] == [1000, 1500, 1500]
    assert resolved["episodes"] == 1
    assert resolved["gif_count"] == 1
    assert resolved["workers"] == 1
    assert resolved["display"]["max_gif_frames"] == 2
    assert [item["max_steps"] for item in resolved["scenarios"]] == [2, 2, 2]


def test_split_episode_indices_covers_every_episode_once_and_spreads_gifs() -> None:
    chunks = split_episode_indices(20, 4)
    assert chunks == [
        [0, 4, 8, 12, 16],
        [1, 5, 9, 13, 17],
        [2, 6, 10, 14, 18],
        [3, 7, 11, 15, 19],
    ]
    assert sorted(index for chunk in chunks for index in chunk) == list(range(20))
    assert [sum(index < 5 for index in chunk) for chunk in chunks] == [2, 1, 1, 1]
