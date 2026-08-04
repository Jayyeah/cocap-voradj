from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.finalize_screened_run import (
    build_rollout_command,
    expected_steps,
    metric_payload,
    score_tuple,
)


def _summaries(
    capture: float,
    mix_capture: float,
    collision: float,
    coverage_cv: float,
    mix_cv: float,
):
    return {
        "capture": {
            "capture_success_rate": capture,
            "collision_rate": collision,
            "avg_steps": 400.0,
        },
        "coverage": {
            "coverage_success_rate": 0.2,
            "coverage_cv015_rate": coverage_cv,
            "collision_rate": 0.0,
            "avg_final_voronoi_cv": 0.1,
            "avg_steps": 1200.0,
        },
        "mix": {
            "capture_success_rate": mix_capture,
            "coverage_success_rate": 0.1,
            "coverage_cv015_rate": mix_cv,
            "collision_rate": collision,
            "avg_final_voronoi_cv": 0.2,
            "avg_steps": 1800.0,
        },
    }


def test_selection_prioritizes_capture_over_coverage() -> None:
    capture_candidate = metric_payload(_summaries(0.8, 0.8, 0.2, 0.3, 0.2))
    coverage_candidate = metric_payload(_summaries(0.7, 0.7, 0.0, 1.0, 1.0))
    assert score_tuple(capture_candidate, 100_000) > score_tuple(
        coverage_candidate,
        200_000,
    )


def test_selection_uses_collision_then_coverage_for_capture_ties() -> None:
    safer = metric_payload(_summaries(0.9, 0.9, 0.0, 0.2, 0.2))
    unsafe = metric_payload(_summaries(0.9, 0.9, 0.2, 1.0, 1.0))
    assert score_tuple(safer, 100_000) > score_tuple(unsafe, 200_000)

    lower_coverage = metric_payload(_summaries(0.9, 0.9, 0.0, 0.3, 0.2))
    higher_coverage = metric_payload(_summaries(0.9, 0.9, 0.0, 0.8, 0.7))
    assert score_tuple(higher_coverage, 100_000) > score_tuple(
        lower_coverage,
        200_000,
    )


def test_selection_prioritizes_ce_strict_over_cv_for_capture_collision_ties() -> None:
    high_ce_summaries = _summaries(1.0, 1.0, 0.0, 0.2, 0.2)
    high_ce_summaries["coverage"]["coverage_success_rate"] = 1.0
    high_ce_summaries["mix"]["coverage_success_rate"] = 1.0
    high_cv_summaries = _summaries(1.0, 1.0, 0.0, 0.9, 0.9)
    high_cv_summaries["coverage"]["coverage_success_rate"] = 0.5
    high_cv_summaries["mix"]["coverage_success_rate"] = 0.6

    assert score_tuple(
        metric_payload(high_ce_summaries),
        300_000,
    ) > score_tuple(
        metric_payload(high_cv_summaries),
        500_000,
    )


def test_expected_steps_requires_exact_stage_cap_grid() -> None:
    assert expected_steps(700_000, 100_000) == [
        100_000,
        200_000,
        300_000,
        400_000,
        500_000,
        600_000,
        700_000,
    ]
    with pytest.raises(ValueError):
        expected_steps(750_000, 100_000)


def test_formal_gifs_disable_trails_unless_explicitly_requested() -> None:
    args = Namespace(
        formal_episodes=20,
        gif_count=10,
        seed=123,
        device="cuda:1",
        workers=4,
        max_steps=2500,
        coverage_max_steps=1500,
        capture_evaders=2,
        draw_trails=False,
    )
    command = build_rollout_command(args, Path("config.yaml"), Path("model.pt"), Path("output"))
    assert "--draw-neighbor-edges" in command
    assert "--draw-sensing-circles" in command
    assert "--draw-trails" not in command

    args.draw_trails = True
    assert "--draw-trails" in build_rollout_command(
        args,
        Path("config.yaml"),
        Path("model.pt"),
        Path("output"),
    )
