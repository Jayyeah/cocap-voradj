from pathlib import Path

import yaml

from maadpg_reproduction.cli import build_parser


ROOT = Path(__file__).resolve().parents[1]


def test_formal_manifest_is_paper_first_with_a_step_safety_cap():
    config = yaml.safe_load(
        (
            ROOT
            / "configs/experiments/maadpg_reproduction_20260823/matched_formal.yaml"
        ).read_text()
    )
    assert config["episode_budget_per_run"] == 1500
    assert config["environment_step_safety_cap_per_run"] == 450000
    assert config["training_seeds"] == [2026082301, 2026082302, 2026082303]


def test_supervisor_is_locked_idempotent_and_passes_both_budgets():
    script = (ROOT / "tools/run_maadpg_formal_queue.sh").read_text()
    assert "flock -n 9" in script
    assert "--episode-budget 1500" in script
    assert "--step-budget 450000" in script
    assert "--resume" in script
    assert script.count("2026082301 2026082302 2026082303") == 1


def test_cli_accepts_episode_budget_and_full_evaluation_modes():
    parser = build_parser()
    train = parser.parse_args(
        [
            "train",
            "--variant",
            "maddpg",
            "--run-dir",
            "/tmp/example",
            "--seed",
            "1",
            "--step-budget",
            "450000",
            "--episode-budget",
            "1500",
        ]
    )
    assert train.episode_budget == 1500
    for mode in ("deterministic", "stochastic", "guidance"):
        evaluation = parser.parse_args(
            [
                "evaluate",
                "--checkpoint",
                "model.pt",
                "--output",
                f"{mode}.json",
                "--mode",
                mode,
            ]
        )
        assert evaluation.mode == mode
