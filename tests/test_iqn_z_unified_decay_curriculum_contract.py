from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools import iqn_z_unified_decay_curriculum_20260919 as run
from tools import supervise_iqn_z_unified_decay_dual_20260919 as dual


def _summary(
    *,
    pure_capture: float,
    mixed_capture: float,
    coverage: float,
    safe: float = 0.0,
    post_ce: float = 0.0,
    collision: float = 0.0,
    ce_rms: float = 0.2,
    area_cv: float = 0.3,
    capture_seconds: float = 20.0,
) -> dict:
    return {
        "coverage": {
            "strict_ce_rate": coverage,
            "collision_rate": collision,
            "ce_rms": {"mean": ce_rms},
            "area_cv": {"mean": area_cv},
        },
        "capture": {
            "normal_capture_rate": pure_capture,
            "collision_rate": collision,
            "capture_seconds": {"mean": capture_seconds},
        },
        "mixed": {
            "capture_rate": mixed_capture,
            "safe_complete_rate": safe,
            "post_capture_ce_rate": post_ce,
            "collision_rate": collision,
        },
    }


def _write_report(root: Path, step: int, summary: dict) -> None:
    path = root / "evaluations" / f"step_{step:09d}" / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "checkpoint": str(root / f"step_{step}.pt"),
                "checkpoint_sha256": f"sha-{step}",
                "summary": summary,
            }
        ),
        encoding="utf-8",
    )


def test_dual_line_contract_diff_has_only_alpha(tmp_path: Path) -> None:
    reports = run.write_contract_reports(tmp_path)
    for stages in reports["arms"].values():
        assert all(report["unexpected_difference_count"] == 0 for report in stages.values())
    for report in reports["cross"].values():
        assert report["non_alpha_difference_count"] == 0
        assert {row["path"] for row in report["differences"]} == run.ALLOWED_ARM_DIFFS


@pytest.mark.parametrize(("arm", "alpha"), [("z05", 0.5), ("z07", 0.7)])
def test_all_stages_keep_registered_scientific_contract(arm: str, alpha: float) -> None:
    for stage in run.STAGE_ORDER:
        config = run.resolved(arm, stage)
        run._assert_production_contract(config, alpha)
        assert config["iqn"]["checkpoint_freq"] == 100_000
        assert config["formal_evaluation"]["episodes_per_scene"] == 20
        assert config["formal_evaluation"]["checkpoint_interval"] == 100_000
        assert config["checkpointing"]["full_resume"] is True


def test_balanced_floor_selection_prefers_joint_strength(tmp_path: Path) -> None:
    milestones = (100_000, 200_000, 300_000)
    _write_report(tmp_path, 100_000, _summary(pure_capture=1.0, mixed_capture=1.0, coverage=0.2))
    _write_report(tmp_path, 200_000, _summary(pure_capture=0.8, mixed_capture=0.8, coverage=0.75))
    _write_report(tmp_path, 300_000, _summary(pure_capture=0.6, mixed_capture=0.6, coverage=0.9))
    report = run.select_balanced(tmp_path, milestones)
    assert report["fallback_used"] is False
    assert report["selected"]["step"] == 200_000
    assert report["selected"]["balanced_floor"] == pytest.approx(0.75)


def test_zero_strict_coverage_fallback_selects_maximin_compromise(tmp_path: Path) -> None:
    milestones = (100_000, 200_000, 300_000)
    _write_report(tmp_path, 100_000, _summary(pure_capture=0.9, mixed_capture=0.9, coverage=0.0, ce_rms=0.5, area_cv=0.8))
    _write_report(tmp_path, 200_000, _summary(pure_capture=0.7, mixed_capture=0.7, coverage=0.0, ce_rms=0.3, area_cv=0.5))
    _write_report(tmp_path, 300_000, _summary(pure_capture=0.5, mixed_capture=0.5, coverage=0.0, ce_rms=0.1, area_cv=0.2))
    report = run.select_balanced(tmp_path, milestones)
    assert report["fallback_used"] is True
    assert report["selected"]["step"] == 200_000
    assert report["selected"]["fallback_maximin"] == pytest.approx(0.5)
    assert "maximin compromise" in report["selection_reason"]


def test_final_dual_comparison_contains_all_stage_metrics_and_chinese_report(tmp_path: Path) -> None:
    for arm, capture in (("z05", 0.6), ("z07", 0.8)):
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        stages = []
        for index, stage in enumerate(run.STAGE_ORDER, start=1):
            summary = _summary(
                pure_capture=capture,
                mixed_capture=capture - 0.1,
                coverage=0.2 * index,
            )
            summary["z"] = {
                "max_lineage_hop": index,
                "post_capture_never_release_episodes": 0,
            }
            stages.append(
                {
                    "stage": stage,
                    "selected_step": index * 100_000,
                    "selected_checkpoint": str(arm_dir / f"{stage}.pt"),
                    "formal_summary": summary,
                }
            )
        (arm_dir / f"{arm.upper()}_CURRICULUM_FINAL_REPORT.json").write_text(
            json.dumps({"arm": arm, "stages": stages}),
            encoding="utf-8",
        )
    dual.write_final_comparison(tmp_path)
    comparison = json.loads((tmp_path / "Z05_VS_Z07_FINAL_COMPARISON.json").read_text())
    assert list(comparison["stage_comparison"]) == list(run.STAGE_ORDER)
    assert comparison["stage_comparison"]["stage3"]["z07"]["capture"]["normal_capture_rate"] == pytest.approx(0.8)
    markdown = (tmp_path / "Z05_VS_Z07_FINAL_COMPARISON_ZH.md").read_text()
    assert "Stage" not in markdown
    assert "stage1" in markdown and "stage2" in markdown and "stage3" in markdown


@pytest.mark.parametrize(
    ("runtime", "expected"),
    [
        ({"status": "running", "pid_alive": True, "tmux_present": True}, False),
        ({"status": "running", "pid_alive": False, "tmux_present": False}, True),
        ({"status": "missing", "pid_alive": False, "tmux_present": False}, True),
        ({"status": "failed_closed", "pid_alive": False, "tmux_present": False}, False),
        ({"status": "complete", "pid_alive": False, "tmux_present": False}, False),
    ],
)
def test_dual_supervisor_launch_policy(runtime: dict, expected: bool) -> None:
    assert dual.needs_launch(runtime) is expected


def test_dual_supervisor_pid_alive_detects_current_process() -> None:
    assert dual.pid_alive(os.getpid()) is True
    assert dual.pid_alive(None) is False


def test_dual_supervisor_arm_runtime_falls_back_to_launch_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm_dir = tmp_path / "z05"
    arm_dir.mkdir()
    (arm_dir / "status.json").write_text(
        json.dumps({"status": "running", "current_step": 123}),
        encoding="utf-8",
    )
    (arm_dir / "launch.json").write_text(
        json.dumps({"pid": os.getpid(), "cuda_visible_devices": "1"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(dual, "tmux_present", lambda _session: True)

    runtime = dual.arm_runtime(tmp_path, "z05", {0: set(), 1: {os.getpid()}})

    assert runtime["pid"] == os.getpid()
    assert runtime["pid_alive"] is True
    assert runtime["assigned_gpu"] == 1
    assert runtime["observed_gpus"] == [1]
    assert runtime["wrong_gpu"] is False
    assert runtime["current_step"] == 123
