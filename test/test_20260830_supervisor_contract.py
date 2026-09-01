from __future__ import annotations

import json
from pathlib import Path

import yaml

from tools import reevaluate_iqn_vxy_best_20260830 as reeval
from tools import run_continuous_ctde_training as ctde
from tools import supervise_iqn_vxy_full_20260830 as full
from tools import supervise_mappo9_v2_20260830 as mappo


def test_full_vxy_supervisor_has_historical_stage_order_and_25k_contract() -> None:
    assert full.INTERVAL == 25_000
    assert [(row["label"], row["steps"]) for row in full.STAGES] == [
        ("4p1e1obs", 2_000_000),
        ("8p2e2obs", 700_000),
        ("12p3e3obs", 700_000),
    ]


def test_fixed_tau_protocols_cover_uniform_and_unique_per_seed_best() -> None:
    uniform = {seed: step for seed, _, _, step in reeval.specs("uniform225")}
    selected = {seed: step for seed, _, _, step in reeval.specs("per-seed-best")}
    assert uniform == {1: 225_000, 2: 225_000, 3: 225_000}
    assert selected == {1: 250_000, 2: 225_000, 3: 225_000}
    assert {seed: step for seed, _, _, step in reeval.specs("uniform300")} == {
        1: 300_000,
        2: 300_000,
        3: 300_000,
    }


def test_reward_tail_summary_uses_environment_steps_and_agent_rows() -> None:
    role_steps = [
        [{"capture_objective_role": "direct_capture", "capture": 1.0, "total": 2.0}],
        [
            {"capture_objective_role": "direct_capture", "capture": 2.0, "total": 3.0},
            {"capture_objective_role": "one_hop_informed", "coverage": 4.0, "total": 5.0},
        ],
        [{"capture_objective_role": "direct_capture", "terminal": 6.0, "total": 7.0}],
    ]
    summary = ctde._reward_tail_summaries(
        role_steps,
        [["boundary"], [], ["obstacle"]],
        (2,),
    )["2"]
    assert summary["observed_window_steps"] == 2
    assert summary["agent_steps"] == 3
    assert summary["component_sums"]["capture"] == 2.0
    assert summary["component_sums"]["coverage"] == 4.0
    assert summary["component_sums"]["terminal"] == 6.0
    assert summary["collision_step_rate"] == 0.5
    assert summary["boundary_step_rate"] == 0.0


def test_full_vxy_gate_blocks_missing_metrics_and_accepts_audited_thresholds(tmp_path: Path) -> None:
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"model")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selected_checkpoint": str(checkpoint),
                "selected_screening": {
                    "metrics": {
                        "capture_success_rate": 0.5,
                        "mix_capture_rate": 0.5,
                        "coverage_ce_strict_rate": 0.1,
                        "mix_ce_strict_rate": 0.1,
                        "max_collision_rate": 0.5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    passed, detail = full.gate_selection(selection)
    assert passed and all(detail["checks"].values())
    selection.write_text(json.dumps({"selected_checkpoint": str(checkpoint)}), encoding="utf-8")
    assert full.gate_selection(selection)[0] is False


def test_mappo_supervisor_warn_and_pause_thresholds_are_sustained() -> None:
    assert mappo.health([{"approx_kl_max": 0.06, "clip_fraction": 0.1}])["warn"] is True
    assert mappo.health([{"approx_kl_max": 0.11}] * 2)["critical"] is False
    critical = mappo.health([{"approx_kl_max": 0.11}] * 3)
    assert critical["critical"] is True
    assert critical["critical_kl_streak"] == 3
    assert mappo.health([{"value_loss": float("inf")}])["critical"] is True
    assert mappo.health([{"actor_loss": float("nan")}])["finite"] is False


def test_full_vxy_supervisor_rejects_nonfinite_metrics() -> None:
    assert full.metrics_are_finite(None) is True
    assert full.metrics_are_finite({"loss": 1.0, "step": 10}) is True
    assert full.metrics_are_finite({"loss": float("inf")}) is False
    assert full.metrics_are_finite({"loss": float("nan")}) is False


def test_mappo_gate_routes_pass_and_healthy_fail(tmp_path: Path, monkeypatch) -> None:
    seeds = []
    for index, capture in enumerate((0.1, 0.2, 0.3), start=1):
        run = tmp_path / f"seed{index}"
        (run / "evaluations").mkdir(parents=True)
        (run / "evaluations/step_000025000.json").write_text(
            json.dumps(
                {
                    "step": 25_000,
                    "deterministic": {"capture": {"capture_rate": capture, "collision_rate": 0.2}},
                }
            ),
            encoding="utf-8",
        )
        (run / "learning_metrics.jsonl").write_text(
            json.dumps({"approx_kl_max": 0.01, "clip_fraction": 0.1}) + "\n",
            encoding="utf-8",
        )
        seeds.append({"index": index, "run": run})
    monkeypatch.setattr(mappo, "SEEDS", tuple(seeds))
    assert mappo.final_gate()["decision"] == "PASS_TO_MAPPO_AW_V2"
    (seeds[2]["run"] / "evaluations/step_000025000.json").write_text(
        json.dumps({"step": 25_000, "deterministic": {"capture": {"capture_rate": 0.0, "collision_rate": 0.2}}}),
        encoding="utf-8",
    )
    assert mappo.final_gate()["decision"] == "HEALTHY_FAIL_TO_DISCRETE_COUNTERFACTUAL_Q"


def test_full_vxy_gate_override_is_explicit_and_stage2_only() -> None:
    valid_gate = {"checks": {"checkpoint": True}}
    assert full.can_override_stage_gate(full.STAGES[1], valid_gate, True) is True
    assert full.can_override_stage_gate(full.STAGES[1], valid_gate, False) is False
    assert full.can_override_stage_gate(full.STAGES[0], valid_gate, True) is False
    assert full.can_override_stage_gate(full.STAGES[2], valid_gate, True) is False
    assert full.can_override_stage_gate(
        full.STAGES[1], {"checks": {"checkpoint": False}}, True
    ) is False


def test_full_vxy_runtime_overlay_records_pretrained_and_external_resume(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("run_name: test\n", encoding="utf-8")
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"model")
    external_resume = tmp_path / "external" / "resume_latest.pt"
    destination = tmp_path / "runtime.yaml"

    assert full.runtime_config(
        base, destination, checkpoint, external_resume
    ) == destination
    payload = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert payload["extends"] == str(base.resolve())
    assert payload["pretrained"] == {
        "path": str(checkpoint.resolve()),
        "compatibility_mode": "shape_compatible",
    }
    assert payload["checkpointing"] == {
        "full_resume_path": str(external_resume.resolve())
    }


def test_full_vxy_stage3_finalizer_uses_explicit_paired_rollout_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    stage = dict(full.STAGES[2])
    paths = {
        "config": tmp_path / "stage3.yaml",
        "run": tmp_path / "run",
        "screen": tmp_path / "screen",
        "selection": tmp_path / "selection.json",
        "runtime": tmp_path / "runtime.yaml",
        "train_log": tmp_path / "train.log",
        "screen_log": tmp_path / "screen.log",
        "final_log": tmp_path / "final.log",
    }
    paths["config"].write_text("run_name: stage3\n", encoding="utf-8")
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(full, "stage_paths", lambda unused_stage: paths)
    monkeypatch.setattr(
        full,
        "runtime_config",
        lambda base, destination, pretrained, full_resume_path=None: base,
    )

    def fake_launch(name: str, command: list[str], unused_log: Path) -> str:
        captured[name] = command
        return "launched"

    monkeypatch.setattr(full, "launch_tmux", fake_launch)
    external_resume = tmp_path / "external" / "resume_latest.pt"
    external_resume.parent.mkdir(parents=True)
    external_resume.write_bytes(b"resume")
    full.ensure_stage(
        stage,
        tmp_path / "stage2_selected.pt",
        "cuda:0",
        "cuda:0",
        formal_gif_count=10,
        formal_workers=4,
        formal_seed=2026082301,
        full_resume_path=external_resume,
    )

    train_command = captured["cocap_vxy_full_s3_12p3e3obs_train"]
    assert train_command[train_command.index("--resume-path") + 1] == str(
        external_resume.resolve()
    )

    command = captured["cocap_vxy_full_s3_12p3e3obs_finalize"]
    assert command[command.index("--seed") + 1] == "2026082301"
    assert command[command.index("--gif-count") + 1] == "10"
    assert command[command.index("--workers") + 1] == "4"
    assert command[command.index("--capture-evaders") + 1] == "3"
    assert command[command.index("--coverage-max-steps") + 1] == "1800"
    assert command[command.index("--max-steps") + 1] == "2800"
