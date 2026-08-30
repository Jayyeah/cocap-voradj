from __future__ import annotations

import json
from pathlib import Path

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
