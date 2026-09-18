#!/usr/bin/env python3
"""Build authoritative B3 z diagnostics and final machine-readable audit."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file
from tools.run_forward_final_bridge_20260908 import atomic_json
ART = ROOT / "artifacts/2026-09-18_iqn_zstate_b3"
B1_ROLLOUT = ROOT / "artifacts/2026-09-17_iqn_cleanup_b1/rollout_report.json"


def quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def analyze_z(rollout: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    support: list[float] = []
    coverage: list[float] = []
    pure_coverage: list[float] = []
    neighbor: list[float] = []
    hop_values: list[int] = []
    direct_checks = direct_violations = 0
    all_high_steps = all_medium_steps = total_steps = 0
    max_high_run_without_direct = 0
    release_rows = []
    for record in rollout["records"]:
        if record["policy"] != "student":
            continue
        with np.load(record["trace"], allow_pickle=False) as trace:
            z = np.asarray(trace["z"], dtype=float)
            direct = np.asarray(trace["direct"], dtype=bool)
            source = np.asarray(trace["source"], dtype=np.uint8)
            hops = np.asarray(trace["hops"], dtype=np.int16)
            role = np.asarray(trace["role"], dtype=np.int8)
        support.extend(z[role == 2].tolist())
        coverage.extend(z[role == 3].tolist())
        if record["scene"] == "coverage":
            pure_coverage.extend(z[role >= 0].tolist())
        neighbor.extend(z[source == 3].tolist())
        hop_values.extend(hops[hops >= 0].astype(int).tolist())
        direct_checks += int(direct.sum())
        direct_violations += int(np.sum(direct & ~np.isclose(z, 1.0)))
        active = role >= 0
        all_high = np.asarray([
            bool(np.any(active[step]) and np.all(z[step][active[step]] >= 0.9))
            for step in range(len(z))
        ])
        all_medium = np.asarray([
            bool(np.any(active[step]) and np.all(z[step][active[step]] >= 0.5))
            for step in range(len(z))
        ])
        all_high_steps += int(all_high.sum())
        all_medium_steps += int(all_medium.sum())
        total_steps += len(z)
        run = 0
        for high, has_direct in zip(all_high, direct.any(axis=1)):
            run = run + 1 if high and not has_direct else 0
            max_high_run_without_direct = max(max_high_run_without_direct, run)
        if record["scene"] == "mixed" and record["captured"]:
            last_direct = max(np.flatnonzero(direct.any(axis=1)), default=-1)
            tail = z[last_direct + 1 :]
            first_lt_05 = next((i + 1 for i, value in enumerate(tail) if float(value.max()) < 0.5), None)
            first_lt_01 = next((i + 1 for i, value in enumerate(tail) if float(value.max()) < 0.1), None)
            required = math.ceil(math.log(0.1) / math.log(float(contract["lambda"])))
            right_censored = first_lt_01 is None and len(tail) < required
            release_rows.append({
                "seed": int(record["seed"]),
                "collision": bool(record["collision"]),
                "post_source_steps_observed": int(len(tail)),
                "steps_to_max_z_lt_0_5": first_lt_05,
                "steps_to_max_z_lt_0_1": first_lt_01,
                "right_censored_before_lt_0_1": bool(right_censored),
                "never_release_with_sufficient_horizon": bool(first_lt_01 is None and not right_censored),
                "last_observed_max_z": float(tail[-1].max()) if len(tail) else None,
            })
    releases = [row["steps_to_max_z_lt_0_1"] for row in release_rows if row["steps_to_max_z_lt_0_1"] is not None]
    return {
        "schema": "iqn-zstate-b3-z-diagnostics-v1",
        "direct_visible_invariant": {
            "checks": direct_checks,
            "violations": direct_violations,
            "z_equals_one_rate": 1.0 - direct_violations / max(direct_checks, 1),
        },
        "distributions": {
            "support_role_z": quantiles(support),
            "coverage_role_z": quantiles(coverage),
            "pure_coverage_z": quantiles(pure_coverage),
            "neighbor_dominant_z": quantiles(neighbor),
            "lineage_hops": quantiles([float(value) for value in hop_values]),
        },
        "propagation": {
            "neighbor_dominant_count": len(neighbor),
            "max_lineage_hop": max(hop_values, default=-1),
        },
        "swarm_saturation": {
            "all_active_z_ge_0_9_fraction": all_high_steps / max(total_steps, 1),
            "all_active_z_ge_0_5_fraction": all_medium_steps / max(total_steps, 1),
            "longest_all_z_ge_0_9_run_without_any_direct_source_steps": max_high_run_without_direct,
            "hidden_global_role_bit_diagnosis": "not_observed",
        },
        "decay": {
            "analytic_half_life_steps": float(contract["half_life_steps"]),
            "analytic_half_life_seconds": float(contract["half_life_seconds"]),
            "analytic_steps_to_lt_0_1": math.ceil(math.log(0.1) / math.log(float(contract["lambda"]))),
            "observed_uncensored_steps_to_max_z_lt_0_1": releases,
            "observed_uncensored_count": len(releases),
            "right_censored_by_early_terminal_count": sum(row["right_censored_before_lt_0_1"] for row in release_rows),
            "never_release_with_sufficient_horizon_count": sum(row["never_release_with_sufficient_horizon"] for row in release_rows),
            "episodes": release_rows,
        },
    }


def main() -> int:
    student = json.loads((ART / "student_report.json").read_text())
    rollout = json.loads((ART / "rollout_report.json").read_text())
    contract = json.loads((ART / "contract.json").read_text())
    dataset = json.loads((ART / "dataset" / "manifest.json").read_text())
    b1_rollout = json.loads(B1_ROLLOUT.read_text())
    zdiag = analyze_z(rollout, contract)
    atomic_json(ART / "z_diagnostics.json", zdiag)

    b3 = student["all_rows"]
    b1 = student["b1_all_rows"]
    offline = {
        "overall": {
            "b3": b3["all"]["action_agreement"],
            "b1": b1["action_agreement"],
            "delta": b3["all"]["action_agreement"] - b1["action_agreement"],
        }
    }
    for name, group in (
        ("pre_capture", "phase"),
        ("post_capture", "phase"),
        ("pure_coverage", "phase"),
        ("direct", "role"),
        ("support", "role"),
        ("coverage", "role"),
    ):
        previous = b1[group][name]["action_agreement"]
        current = b3[name]["action_agreement"]
        offline[name] = {"b3": current, "b1": previous, "delta": current - previous}
    b1_online = b1_rollout["student_on_student_trajectory_teacher_agreement"]["action_agreement"]
    b3_online = rollout["student_on_student_trajectory_teacher_agreement"]["action_agreement"]
    offline["on_policy"] = {"b3": b3_online, "b1": b1_online, "delta": b3_online - b1_online}

    audit = {
        "schema": "iqn-zstate-b3-final-audit-v2",
        "status": "complete",
        "classification": "B3_FAIL",
        "branch": "experiment/iqn-zstate-b3-20260918",
        "baseline_head": "9c8f603ea81cb8bb1104549e360058e8eaa41f7c",
        "contract": contract,
        "dataset": {
            "manifest": str(ART / "dataset" / "manifest.json"),
            "rows": dataset["row_count"],
            "coverage": dataset["coverage"],
            "targeted_category_coverage": dataset["targeted_category_coverage"],
            "old_replay_reused": False,
        },
        "student": {
            "checkpoint": str(ART / "student_zstate.pt"),
            "checkpoint_sha256": sha256_file(ART / "student_zstate.pt"),
            "offline_agreement_vs_b1": offline,
            "categorical_kl": b3["all"]["categorical_kl"],
            "q_ranking_agreement": b3["all"]["q_ranking_agreement"],
            "ring2_action_agreement": b3["ring2"]["action_agreement"],
            "ring3_action_agreement": b3["ring3"]["action_agreement"],
        },
        "rollout": {
            "episodes_per_scene_policy": rollout["episodes_per_scene_policy"],
            "summary": rollout["summary"],
            "matched_seed_base": rollout["matched_seed_base"],
        },
        "z_diagnostics": zdiag,
        "typical_traces": rollout["typical_trace_selection"],
        "rl_fine_tune_launched": False,
        "stop_reason": "BC gain is small overall and formal coverage/recovery collapse; short warm RL is not justified.",
        "local_voronoi_recommended_next": False,
        "local_voronoi_note": "Do not enter local Voronoi from this failed replacement; revisit the missing state representation first.",
    }
    atomic_json(ART / "B3_FINAL_AUDIT.json", audit)
    print(json.dumps({
        "classification": audit["classification"],
        "audit": str(ART / "B3_FINAL_AUDIT.json"),
        "z_diagnostics": str(ART / "z_diagnostics.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
