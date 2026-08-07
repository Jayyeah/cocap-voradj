"""Stage 3A 25k analysis: CE energy vs random/no-op baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--baselines", default=str(ROOT / "artifacts/2026-08-07_positive_feedback_ladder/stage3a/stage3a_baselines.json"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    baselines = json.loads(Path(args.baselines).read_text(encoding="utf-8"))
    eval_data = report.get("diagnostic_eval_400", {}).get("pure_ce", {})
    records = eval_data.get("records", [])
    initial = [item.get("initial_ce_energy") for item in records if item.get("initial_ce_energy") is not None]
    final = [item.get("final_ce_energy") for item in records if item.get("final_ce_energy") is not None]
    progress = [item.get("ce_energy_progress") for item in records if item.get("ce_energy_progress") is not None]
    initial_mean = float(sum(initial) / len(initial)) if initial else None
    final_mean = float(sum(final) / len(final)) if final else None
    progress_mean = float(sum(progress) / len(progress)) if progress else None
    relative_improvement = (
        float((initial_mean - final_mean) / max(abs(initial_mean), 1e-9))
        if initial_mean is not None and final_mean is not None and initial_mean != 0
        else None
    )
    random_b = baselines["summary"]["random"]
    noop_b = baselines["summary"]["noop"]
    collision_rate = float(eval_data.get("collision_rate", 0.0))
    gate = {
        "initial_ce_energy_mean": initial_mean,
        "final_ce_energy_mean": final_mean,
        "ce_energy_progress_mean": progress_mean,
        "relative_improvement": relative_improvement,
        "random_progress_mean": float(random_b["ce_energy_progress_mean"]),
        "noop_progress_mean": float(noop_b["ce_energy_progress_mean"]),
        "collision_rate": collision_rate,
        "random_collision_rate": float(random_b["collision_rate"]),
        "beats_random": progress_mean is not None and progress_mean > float(random_b["ce_energy_progress_mean"]),
        "beats_noop": progress_mean is not None and progress_mean > float(noop_b["ce_energy_progress_mean"]),
        "improvement_ge_15pct": relative_improvement is not None and relative_improvement >= 0.15,
        "collision_controlled": collision_rate <= 0.3,
    }
    gate["status"] = (
        "PASS"
        if gate["beats_random"] and gate["beats_noop"] and gate["collision_controlled"]
        else "OPTIMISTIC_PARTIAL"
        if gate["beats_random"] or gate["improvement_ge_15pct"]
        else "FAIL"
    )
    payload = {
        "kind": "stage3a_25k_analysis",
        "report": str(args.report),
        "gate": gate,
        "all_finite": report.get("all_finite"),
        "updates": report.get("updates"),
        "transition_count": report.get("transition_count"),
    }
    out = Path(args.out) if args.out else Path(args.report).with_name("stage3a_25k_analysis.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
