"""Stage 4A 25k analysis: capture rate / min-distance vs baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--baselines", default=str(ROOT / "artifacts/2026-08-07_positive_feedback_ladder/stage4a/stage4a_baselines.json"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    baselines = json.loads(Path(args.baselines).read_text(encoding="utf-8"))
    eval_data = report.get("diagnostic_eval_400", {}).get("capture", {})
    records = eval_data.get("records", [])
    capture_rate = float(eval_data.get("capture_rate", 0.0))
    collision_rate = float(eval_data.get("collision_rate", 0.0))
    min_dists = [item.get("min_min_distance") for item in records if item.get("min_min_distance") is not None]
    avg_min_distance = float(sum(min_dists) / len(min_dists)) if min_dists else None
    random_b = baselines["summary"]["random"]
    noop_b = baselines["summary"]["noop"]
    gate = {
        "capture_rate": capture_rate,
        "random_capture_rate": float(random_b["capture_rate"]),
        "noop_capture_rate": float(noop_b["capture_rate"]),
        "collision_rate": collision_rate,
        "random_collision_rate": float(random_b["collision_rate"]),
        "avg_min_distance": avg_min_distance,
        "random_avg_min_distance": float(random_b["avg_min_distance"]),
        "noop_avg_min_distance": float(noop_b["avg_min_distance"]),
        "capture_beats_random": capture_rate > float(random_b["capture_rate"]),
        "min_distance_beats_baselines": (
            avg_min_distance is not None
            and avg_min_distance < min(float(random_b["avg_min_distance"]), float(noop_b["avg_min_distance"]))
        ),
        "collision_controlled": collision_rate <= 0.5,
    }
    gate["status"] = (
        "PASS"
        if (gate["capture_beats_random"] or gate["min_distance_beats_baselines"]) and gate["collision_controlled"]
        else "OPTIMISTIC_PARTIAL"
        if gate["capture_rate"] > 0 or avg_min_distance is not None
        else "FAIL"
    )
    payload = {
        "kind": "stage4a_25k_analysis",
        "report": str(args.report),
        "gate": gate,
        "all_finite": report.get("all_finite"),
        "updates": report.get("updates"),
        "transition_count": report.get("transition_count"),
    }
    out = Path(args.out) if args.out else Path(args.report).with_name("stage4a_25k_analysis.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
