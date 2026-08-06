"""Stage 2 25k analysis: compare policy eval against random/no-op baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--baselines", default=str(ROOT / "artifacts/2026-08-07_positive_feedback_ladder/stage2/stage2_baselines.json"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    report = load_json(args.report)
    baselines = load_json(args.baselines)
    eval20 = report.get("diagnostic_eval_400", {}).get("capture", {})
    records = eval20.get("records", [])
    capture_rate = float(eval20.get("capture_rate", 0.0))
    collision_rate = float(eval20.get("collision_rate", 0.0))
    avg_length = float(sum(item.get("length", 0) for item in records) / max(len(records), 1))
    min_distances = [item.get("min_min_distance") for item in records if item.get("min_min_distance") is not None]
    avg_min_distance = float(sum(min_distances) / max(len(min_distances), 1)) if min_distances else None
    initial_dists = [item.get("initial_min_distance") for item in records if item.get("initial_min_distance") is not None]
    avg_initial_distance = float(sum(initial_dists) / max(len(initial_dists), 1)) if initial_dists else None
    random = baselines["summary"]["random"]
    noop = baselines["summary"]["noop"]
    gate = {
        "capture_rate": capture_rate,
        "random_capture_rate": float(random["capture_rate"]),
        "noop_capture_rate": float(noop["capture_rate"]),
        "collision_rate": collision_rate,
        "random_collision_rate": float(random["collision_rate"]),
        "avg_length": avg_length,
        "avg_min_distance": avg_min_distance,
        "random_avg_min_distance": float(random["avg_min_distance"]),
        "noop_avg_min_distance": float(noop["avg_min_distance"]),
        "avg_initial_distance": avg_initial_distance,
        "capture_beats_random": capture_rate > float(random["capture_rate"]),
        "min_distance_beats_baselines": (
            avg_min_distance is not None
            and avg_min_distance < min(float(random["avg_min_distance"]), float(noop["avg_min_distance"]))
        ),
        "collision_controlled": collision_rate <= 0.3,
    }
    gate["status"] = (
        "PASS" if gate["capture_beats_random"] and gate["collision_controlled"] else
        "OPTIMISTIC_PARTIAL" if (gate["capture_beats_random"] or gate["min_distance_beats_baselines"]) else
        "FAIL"
    )
    payload = {
        "kind": "stage2_25k_analysis",
        "report": str(args.report),
        "gate": gate,
        "metrics_tail": report.get("metrics_history_tail", [])[-5:],
        "all_finite": report.get("all_finite"),
        "updates": report.get("updates"),
        "transition_count": report.get("transition_count"),
    }
    out = Path(args.out) if args.out else Path(args.report).with_name("stage2_25k_analysis.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
