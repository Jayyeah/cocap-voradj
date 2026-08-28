#!/usr/bin/env python3
"""Aggregate final three-seed learning/evaluation evidence without cherry-picking."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/2026-08-28_small_step_ac"
FIELDS = (
    "normal_capture_rate", "stationary_capture_rate", "capture_rate", "collision_rate",
    "agent_agent_collision_rate", "visited_2plus_ring_rate", "visited_3plus_ring_rate",
    "max_3plus_ring_hold_steps", "mean_largest_angular_gap_rad", "mean_pairwise_angular_separation_min_rad",
    "mean_episode_length", "mean_episode_return", "mean_team_episode_return", "mean_distance_progress", "mean_action_norm", "mean_speed",
)


def load_last_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file(): return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else {}


def summarize(algorithm: str) -> dict[str, Any]:
    seeds = []
    for index in (1, 2, 3):
        run = ARTIFACT_ROOT / f"{algorithm}_seed{index}"
        evaluations = sorted((run / "evaluations").glob("step_*.json"))
        if not evaluations: continue
        evaluation = json.loads(evaluations[-1].read_text(encoding="utf-8"))
        learning = load_last_jsonl(run / "learning_metrics.jsonl")
        row = {"seed_index": index, "step": int(evaluation["step"]), "evaluation_file": str(evaluations[-1]),
               "learning": learning, "deterministic": evaluation["deterministic"]["capture"],
               "stochastic": evaluation["stochastic"]["capture"]}
        seeds.append(row)
    aggregate = {}
    for mode in ("deterministic", "stochastic"):
        aggregate[mode] = {}
        for field in FIELDS:
            values = [float(seed[mode][field]) for seed in seeds if field in seed[mode]]
            if values: aggregate[mode][field] = {"mean": statistics.fmean(values), "min": min(values), "max": max(values), "values": values}
        collision_names = sorted({name for seed in seeds for name in seed[mode].get("collision_type_episode_rates", {})})
        aggregate[mode]["collision_type_episode_rates"] = {
            name: {"mean": statistics.fmean([float(seed[mode].get("collision_type_episode_rates", {}).get(name, 0.0)) for seed in seeds]),
                   "values": [float(seed[mode].get("collision_type_episode_rates", {}).get(name, 0.0)) for seed in seeds]}
            for name in collision_names
        }
    return {"schema": "small-step-ac-three-seed-summary-v1", "algorithm": algorithm,
            "complete_seed_count": len(seeds), "no_cherry_pick": True, "seeds": seeds, "aggregate": aggregate}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("algorithm"); args = parser.parse_args()
    payload = summarize(args.algorithm); destination = ARTIFACT_ROOT / "summaries" / f"{args.algorithm}_three_seed.json"
    destination.parent.mkdir(parents=True, exist_ok=True); temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); temporary.replace(destination)
    print(destination); return 0


if __name__ == "__main__": raise SystemExit(main())
