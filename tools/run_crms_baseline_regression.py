#!/usr/bin/env python3
"""Run a small fixed-seed regression against an existing CR-MS IQN baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_continuous_baseline_manifest import BASELINE_DEFINITIONS, ROOT


CORE_FIELDS = (
    "capture_success_bool",
    "coverage_success_bool",
    "episode_success_bool",
    "collision_event",
    "coverage_settled_success",
    "coverage_cv015_success",
    "steps",
)
TRAJECTORY_SUMMARY_FIELDS = CORE_FIELDS + (
    "soft_oob_event",
    "coverage_geometric_success",
    "coverage_cv015_best_success",
    "final_voronoi_cv",
    "best_voronoi_cv",
    "episode_end_mean_speed",
    "episode_end_max_speed",
    "mean_path_length",
    "mean_start_end_displacement",
    "final_step_mean_displacement",
    "final_step_max_displacement",
    "rotation_sign_consistency",
    "mean_abs_angle_delta",
    "mean_centroid_move",
    "mean_radius_cv",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_formal_seed(formal_dir: Path) -> int:
    records = read_json(formal_dir / "capture" / "batch_summary.json").get("records", [])
    if not records:
        raise RuntimeError(f"no formal capture records: {formal_dir}")
    return int(records[0]["seed"])


def record_for_seed(path: Path, seed: int) -> dict[str, Any]:
    for record in read_json(path).get("records", []):
        if int(record.get("seed", -1)) == int(seed):
            return record
    raise RuntimeError(f"seed {seed} not found in {path}")


def trajectory_summary_hash(record: dict[str, Any]) -> str:
    payload = {
        field: record.get(field)
        for field in TRAJECTORY_SUMMARY_FIELDS
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_records(formal_dir: Path, output_root: Path, seed: int) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for scenario in ("capture", "coverage", "mix"):
        formal_path = formal_dir / scenario / "batch_summary.json"
        observed_path = output_root / scenario / "batch_summary.json"
        expected = record_for_seed(formal_path, seed)
        observed = record_for_seed(observed_path, seed)
        for field in CORE_FIELDS:
            if expected.get(field) != observed.get(field):
                mismatches.append(
                    {
                        "scenario": scenario,
                        "field": field,
                        "expected": expected.get(field),
                        "observed": observed.get(field),
                    }
                )
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", choices=sorted(BASELINE_DEFINITIONS), default="4v1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    definition = BASELINE_DEFINITIONS[args.line]
    formal_dir = ROOT / definition["formal_dir"]
    run_dir = ROOT / definition["run_dir"]
    config = ROOT / definition["config"]
    seed = int(args.seed) if args.seed is not None else first_formal_seed(formal_dir)
    capture_evaders = int(definition["capture_evaders"])
    output_root = (
        ROOT / args.output_root
        if args.output_root
        else ROOT / "artifacts/2026-08-04_continuous_marl_refactor/old_iqn_regression"
        / f"{args.line}_seed_{seed}_evaders_{capture_evaders}"
    )
    if output_root.exists() and not args.report_only:
        raise FileExistsError(f"refusing to overwrite existing regression: {output_root}")

    checkpoint = run_dir / "checkpoints" / f"step_{int(definition['selected_step'])}.pt"
    command = [
        sys.executable,
        str(ROOT / "tools/batch_rollouts_parallel.py"),
        "--config", str(config),
        "--checkpoint", str(checkpoint),
        "--output-root", str(output_root),
        "--episodes", str(int(args.episodes)),
        "--gif-count", "0",
        "--scenarios", "capture", "coverage", "mix",
        "--seed", str(seed),
        "--device", str(args.device),
        "--workers", str(int(args.workers)),
        "--max-steps", "2200",
        "--capture-max-steps", "1000",
        "--coverage-max-steps", "1200",
        "--capture-evaders", str(capture_evaders),
        "--max-gif-frames", "1",
    ]
    output_root.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.CompletedProcess(command, 0) if args.report_only else subprocess.run(
        command, cwd=ROOT, check=False
    )
    report: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "line": args.line,
        "seed": seed,
        "episodes": int(args.episodes),
        "command": command,
        "returncode": int(result.returncode),
        "formal_dir": str(formal_dir),
        "checkpoint": str(checkpoint),
        "output_root": str(output_root),
        "draw_trails_requested": False,
    }
    report_path = output_root / "regression_report.json"
    if result.returncode != 0:
        report["status"] = "rollout_failed"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return int(result.returncode)
    run_args = read_json(output_root / "run_args.json")
    report["draw_trails_observed"] = bool(run_args.get("draw_trails", False))
    report["all_summaries"] = read_json(output_root / "all_summaries.json")
    mismatches = compare_records(formal_dir, output_root, seed)
    report["core_field_mismatches"] = mismatches
    trajectory_hashes: dict[str, Any] = {}
    for scenario in ("capture", "coverage", "mix"):
        expected = record_for_seed(formal_dir / scenario / "batch_summary.json", seed)
        observed = record_for_seed(output_root / scenario / "batch_summary.json", seed)
        trajectory_hashes[scenario] = {
            "expected": trajectory_summary_hash(expected),
            "observed": trajectory_summary_hash(observed),
            "match": trajectory_summary_hash(expected) == trajectory_summary_hash(observed),
        }
    report["trajectory_summary_hashes"] = trajectory_hashes
    report["status"] = "passed" if not mismatches and not report["draw_trails_observed"] else "mismatch"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.no_strict:
        return 0
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
