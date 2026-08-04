#!/usr/bin/env python3
"""Select a stage-cap historical best and run its formal display evaluation.

The finalizer is independent from training and screening. It waits for the
final training checkpoint and every expected screening result, ranks the
checkpoints with capture as the primary objective, then runs deterministic
20-rollout/10-GIF capture/coverage/mix evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_SCRIPT = ROOT / "tools" / "batch_rollouts_parallel.py"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def scalar(summary: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(summary.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def metric_payload(summaries: dict[str, Any]) -> dict[str, float]:
    capture = summaries.get("capture", {}) or {}
    coverage = summaries.get("coverage", {}) or {}
    mix = summaries.get("mix", {}) or {}
    metrics = {
        "capture_success_rate": scalar(capture, "capture_success_rate"),
        "capture_collision_rate": scalar(capture, "collision_rate"),
        "capture_avg_steps": scalar(capture, "avg_steps", 9999.0),
        "coverage_ce_strict_rate": scalar(coverage, "coverage_success_rate"),
        "coverage_cv015_rate": scalar(coverage, "coverage_cv015_rate"),
        "coverage_collision_rate": scalar(coverage, "collision_rate"),
        "coverage_avg_final_voronoi_cv": scalar(coverage, "avg_final_voronoi_cv", 999.0),
        "coverage_avg_steps": scalar(coverage, "avg_steps", 9999.0),
        "mix_capture_rate": scalar(mix, "capture_success_rate"),
        "mix_ce_strict_rate": scalar(mix, "coverage_success_rate"),
        "mix_cv015_rate": scalar(mix, "coverage_cv015_rate"),
        "mix_collision_rate": scalar(mix, "collision_rate"),
        "mix_avg_final_voronoi_cv": scalar(mix, "avg_final_voronoi_cv", 999.0),
        "mix_avg_steps": scalar(mix, "avg_steps", 9999.0),
    }
    metrics["max_collision_rate"] = max(
        metrics["capture_collision_rate"],
        metrics["coverage_collision_rate"],
        metrics["mix_collision_rate"],
    )
    return metrics


def score_tuple(metrics: dict[str, float], step: int) -> list[float | int]:
    capture = float(metrics["capture_success_rate"])
    mix_capture = float(metrics["mix_capture_rate"])
    return [
        min(capture, mix_capture),
        (capture + mix_capture) / 2.0,
        -float(metrics["max_collision_rate"]),
        float(metrics["mix_ce_strict_rate"]),
        float(metrics["coverage_ce_strict_rate"]),
        float(metrics["mix_cv015_rate"]),
        float(metrics["coverage_cv015_rate"]),
        -float(metrics["mix_avg_final_voronoi_cv"]),
        -float(metrics["coverage_avg_final_voronoi_cv"]),
        -float(metrics["capture_avg_steps"]) / 1000.0,
        -float(metrics["mix_avg_steps"]) / 3000.0,
        int(step),
    ]


def expected_steps(total_steps: int, interval: int) -> list[int]:
    if total_steps <= 0 or interval <= 0 or total_steps % interval != 0:
        raise ValueError("total_steps must be a positive multiple of checkpoint_interval")
    return list(range(interval, total_steps + 1, interval))


def read_candidates(
    screening_root: Path,
    steps: list[int],
) -> tuple[list[dict[str, Any]], list[int]]:
    candidates: list[dict[str, Any]] = []
    missing: list[int] = []
    for step in steps:
        step_root = screening_root / f"step_{step}"
        summary_path = step_root / "all_summaries.json"
        if not (step_root / "DONE").is_file() or not summary_path.is_file():
            missing.append(step)
            continue
        summaries = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = metric_payload(summaries)
        candidates.append(
            {
                "step": step,
                "screening_root": str(step_root),
                "metrics": metrics,
                "score_tuple": score_tuple(metrics, step),
            }
        )
    return candidates, missing


def copy_input(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return str(dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--screening-root", required=True)
    parser.add_argument("--best-root", required=True)
    parser.add_argument("--line-label", required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--capture-evaders", type=int, required=True)
    parser.add_argument("--coverage-max-steps", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--formal-episodes", type=int, default=20)
    parser.add_argument("--gif-count", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument(
        "--draw-trails",
        action="store_true",
        default=False,
        help="Draw faded pursuer trails in formal GIFs (disabled by default).",
    )
    return parser.parse_args()


def build_rollout_command(
    args: argparse.Namespace,
    config: Path,
    checkpoint: Path,
    formal_root: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROLLOUT_SCRIPT),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--output-root",
        str(formal_root),
        "--episodes",
        str(int(args.formal_episodes)),
        "--gif-count",
        str(int(args.gif_count)),
        "--scenarios",
        "capture",
        "coverage",
        "mix",
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--workers",
        str(int(args.workers)),
        "--max-steps",
        str(int(args.max_steps)),
        "--capture-max-steps",
        "1000",
        "--coverage-max-steps",
        str(int(args.coverage_max_steps)),
        "--capture-evaders",
        str(int(args.capture_evaders)),
        "--max-gif-frames",
        "1000",
        "--frame-duration-ms",
        "100",
        "--draw-neighbor-edges",
        "--draw-sensing-circles",
    ]
    if bool(args.draw_trails):
        cmd.append("--draw-trails")
    return cmd


def main() -> int:
    args = parse_args()
    config = Path(args.config)
    run_dir = Path(args.run_dir)
    screening_root = Path(args.screening_root)
    best_root = Path(args.best_root)
    if not config.is_file():
        raise FileNotFoundError(f"config does not exist: {config}")

    steps = expected_steps(int(args.total_steps), int(args.checkpoint_interval))
    final_checkpoint = run_dir / "checkpoints" / f"final_step_{int(args.total_steps)}.pt"
    best_root.mkdir(parents=True, exist_ok=True)
    selection_path = best_root / f"{args.line_label}_selection.json"

    while True:
        candidates, missing = read_candidates(screening_root, steps)
        if final_checkpoint.is_file() and not missing:
            break
        print(
            json.dumps(
                {
                    "time": now(),
                    "state": "waiting",
                    "line": args.line_label,
                    "final_checkpoint_ready": final_checkpoint.is_file(),
                    "screened": len(candidates),
                    "expected": len(steps),
                    "next_missing_steps": missing[:5],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        time.sleep(max(1, int(args.poll_seconds)))

    ranking = sorted(candidates, key=lambda item: item["score_tuple"], reverse=True)
    selected = ranking[0]
    selected_step = int(selected["step"])
    checkpoint = run_dir / "checkpoints" / f"step_{selected_step}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    formal_root = best_root / f"{args.line_label}_step_{selected_step}"
    formal_root.mkdir(parents=True, exist_ok=True)
    effective_config = run_dir / "effective_config.yaml"
    backups = {
        "config": copy_input(config, formal_root / config.name),
        "checkpoint": copy_input(checkpoint, formal_root / checkpoint.name),
    }
    if effective_config.is_file():
        backups["effective_config"] = copy_input(
            effective_config,
            formal_root / "effective_config.yaml",
        )

    selection = {
        "selected_at": now(),
        "line": args.line_label,
        "selection_policy": (
            "lexicographic: min(capture,mix-capture), mean capture, lower max collision, "
            "mix/coverage CE strict, mix/coverage CV<0.15, lower final CV, fewer steps, "
            "later checkpoint"
        ),
        "selected_step": selected_step,
        "selected_checkpoint": str(checkpoint),
        "selected_screening": selected,
        "ranking": ranking,
        "formal_output_root": str(formal_root),
        "backups": backups,
    }
    selection_path.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (formal_root / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_path = formal_root / "all_summaries.json"
    done_path = formal_root / "FORMAL_DONE"
    if summary_path.is_file() and done_path.is_file():
        print(
            json.dumps(
                {"state": "already_complete", "selection": selection},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    cmd = build_rollout_command(args, config, checkpoint, formal_root)
    print(
        json.dumps(
            {"time": now(), "state": "formal_running", "command": cmd},
            ensure_ascii=False,
        ),
        flush=True,
    )
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"formal rollout failed with code {result.returncode}")
    done_path.write_text(f"{now()}\n", encoding="utf-8")
    print(
        json.dumps(
            {"time": now(), "state": "formal_complete", "selection": selection},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
