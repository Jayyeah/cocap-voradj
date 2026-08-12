#!/usr/bin/env python3
"""Run formal 20-rollout evaluation, then render representative GIFs by outcome."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cocap_voradj.training.trainer import load_config
from tools.rollout_voradj_visual import render_gif
from tools.run_continuous_ctde_training import _make_trainer
from tools.run_masac_rollout_gifs import _limit_cpu_threads, load_preset, rollout_episode


def number(record: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(record.get(key, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def choose(records: list[dict[str, Any]], count: int) -> list[tuple[dict[str, Any], str]]:
    selected: list[tuple[dict[str, Any], str]] = []
    used: set[int] = set()

    def add(record: dict[str, Any] | None, reason: str) -> None:
        if record is None:
            return
        seed = int(record["seed"])
        if seed not in used and len(selected) < count:
            used.add(seed)
            selected.append((record, reason))

    successful = [row for row in records if bool(row.get("episode_success")) or bool(row.get("captured"))]
    closest = min(records, key=lambda row: number(row, "min_min_distance", float("inf")), default=None)
    if successful:
        add(min(successful, key=lambda row: number(row, "length", float("inf"))), "successful_episode")
    else:
        add(closest, "closest_to_capture_no_success")
    collisions = [row for row in records if bool(row.get("collision_event"))]
    add(
        min(collisions, key=lambda row: number(row, "min_min_distance", float("inf")), default=None),
        "representative_collision",
    )
    failures = [
        row for row in records
        if not bool(row.get("episode_success")) and not bool(row.get("captured"))
    ]
    ordered_failures = sorted(
        failures,
        key=lambda row: (
            bool(row.get("collision_event")),
            number(row, "distance_progress", -float("inf")),
        ),
    )
    if ordered_failures:
        add(ordered_failures[len(ordered_failures) // 2], "typical_failure_geometry")
    add(
        max(records, key=lambda row: number(row, "distance_progress", -float("inf")), default=None),
        "best_distance_progress",
    )
    add(
        min(records, key=lambda row: number(row, "distance_progress", float("inf")), default=None),
        "worst_distance_progress",
    )
    for row in sorted(records, key=lambda item: int(item["seed"])):
        add(row, "diversity_fill")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--gif-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026081201)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    _limit_cpu_threads()
    args = parse_args()
    output_root = Path(args.output_root)
    command = [
        sys.executable,
        str(ROOT / "tools/run_masac_rollout_gifs.py"),
        "--preset-config", args.preset_config,
        "--config", args.config,
        "--checkpoint", args.checkpoint,
        "--output-root", str(output_root),
        "--device", "cpu",
        "--episodes", str(args.episodes),
        "--gif-count", "0",
        "--seed", str(args.seed),
        "--workers", str(args.workers),
    ]
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })
    subprocess.run(command, check=True, cwd=ROOT, env=env)

    config = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer = _make_trainer(config, "cpu")
    trainer.load_checkpoint(checkpoint, dict(saved["contract"]))
    trainer.actor.eval()
    preset = load_preset(args.preset_config)
    specs = {str(item["output_name"]): item for item in preset["scenarios"]}
    display = dict(preset["display"])
    selection_report: dict[str, Any] = {}

    for output_name, spec in specs.items():
        batch_path = output_root / output_name / "batch_summary.json"
        records = json.loads(batch_path.read_text(encoding="utf-8"))["records"]
        selections = choose(records, int(args.gif_count))
        rendered: list[dict[str, Any]] = []
        for record, reason in selections:
            seed = int(record["seed"])
            frames, rerun_summary = rollout_episode(trainer, config, spec, seed)
            gif_path = output_root / output_name / f"representative_{reason}_seed_{seed}.gif"
            render_record = render_gif(
                frames,
                gif_path,
                int(display["max_gif_frames"]),
                int(display.get("trail_window", 70)),
                int(display["frame_duration_ms"]),
                f"MASAC {preset['profile']} / {output_name} / {reason}",
                bool(display.get("draw_neighbor_edges", False)),
                bool(display.get("draw_trails", False)),
                bool(display.get("draw_sensing_circles", False)),
                bool(spec.get("draw_ce_targets", False)),
            )
            rerun_summary["representative_reason"] = reason
            rerun_summary["render"] = render_record
            rendered.append(rerun_summary)
            (output_root / output_name / f"representative_summary_{reason}_seed_{seed}.json").write_text(
                json.dumps(rerun_summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        selection_report[output_name] = rendered
    (output_root / "representative_selection.json").write_text(
        json.dumps(selection_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(output_root), "representative_gifs": selection_report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
