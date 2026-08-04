#!/usr/bin/env python3
"""Run batch_voradj_rollouts in parallel worker processes and merge summaries.

This wrapper is intentionally process-based, not thread-based. The rollout
loop is dominated by Python/env/Voronoi work, so multiple child Python
processes are needed to use multiple CPU cores. Each worker loads its own
model copy on the requested device; for the first test we keep all workers on
cuda:1 because the model is small and observed GPU utilization is low.

Workers write under output-root/_workers to avoid filename races. The parent
process merges records into the same scenario/batch_summary.json and
all_summaries.json layout produced by batch_voradj_rollouts_20260711.py.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools.batch_rollouts import SCENARIOS, summarize


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def split_counts(total: int, workers: int) -> list[tuple[int, int]]:
    workers = max(1, int(workers))
    base = int(total) // workers
    extra = int(total) % workers
    chunks: list[tuple[int, int]] = []
    offset = 0
    for idx in range(workers):
        count = base + (1 if idx < extra else 0)
        if count <= 0:
            continue
        chunks.append((offset, count))
        offset += count
    return chunks


def copy_worker_files(worker_root: Path, output_root: Path, scenarios: list[str]) -> None:
    for scenario in scenarios:
        src_dir = worker_root / scenario
        if not src_dir.is_dir():
            continue
        dst_dir = output_root / scenario
        dst_dir.mkdir(parents=True, exist_ok=True)
        for path in src_dir.iterdir():
            if path.name == "batch_summary.json":
                continue
            if path.is_file():
                target = dst_dir / path.name
                if not target.exists():
                    shutil.copy2(path, target)


def merge_worker_outputs(output_root: Path, worker_roots: list[Path], scenarios: list[str]) -> dict[str, Any]:
    all_summaries: dict[str, Any] = {}
    for scenario in scenarios:
        records: list[dict[str, Any]] = []
        scenario_dir = output_root / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        for worker_root in worker_roots:
            payload_path = worker_root / scenario / "batch_summary.json"
            if not payload_path.is_file():
                continue
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            records.extend(payload.get("records", []))
        records.sort(key=lambda item: int(item.get("seed", 0)))
        summary = summarize(records)
        payload = {"scenario": scenario, "summary": summary, "records": records}
        (scenario_dir / "batch_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        all_summaries[scenario] = summary
    (output_root / "all_summaries.json").write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    return all_summaries


def resolve_input_path(path: str) -> Path:
    src = Path(path)
    if not src.is_absolute():
        src = ROOT / src
    return src


def backup_best_rollout_inputs(output_root: Path, config: str, checkpoint: str) -> dict[str, str]:
    if "best_20rollout10gif" not in output_root.parts:
        return {}
    backups: dict[str, str] = {}
    for label, raw_path in (("config_backup", config), ("checkpoint_backup", checkpoint)):
        src = resolve_input_path(raw_path)
        dst = output_root / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        backups[label] = str(dst)
    return backups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--gif-count", type=int, default=0)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["capture", "coverage", "mix"])
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2300)
    parser.add_argument("--capture-max-steps", type=int, default=700)
    parser.add_argument("--coverage-max-steps", type=int, default=1600)
    parser.add_argument("--capture-evaders", type=int, default=None, help="Override capture/mix evader count; defaults to env.num_evaders from the config.")
    parser.add_argument("--max-gif-frames", type=int, default=1)
    parser.add_argument("--frame-duration-ms", type=int, default=100)
    parser.add_argument("--draw-neighbor-edges", dest="draw_neighbor_edges", action="store_true", default=False)
    parser.add_argument("--no-neighbor-edges", dest="draw_neighbor_edges", action="store_false")
    parser.add_argument("--draw-trails", action="store_true", default=False)
    parser.add_argument("--draw-sensing-circles", dest="draw_sensing_circles", action="store_true", default=False)
    parser.add_argument("--no-sensing-circles", dest="draw_sensing_circles", action="store_false")
    args = parser.parse_args()
    parallel_wall_start = time.perf_counter()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    input_backups = backup_best_rollout_inputs(output_root, str(args.config), str(args.checkpoint))
    worker_parent = output_root / "_workers"
    worker_parent.mkdir(parents=True, exist_ok=True)

    chunks = split_counts(int(args.episodes), int(args.workers))
    if not chunks:
        raise RuntimeError("no worker chunks were produced")

    run_args = vars(args) | {
        "created_at": now(),
        "parallel_note": "Each worker is a child Python process with its own model/env. Worker output is isolated under _workers and merged after completion.",
        "input_backups": input_backups,
        "chunks": [{"worker": idx, "seed_offset": offset, "episodes": count} for idx, (offset, count) in enumerate(chunks)],
    }
    (output_root / "run_args.json").write_text(json.dumps(run_args, indent=2, ensure_ascii=False), encoding="utf-8")

    processes: list[tuple[int, Path, Path, Any, subprocess.Popen[bytes]]] = []
    gif_remaining = int(args.gif_count)
    for worker_idx, (offset, count) in enumerate(chunks):
        worker_root = worker_parent / f"worker_{worker_idx:02d}"
        worker_root.mkdir(parents=True, exist_ok=True)
        worker_gifs = min(gif_remaining, count)
        gif_remaining -= worker_gifs
        log_path = worker_root / "worker.log"
        cmd = [
            sys.executable,
            str(ROOT / "tools/batch_rollouts.py"),
            "--config",
            str(args.config),
            "--checkpoint",
            str(args.checkpoint),
            "--output-root",
            str(worker_root),
            "--episodes",
            str(count),
            "--gif-count",
            str(worker_gifs),
            "--scenarios",
            *list(args.scenarios),
            "--seed",
            str(int(args.seed) + offset),
            "--device",
            str(args.device),
            "--max-steps",
            str(int(args.max_steps)),
            "--capture-max-steps",
            str(int(args.capture_max_steps)),
            "--coverage-max-steps",
            str(int(args.coverage_max_steps)),
            "--max-gif-frames",
            str(int(args.max_gif_frames)),
            "--frame-duration-ms",
            str(int(args.frame_duration_ms)),
        ]
        if args.capture_evaders is not None:
            cmd.extend(["--capture-evaders", str(int(args.capture_evaders))])
        if bool(args.draw_neighbor_edges):
            cmd.append("--draw-neighbor-edges")
        if bool(args.draw_trails):
            cmd.append("--draw-trails")
        if bool(args.draw_sensing_circles):
            cmd.append("--draw-sensing-circles")
        handle = log_path.open("wb")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((worker_idx, worker_root, log_path, handle, proc))

    failures: list[dict[str, Any]] = []
    for worker_idx, worker_root, log_path, handle, proc in processes:
        rc = proc.wait()
        handle.close()
        if rc != 0:
            failures.append({"worker": worker_idx, "returncode": rc, "worker_root": str(worker_root), "log": str(log_path)})
    if failures:
        (output_root / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError(f"parallel rollout worker failures: {failures}")

    worker_roots = [item[1] for item in processes]
    copy_worker_files(worker_parent, output_root, [])
    for worker_root in worker_roots:
        copy_worker_files(worker_root, output_root, list(args.scenarios))
    all_summaries = merge_worker_outputs(output_root, worker_roots, list(args.scenarios))

    worker_timings: list[dict[str, Any]] = []
    for worker_idx, worker_root in enumerate(worker_roots):
        timing_path = worker_root / "timing.json"
        if timing_path.is_file():
            item = json.loads(timing_path.read_text(encoding="utf-8"))
            item["worker"] = worker_idx
            item["worker_root"] = str(worker_root)
            worker_timings.append(item)

    scenario_timings: dict[str, Any] = {}
    for scenario in list(args.scenarios):
        episodes = 0
        worker_scenario_seconds = 0.0
        worker_rollout_seconds = 0.0
        for item in worker_timings:
            scenario_item = (item.get("scenarios", {}) or {}).get(scenario, {}) or {}
            episode_count = int(scenario_item.get("episodes", 0))
            episodes += episode_count
            worker_scenario_seconds += float(scenario_item.get("wall_seconds", 0.0))
            worker_rollout_seconds += float(scenario_item.get("avg_rollout_wall_seconds_per_episode", 0.0)) * episode_count
        scenario_timings[scenario] = {
            "episodes": episodes,
            "sum_worker_scenario_wall_seconds": worker_scenario_seconds,
            "avg_worker_scenario_wall_seconds_per_episode": worker_scenario_seconds / max(episodes, 1),
            "avg_rollout_wall_seconds_per_episode": worker_rollout_seconds / max(episodes, 1),
        }

    total_wall_seconds = time.perf_counter() - parallel_wall_start
    total_rollout_episodes = int(args.episodes) * len(list(args.scenarios))
    sum_worker_total_wall_seconds = sum(float(item.get("total_wall_seconds", 0.0)) for item in worker_timings)
    sum_worker_scenario_seconds = sum(float(item.get("sum_worker_scenario_wall_seconds", 0.0)) for item in scenario_timings.values())
    timing_payload = {
        "created_at": now(),
        "device": str(args.device),
        "workers": len(worker_roots),
        "requested_workers": int(args.workers),
        "checkpoint": str(args.checkpoint),
        "output_root": str(output_root),
        "total_wall_seconds": total_wall_seconds,
        "total_rollout_episodes": total_rollout_episodes,
        "effective_wall_seconds_per_rollout_episode": total_wall_seconds / max(total_rollout_episodes, 1),
        "sum_worker_total_wall_seconds": sum_worker_total_wall_seconds,
        "avg_worker_total_wall_seconds_per_rollout_episode": sum_worker_total_wall_seconds / max(total_rollout_episodes, 1),
        "sum_worker_scenario_wall_seconds": sum_worker_scenario_seconds,
        "avg_worker_scenario_wall_seconds_per_rollout_episode": sum_worker_scenario_seconds / max(total_rollout_episodes, 1),
        "scenarios": scenario_timings,
        "workers_detail": worker_timings,
        "timing_note": "effective_wall_seconds_per_rollout_episode is the ETA throughput metric; avg_worker_* fields describe per-process rollout cost before parallel speedup.",
    }
    (output_root / "timing.json").write_text(json.dumps(timing_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    run_args["timing"] = timing_payload
    (output_root / "run_args.json").write_text(json.dumps(run_args, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "workers": len(worker_roots), "timing": timing_payload, "summaries": all_summaries}, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
