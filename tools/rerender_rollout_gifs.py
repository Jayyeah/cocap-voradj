#!/usr/bin/env python3
"""Re-render existing rollout GIFs from saved episode JSON files.

This does not load a checkpoint or run an environment. Seeds, frames, metrics,
and output paths remain unchanged. Faded trails are disabled unless the caller
explicitly passes ``--draw-trails``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def discover_episode_files(root: Path) -> list[Path]:
    episodes: list[Path] = []
    for stage_root in root.iterdir():
        if not stage_root.is_dir():
            continue
        for scenario in ("capture", "coverage", "mix"):
            episodes.extend((stage_root / scenario).glob("episode_*_seed_*.json"))
    return sorted(path for path in episodes if "_workers" not in path.parts)


def stage_run_args(episode_path: Path) -> dict[str, Any]:
    run_args_path = episode_path.parent.parent / "run_args.json"
    if not run_args_path.is_file():
        return {}
    return json.loads(run_args_path.read_text(encoding="utf-8"))


def worker_copies(episode_path: Path, filename: str) -> list[Path]:
    stage_root = episode_path.parent.parent
    scenario = episode_path.parent.name
    return sorted((stage_root / "_workers").glob(f"worker_*/{scenario}/{filename}"))


def update_run_args_metadata(root: Path, draw_trails: bool, rerendered_at: str) -> int:
    updated = 0
    for path in sorted(root.glob("*/run_args.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "draw_trails" not in payload:
            continue
        payload["draw_trails"] = bool(draw_trails)
        payload["gif_rerender"] = {
            "rerendered_at": rerendered_at,
            "mode": "render_saved_frames_only",
            "rollouts_rerun": False,
            "draw_trails": bool(draw_trails),
            "seeds_and_output_paths_preserved": True,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        updated += 1
    return updated


def rerender_one(raw_path: str, draw_trails: bool) -> dict[str, Any]:
    from tools.rollout_voradj_visual import render_gif

    episode_path = Path(raw_path)
    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    summary = dict(payload.get("summary", {}) or {})
    frames = list(payload.get("frames", []) or [])
    if not frames:
        raise ValueError(f"episode has no frames: {episode_path}")

    scenario = str(summary.get("scenario") or episode_path.parent.name)
    seed = int(summary.get("seed"))
    gif_path = episode_path.parent / f"rollout_{scenario}_seed_{seed}.gif"
    summary_path = episode_path.parent / f"summary_{scenario}_seed_{seed}.json"
    run_args = stage_run_args(episode_path)
    max_frames = int(summary.get("rendered_frames") or run_args.get("max_gif_frames") or 1000)
    frame_duration_ms = int(run_args.get("frame_duration_ms", 100))
    draw_neighbor_edges = bool(summary.get("draw_neighbor_edges", run_args.get("draw_neighbor_edges", False)))
    draw_sensing_circles = bool(summary.get("draw_sensing_circles", run_args.get("draw_sensing_circles", False)))
    draw_ce_targets = bool(summary.get("draw_ce_targets", scenario == "coverage"))

    temp_gif = gif_path.with_name(f".{gif_path.stem}.rerender.tmp.gif")
    render_record = render_gif(
        frames,
        temp_gif,
        max_frames,
        70,
        frame_duration_ms,
        scenario,
        draw_neighbor_edges,
        bool(draw_trails),
        draw_sensing_circles,
        draw_ce_targets,
    )
    os.replace(temp_gif, gif_path)
    render_record["gif"] = str(gif_path)

    summary.update(render_record)
    payload["summary"] = summary
    episode_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    synced = 0
    for source in (gif_path, episode_path, summary_path):
        for destination in worker_copies(episode_path, source.name):
            shutil.copy2(source, destination)
            synced += 1
    return {
        "episode": str(episode_path),
        "gif": str(gif_path),
        "scenario": scenario,
        "seed": seed,
        "frames": int(render_record["rendered_frames"]),
        "draw_trails": bool(draw_trails),
        "synced_worker_files": synced,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--draw-trails",
        action="store_true",
        default=False,
        help="Draw faded pursuer trails (disabled by default).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    episodes = discover_episode_files(root)
    if not episodes:
        raise RuntimeError(f"no saved rollout episodes found under {root}")

    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(rerender_one, str(path), bool(args.draw_trails)): path
            for path in episodes
        }
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(
                json.dumps(
                    {"completed": index, "total": len(episodes), **record},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    records.sort(key=lambda item: (item["gif"], item["seed"]))
    created_at = now()
    run_args_updated = update_run_args_metadata(root, bool(args.draw_trails), created_at)
    manifest = {
        "created_at": created_at,
        "root": str(root),
        "mode": "render_saved_frames_only",
        "rollouts_rerun": False,
        "draw_trails": bool(args.draw_trails),
        "gif_count": len(records),
        "run_args_updated": run_args_updated,
        "records": records,
    }
    manifest_path = root / "gif_rerender_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "gif_count": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
