#!/usr/bin/env python3
"""Run ZoneDemo rollout configs into one organized artifact batch."""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]


def expand_configs(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = sorted(Path(match) for match in glob.glob(pattern))
        if not matches:
            candidate = Path(pattern)
            matches = [candidate] if candidate.exists() else []
        for path in matches:
            resolved = path if path.is_absolute() else REPO_ROOT / path
            resolved = resolved.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)
    return paths


def case_name(config_path: Path) -> str:
    name = config_path.stem
    for prefix in ("zone_v0_", "zonedemo_v0_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True, help="Config files or glob patterns.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint used by every rollout in the batch.")
    parser.add_argument("--batch-name", required=True, help="Subdirectory under artifacts/zonedemo_v0.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="One or more rollout seeds.")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--capture-evaders", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--max-gif-frames", type=int, default=220)
    parser.add_argument("--frame-duration-ms", type=int, default=80)
    parser.add_argument("--scenario", default="mix", choices=["mix", "capture", "coverage", "ab", "ba"])
    parser.add_argument("--artifact-root", default="artifacts/zonedemo_v0")
    parser.add_argument("--draw-neighbor-edges", action="store_true")
    parser.add_argument("--draw-trails", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configs = expand_configs(args.configs)
    if not configs:
        raise SystemExit("No config files matched --configs")

    checkpoint = Path(args.checkpoint)
    checkpoint_arg = rel(checkpoint if checkpoint.is_absolute() else REPO_ROOT / checkpoint)
    batch_root = REPO_ROOT / args.artifact_root / args.batch_name
    if not args.dry_run:
        batch_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for config_path in configs:
            output_root = batch_root / f"{case_name(config_path)}_seed{int(seed)}"
            cmd = [
                sys.executable,
                "tools/rollout_voradj_visual.py",
                "--config",
                rel(config_path),
                "--checkpoint",
                checkpoint_arg,
                "--scenario",
                str(args.scenario),
                "--capture-evaders",
                str(args.capture_evaders),
                "--seed",
                str(int(seed)),
                "--device",
                str(args.device),
                "--output-root",
                rel(output_root),
                "--max-steps",
                str(args.max_steps),
                "--max-gif-frames",
                str(args.max_gif_frames),
                "--frame-duration-ms",
                str(args.frame_duration_ms),
            ]
            cmd.append("--draw-neighbor-edges" if args.draw_neighbor_edges else "--no-neighbor-edges")
            if args.draw_trails:
                cmd.append("--draw-trails")
            print(" ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
