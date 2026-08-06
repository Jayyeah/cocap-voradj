#!/usr/bin/env python3
"""Render selected CE8/CE9 checkpoint GIFs for strict/loose comparison."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "2026-07-30_ce_energy_curriculum"
OUTPUT_ROOT = ARTIFACT_ROOT / "selected_gifs_strict_loose"


CANDIDATES: list[dict[str, Any]] = [
    {
        "label": "ce8_step300k_strict_best",
        "line": "ce8_energy0005",
        "step": 300000,
        "run_dir": ROOT
        / "runs"
        / "ce8_delayed_energy0005_lrdrop200k_4p0e1obs_scratch_300k_20260730",
        "checkpoint": ROOT
        / "runs"
        / "ce8_delayed_energy0005_lrdrop200k_4p0e1obs_scratch_300k_20260730"
        / "checkpoints"
        / "step_300000.pt",
        "selection_reason": (
            "Strict-screening provisional best: success=1.00, collision=0.00, "
            "final geometry retention=0.95."
        ),
        "seeds": [
            {"seed": 2026072903, "tag": "slow_strict_success"},
            {"seed": 2026072919, "tag": "high_motion_borderline_retention"},
        ],
    },
    {
        "label": "ce9_step300k_low_speed_borderline",
        "line": "ce9_energy001",
        "step": 300000,
        "run_dir": ROOT
        / "runs"
        / "ce9_delayed_energy001_lrdrop200k_4p0e1obs_scratch_300k_20260730",
        "checkpoint": ROOT
        / "runs"
        / "ce9_delayed_energy001_lrdrop200k_4p0e1obs_scratch_300k_20260730"
        / "checkpoints"
        / "step_300000.pt",
        "selection_reason": (
            "Best low-speed near-eligible checkpoint: strict=0.95, loose=1.00, "
            "final geometry retention=0.947."
        ),
        "seeds": [
            {"seed": 2026072900, "tag": "static_near_miss"},
            {"seed": 2026072903, "tag": "static_strict_success"},
        ],
    },
    {
        "label": "ce9_step225k_loose_static",
        "line": "ce9_energy001",
        "step": 225000,
        "run_dir": ROOT
        / "runs"
        / "ce9_delayed_energy001_lrdrop200k_4p0e1obs_scratch_300k_20260730",
        "checkpoint": ROOT
        / "runs"
        / "ce9_delayed_energy001_lrdrop200k_4p0e1obs_scratch_300k_20260730"
        / "checkpoints"
        / "step_225000.pt",
        "selection_reason": (
            "Loose/static diagnostic checkpoint: strict=0.15, loose075=0.95, "
            "near-static=0.90."
        ),
        "seeds": [
            {"seed": 2026072900, "tag": "static_loose_fail"},
            {"seed": 2026072902, "tag": "strict_success_reference"},
        ],
    },
    {
        "label": "ce8_step250k_low_speed_loose",
        "line": "ce8_energy0005",
        "step": 250000,
        "run_dir": ROOT
        / "runs"
        / "ce8_delayed_energy0005_lrdrop200k_4p0e1obs_scratch_300k_20260730",
        "checkpoint": ROOT
        / "runs"
        / "ce8_delayed_energy0005_lrdrop200k_4p0e1obs_scratch_300k_20260730"
        / "checkpoints"
        / "step_250000.pt",
        "selection_reason": (
            "Low-speed loose checkpoint: strict=0.85, loose=1.00, "
            "near-static=0.90."
        ),
        "seeds": [
            {"seed": 2026072900, "tag": "static_loose_fail"},
            {"seed": 2026072916, "tag": "low_speed_strict_success"},
        ],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--post-success-steps", type=int, default=30)
    parser.add_argument("--max-gif-frames", type=int, default=300)
    parser.add_argument("--frame-duration-ms", type=int, default=90)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "device": str(args.device),
        "max_steps": int(args.max_steps),
        "post_success_steps": int(args.post_success_steps),
        "max_gif_frames": int(args.max_gif_frames),
        "frame_duration_ms": int(args.frame_duration_ms),
        "candidates": [],
    }

    for candidate in CANDIDATES:
        candidate_dir = output_root / candidate["label"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        manifest_candidate = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in candidate.items()
            if key != "seeds"
        }
        manifest_candidate["renders"] = []
        for seed_info in candidate["seeds"]:
            render_dir = candidate_dir / f"seed_{seed_info['seed']}_{seed_info['tag']}"
            cmd = [
                sys.executable,
                str(ROOT / "tools" / "visualize_ce_coverage.py"),
                "--run-dir",
                str(candidate["run_dir"]),
                "--checkpoint",
                str(candidate["checkpoint"]),
                "--output-root",
                str(render_dir),
                "--seed",
                str(seed_info["seed"]),
                "--device",
                str(args.device),
                "--max-steps",
                str(args.max_steps),
                "--post-success-steps",
                str(args.post_success_steps),
                "--max-gif-frames",
                str(args.max_gif_frames),
                "--frame-duration-ms",
                str(args.frame_duration_ms),
            ]
            print("RUN", " ".join(cmd), flush=True)
            completed = subprocess.run(
                cmd,
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log_path = render_dir / "visualize_stdout.log"
            render_dir.mkdir(parents=True, exist_ok=True)
            log_path.write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(
                    f"visualization failed for {candidate['label']} seed "
                    f"{seed_info['seed']}; see {log_path}"
                )
            manifest_candidate["renders"].append(
                {
                    **seed_info,
                    "output_dir": str(render_dir),
                    "gif": str(
                        render_dir / f"rollout_ce_coverage_seed_{seed_info['seed']}.gif"
                    ),
                    "stdout_log": str(log_path),
                }
            )
        manifest["candidates"].append(manifest_candidate)

    manifest_path = output_root / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "manifest": str(manifest_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
