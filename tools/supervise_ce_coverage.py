from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "ce0": ROOT / "runs/ce0_center_only_pbrs_4p0e1obs_scratch_300k_20260729",
    "ce1": ROOT / "runs/ce1_energy005_no_pbrs_4p0e1obs_scratch_300k_20260729",
    "ce2": ROOT / "runs/ce2_energy005_pbrs_4p0e1obs_scratch_300k_20260729",
    "ce3": ROOT / "runs/ce3_energy010_pbrs_4p0e1obs_scratch_300k_20260729",
}
CHECKPOINT_STEPS = (100_000, 200_000, 300_000)


def _evaluation_path(output_root: Path, line: str, step: int) -> Path:
    return output_root / line / f"step_{step}" / "evaluation_20rollout.json"


def _run_evaluation(
    line: str,
    run_dir: Path,
    step: int,
    output: Path,
    args: argparse.Namespace,
) -> None:
    checkpoint = run_dir / "checkpoints" / f"step_{step}.pt"
    command = [
        sys.executable,
        str(ROOT / "tools/evaluate_ce_coverage.py"),
        "--run-dir",
        str(run_dir),
        "--checkpoint",
        str(checkpoint),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--max-steps",
        str(args.max_steps),
        "--post-success-steps",
        str(args.post_success_steps),
        "--output",
        str(output),
    ]
    print(json.dumps({"event": "evaluation_start", "line": line, "step": step}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    print(json.dumps({"event": "evaluation_complete", "line": line, "step": step}), flush=True)


def _write_summary(output_root: Path) -> None:
    rows = []
    for line in RUNS:
        for step in CHECKPOINT_STEPS:
            path = _evaluation_path(output_root, line, step)
            if not path.is_file():
                continue
            payload: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"line": line, "step": step, **payload["summary"]})
    (output_root / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch and evaluate CE-P2 checkpoints.")
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "artifacts/2026-07-29_ce_coverage_p2_quickscreen"),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026072900)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--post-success-steps", type=int, default=30)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    while True:
        pending = []
        for line, run_dir in RUNS.items():
            for step in CHECKPOINT_STEPS:
                checkpoint = run_dir / "checkpoints" / f"step_{step}.pt"
                output = _evaluation_path(output_root, line, step)
                if output.is_file():
                    continue
                pending.append((line, run_dir, step, checkpoint, output))
        if not pending:
            _write_summary(output_root)
            print(json.dumps({"event": "all_evaluations_complete"}), flush=True)
            return

        progressed = False
        for line, run_dir, step, checkpoint, output in pending:
            if not checkpoint.is_file() or time.time() - checkpoint.stat().st_mtime < 10.0:
                continue
            try:
                _run_evaluation(line, run_dir, step, output, args)
                _write_summary(output_root)
                progressed = True
            except Exception as exc:
                print(
                    json.dumps(
                        {"event": "evaluation_retry", "line": line, "step": step, "error": repr(exc)}
                    ),
                    flush=True,
                )
        if not progressed:
            time.sleep(max(args.poll_seconds, 1))


if __name__ == "__main__":
    main()
