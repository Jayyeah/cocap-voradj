from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from supervise_ce_coverage import _run_evaluation


RUNS = {
    "ce4_seed1544": ROOT / "runs/ce4_ce0_seed1544_4p0e1obs_scratch_300k_20260729",
    "ce5_seed1545": ROOT / "runs/ce5_ce0_seed1545_4p0e1obs_scratch_300k_20260729",
    "ce6_lrdrop": ROOT / "runs/ce6_ce0_lrdrop200k_4p0e1obs_scratch_300k_20260729",
    "ce7_energy0002": ROOT / "runs/ce7_energy0002_pbrs_4p0e1obs_scratch_300k_20260729",
}
CHECKPOINT_STEPS = (150_000, 200_000, 250_000, 300_000)


def _evaluation_path(output_root: Path, line: str, step: int, episodes: int) -> Path:
    return output_root / line / f"step_{step}" / f"evaluation_{episodes}rollout.json"


def _write_summary(output_root: Path, episodes: int) -> None:
    rows = []
    for line in RUNS:
        for step in CHECKPOINT_STEPS:
            path = _evaluation_path(output_root, line, step, episodes)
            if not path.is_file():
                continue
            payload: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"line": line, "step": step, **payload["summary"]})
    (output_root / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch CE follow-up checkpoints.")
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "artifacts/2026-07-29_ce_coverage_p3_followup"),
    )
    parser.add_argument("--episodes", type=int, default=10)
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
                output = _evaluation_path(output_root, line, step, args.episodes)
                checkpoint = run_dir / "checkpoints" / f"step_{step}.pt"
                if not output.is_file():
                    pending.append((line, run_dir, step, checkpoint, output))
        if not pending:
            _write_summary(output_root, args.episodes)
            print(json.dumps({"event": "all_followup_evaluations_complete"}), flush=True)
            return

        progressed = False
        for line, run_dir, step, checkpoint, output in pending:
            if not checkpoint.is_file() or time.time() - checkpoint.stat().st_mtime < 10.0:
                continue
            try:
                _run_evaluation(line, run_dir, step, output, args)
                _write_summary(output_root, args.episodes)
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
