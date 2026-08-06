from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from supervise_ce_coverage import _run_evaluation


RUNS = {
    "ce8_energy0005": (
        ROOT / "runs/ce8_delayed_energy0005_lrdrop200k_4p0e1obs_scratch_300k_20260730"
    ),
    "ce9_energy001": (
        ROOT / "runs/ce9_delayed_energy001_lrdrop200k_4p0e1obs_scratch_300k_20260730"
    ),
}
CHECKPOINT_STEPS = (200_000, 225_000, 250_000, 275_000, 300_000)
SUCCESS_GATE = 0.90
COLLISION_GATE = 0.05
FINAL_GEOMETRY_RETENTION_GATE = 0.95


def _evaluation_path(
    output_root: Path, line: str, step: int, episodes: int
) -> Path:
    return (
        output_root
        / line
        / f"step_{step}"
        / f"evaluation_{episodes}rollout.json"
    )


def _finite(value: Any, fallback: float = math.inf) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return candidate if math.isfinite(candidate) else float(fallback)


def _row(
    line: str, step: int, payload: Dict[str, Any]
) -> Dict[str, Any]:
    summary = dict(payload["summary"])
    success_rate = float(summary.get("success_rate", 0.0))
    collision_rate = float(summary.get("collision_rate", 1.0))
    retention_rate = float(
        summary.get("post_success_final_geometry_retained_rate", 0.0)
    )
    active_agent_steps = _finite(
        summary.get("avg_objective_active_agent_steps"), fallback=0.0
    )
    speed_sq_sum = _finite(summary.get("avg_objective_speed_sq_sum"))
    speed_sq_per_agent_step = (
        speed_sq_sum / active_agent_steps
        if active_agent_steps > 0.0
        else math.inf
    )
    gates = {
        "success": success_rate >= SUCCESS_GATE,
        "collision": collision_rate <= COLLISION_GATE,
        "final_geometry_retention": (
            retention_rate >= FINAL_GEOMETRY_RETENTION_GATE
        ),
    }
    return {
        "line": line,
        "step": int(step),
        **summary,
        "avg_objective_speed_sq_per_agent_step": speed_sq_per_agent_step,
        "gates": gates,
        "eligible": bool(all(gates.values())),
    }


def _write_summary(output_root: Path, episodes: int) -> None:
    rows: List[Dict[str, Any]] = []
    for line in RUNS:
        for step in CHECKPOINT_STEPS:
            path = _evaluation_path(output_root, line, step, episodes)
            if not path.is_file():
                continue
            payload: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            rows.append(_row(line, step, payload))

    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            _finite(row.get("avg_objective_speed_sq_sum")),
            _finite(row.get("avg_success_step")),
            _finite(row.get("avg_post_success_mean_speed")),
        )
    )
    for rank, row in enumerate(eligible, start=1):
        row["eligible_rank"] = rank

    payload = {
        "protocol": {
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "episodes": int(episodes),
            "seed_start": 2026072900,
            "policy_quantile_mode": "fixed_midpoint_32",
            "max_steps": 900,
            "post_success_steps": 30,
            "gates": {
                "success_rate_min": SUCCESS_GATE,
                "collision_rate_max": COLLISION_GATE,
                "post_success_final_geometry_retained_rate_min": (
                    FINAL_GEOMETRY_RETENTION_GATE
                ),
            },
            "ranking": (
                "Eligible checkpoints only; lexicographic minimum of "
                "objective speed-squared sum, success step, then "
                "post-success mean speed."
            ),
        },
        "rows": rows,
        "eligible_ranking": eligible,
        "provisional_best": eligible[0] if eligible else None,
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch and evaluate CE8/CE9 delayed-energy checkpoints."
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "artifacts/2026-07-30_ce_energy_curriculum"),
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
                output = _evaluation_path(output_root, line, step, args.episodes)
                if not output.is_file():
                    pending.append((line, run_dir, step, checkpoint, output))
        if not pending:
            _write_summary(output_root, args.episodes)
            print(
                json.dumps(
                    {"event": "all_ce8_ce9_evaluations_complete"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return

        progressed = False
        for line, run_dir, step, checkpoint, output in pending:
            if (
                not checkpoint.is_file()
                or time.time() - checkpoint.stat().st_mtime < 10.0
            ):
                continue
            try:
                _run_evaluation(line, run_dir, step, output, args)
                _write_summary(output_root, args.episodes)
                progressed = True
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "event": "evaluation_retry",
                            "line": line,
                            "step": step,
                            "error": repr(exc),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        if not progressed:
            time.sleep(max(args.poll_seconds, 1))


if __name__ == "__main__":
    main()
