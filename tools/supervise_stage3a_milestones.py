"""Wait for Stage3A 25k reports and auto-run the gate analyzer."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "artifacts/2026-08-07_positive_feedback_ladder/stage3a"
DEFAULT_BASELINES = DEFAULT_ROOT / "stage3a_baselines.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--baselines", default=str(DEFAULT_BASELINES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--config", default=str(ROOT / "configs/experiments/positive_feedback_ladder_20260807/stage3a_pure_ce_aw.yaml"))
    parser.add_argument("--prefix", default="stage3a_seed")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    pending = {seed: True for seed in args.seeds}
    while any(pending.values()):
        for seed in list(pending):
            if not pending[seed]:
                continue
            run_name = args.run_name if args.run_name else f"{args.prefix}{seed}_25k"
            run_dir = root / run_name
            report = run_dir / f"{run_name}_report.json"
            analysis = run_dir / f"{run_name}_analysis.json"
            eval20 = run_dir / f"{run_name}_eval20.json"
            if report.is_file() and not analysis.is_file():
                print(f"[supervisor] analyzing seed{seed}", flush=True)
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "tools/analyze_stage3a_25k.py"),
                        "--report",
                        str(report),
                        "--baselines",
                        args.baselines,
                        "--out",
                        str(analysis),
                    ],
                    check=False,
                )
            if report.is_file() and not eval20.is_file():
                checkpoint = run_dir / f"{run_name}_step25000.pt"
                if checkpoint.is_file():
                    print(f"[supervisor] eval20 seed{seed}", flush=True)
                    subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "tools/evaluate_ctde_formal.py"),
                            "--config",
                            args.config,
                            "--checkpoint",
                            str(checkpoint),
                            "--tag",
                            str(eval20),
                            "--episodes",
                            "20",
                            "--max-steps",
                            "400",
                            "--scenes",
                            "pure_ce",
                            "--seed",
                            str(2026080700 + seed),
                            "--device",
                            args.eval_device,
                        ],
                        check=False,
                    )
            if analysis.is_file():
                pending[seed] = False
        if args.once:
            break
        time.sleep(float(args.poll_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
