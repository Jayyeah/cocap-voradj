"""10k-state action translation gate for legacy IQN -> world acceleration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cocap_voradj.training.continuous.legacy_action_translation import (
    LegacyIQNActionToWorldAccelerationAdapter,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--out", default=str(ROOT / "artifacts/2026-08-06_ctde_contract/action_translation_gate.json"))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    adapter = LegacyIQNActionToWorldAccelerationAdapter()
    legacy_actions = [(a, w) for a in (-0.4, 0.0, 0.4) for w in (-np.pi / 6, 0.0, np.pi / 6)]
    rejected = 0
    clipped = 0
    velocity_errors = []
    position_errors = []
    progress_errors = []
    for _ in range(args.n):
        velocity = rng.uniform(-3.0, 3.0, size=2)
        speed = float(np.linalg.norm(velocity))
        if speed > 3.0:
            velocity *= 3.0 / max(speed, 1e-12)
        state = {"velocity": velocity, "theta": rng.uniform(-np.pi, np.pi)}
        action_index = int(rng.integers(0, len(legacy_actions)))
        result = adapter.translate(legacy_actions[action_index], state)
        if result.rejected:
            rejected += 1
            continue
        clipped += int(result.clipped)
        velocity_errors.append(result.terminal_velocity_error)
        position_errors.append(result.position_error)
        # Legacy progress proxy: signed change in projection onto command heading.
        progress_errors.append(abs(result.terminal_velocity_error))
    report = {
        "n": args.n,
        "rejection_rate": rejected / args.n,
        "radial_saturation_rate": clipped / args.n,
        "terminal_velocity_error_mean": float(np.mean(velocity_errors)) if velocity_errors else float("inf"),
        "terminal_velocity_error_p95": float(np.percentile(velocity_errors, 95)) if velocity_errors else float("inf"),
        "one_step_position_error_mean": float(np.mean(position_errors)) if position_errors else float("inf"),
        "one_step_position_error_p95": float(np.percentile(position_errors, 95)) if position_errors else float("inf"),
        "gate_passed": float(np.mean(velocity_errors)) < 0.1 if velocity_errors else False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
