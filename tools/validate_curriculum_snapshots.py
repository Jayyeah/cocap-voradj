"""Validate curriculum snapshot restore into the new world-acceleration env."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cocap_voradj.training.continuous.curriculum_snapshots import restore_snapshot
from cocap_voradj.training.continuous.formal_config import resolve_formal_config, scene_config


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "configs/experiments/continuous_marl_20260804/p6_formal_central_masac_4v1.yaml"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "artifacts/2026-08-06_ctde_contract/curriculum_snapshots/snapshots.jsonl"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", default=str(ROOT / "artifacts/2026-08-06_ctde_contract/curriculum_snapshot_validation.json"))
    args = parser.parse_args()
    root_config = resolve_formal_config(FORMAL)
    rows = []
    with Path(args.dataset).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = rows[: args.limit]
    failures = []
    checked = 0
    for snapshot in rows:
        scene = str(snapshot["scene"])
        config = scene_config(root_config, "pure_ce" if scene == "pure_ce" else scene)
        try:
            env, diagnostics = restore_snapshot(snapshot, config, seed=2026080601, mode="geometry_reset")
        except Exception as exc:  # noqa: BLE001
            failures.append({"snapshot": snapshot.get("step"), "error": str(exc)})
            continue
        if len(env.pursuers) != snapshot["entity_counts"]["pursuers"]:
            failures.append({"snapshot": snapshot.get("step"), "error": "pursuer count mismatch"})
            continue
        if env.obstacles and len(env.obstacles) != snapshot["entity_counts"]["obstacles"]:
            failures.append({"snapshot": snapshot.get("step"), "error": "obstacle count mismatch"})
            continue
        if env.last_task_labels:
            if len(env.last_task_labels) != len(env.pursuers):
                failures.append({"snapshot": snapshot.get("step"), "error": "task label count mismatch"})
                continue
        checked += 1
    report = {
        "dataset": str(args.dataset),
        "snapshots_loaded": len(rows),
        "snapshots_checked": checked,
        "failures": failures,
        "ok": not failures and checked > 0,
        "restore_mode": "geometry_reset",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
