#!/usr/bin/env python3
"""Merge separately executed AW/VXY formal100 records into one paired audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from evaluate_vxy_stage2_cross_retention_20260903 import paired_comparison, summarize


def load_records(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"capture", "coverage", "mix"}:
        raise ValueError(f"unexpected scenario set: {path}")
    for rows in payload.values():
        for row in rows:
            if "first_detect_to_capture_steps" not in row:
                capture = row.get("capture_step")
                detection = row.get("first_direct_enemy_seen_step")
                row["first_detect_to_capture_steps"] = (
                    int(capture) - int(detection)
                    if capture is not None and detection is not None and int(capture) >= int(detection)
                    else None
                )
            if "two_plus_to_three_plus_steps" not in row:
                two = row.get("first_2plus_ring_step")
                three = row.get("first_3plus_ring_step")
                row["two_plus_to_three_plus_steps"] = (
                    int(three) - int(two)
                    if two is not None and three is not None and int(three) >= int(two)
                    else None
                )
    return payload


def paired_stage(aw_path: Path, vxy_path: Path) -> dict[str, Any]:
    aw = load_records(aw_path)
    vxy = load_records(vxy_path)
    result: dict[str, Any] = {
        "aw_records": str(aw_path.resolve()),
        "vxy_records": str(vxy_path.resolve()),
        "summaries": {},
        "paired": {},
    }
    for scenario in ("capture", "coverage", "mix"):
        aw_seeds = [int(row["seed"]) for row in aw[scenario]]
        vxy_seeds = [int(row["seed"]) for row in vxy[scenario]]
        if aw_seeds != vxy_seeds or len(aw_seeds) != 100:
            raise ValueError(f"formal100 paired seed mismatch in {scenario}")
        result["summaries"][scenario] = {
            "aw": summarize(aw[scenario]),
            "vxy": summarize(vxy[scenario]),
        }
        result["paired"][scenario] = paired_comparison(aw[scenario], vxy[scenario], scenario)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-aw", required=True)
    parser.add_argument("--stage2-vxy", required=True)
    parser.add_argument("--stage3-aw", required=True)
    parser.add_argument("--stage3-vxy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {
        "schema": "iqn-aw-vxy-matched-formal100-report-v1",
        "candidate_minus_baseline": "VXY minus Final-AW; latency/collision deltas below zero are better",
        "stage2": paired_stage(Path(args.stage2_aw), Path(args.stage2_vxy)),
        "stage3": paired_stage(Path(args.stage3_aw), Path(args.stage3_vxy)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": "complete", "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
