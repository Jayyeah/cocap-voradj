#!/usr/bin/env python3
"""Run deterministic formal rollouts while counting normal/stationary capture separately."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.continuous.formal_config import resolve_ladder_config
from tools.run_continuous_ctde_training import _make_trainer, _screen


def evaluate(
    *, config_path: Path, checkpoint: Path, seed: int,
    episodes: int, device: str,
) -> dict[str, Any]:
    config = resolve_ladder_config(config_path)
    trainer = _make_trainer(config, device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    trainer.load_checkpoint(checkpoint, payload["contract"])
    counts = {"normal": 0, "stationary": 0, "captured_without_type": 0}
    original = VorAdjEnv.episode_record

    def normal_only_record(self: VorAdjEnv, *args: Any, **kwargs: Any) -> dict[str, Any]:
        record = original(self, *args, **kwargs)
        if bool(record.get("captured", False)):
            types = [str(event.get("capture_type", "")) for event in self.last_capture_events]
            has_normal = any(value and value != "stationary" for value in types)
            has_stationary = any(value == "stationary" for value in types)
            if has_normal:
                counts["normal"] += 1
            if has_stationary:
                counts["stationary"] += 1
            if not types:
                counts["captured_without_type"] += 1
            # _screen's capture_rate becomes strictly normal-capture rate.
            if not has_normal:
                record["captured"] = False
                record["episode_success"] = False
        return record

    VorAdjEnv.episode_record = normal_only_record
    try:
        screening = _screen(
            trainer, config, seed, episodes=episodes, device=device,
            scenes=("capture",), max_steps=1000,
        )
    finally:
        VorAdjEnv.episode_record = original
    capture = dict(screening.get("capture", {}))
    normal_rate = float(capture.get("capture_rate", 0.0))
    mainly_normal = bool(counts["normal"] > counts["stationary"] and counts["captured_without_type"] == 0)
    return {
        "schema_version": 1,
        "kind": "cf3_formal_deterministic_normal_capture_gate",
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "seed": seed,
        "device": device,
        "episodes": episodes,
        "normal_capture_count": counts["normal"],
        "stationary_capture_count": counts["stationary"],
        "captured_without_type": counts["captured_without_type"],
        "normal_capture_rate": normal_rate,
        "success_not_mainly_stationary": mainly_normal,
        "stable_gate_passed": bool(normal_rate >= 0.20 and mainly_normal),
        "screening_normal_only": screening,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=2026081304)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate(
        config_path=Path(args.config), checkpoint=Path(args.checkpoint), seed=args.seed,
        episodes=args.episodes, device=args.device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
