"""Scripted oracle baseline for the Stage 2 simplest capture task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.continuous.formal_config import resolve_ladder_config, scene_config
from cocap_voradj.training.trainer import set_global_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/positive_feedback_ladder_20260807/stage2_simple_aw.yaml"


def run_episode(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    set_global_config(config)
    env = VorAdjEnv(config, seed=seed)
    env.reset()
    w_max = float(np.pi / 6.0)
    for _ in range(int(config["env"]["episode_max_length"])):
        pursuer = env.pursuers[0]
        if pursuer.deactivated or not env.evaders:
            break
        target = env.evaders[0]
        desired = float(np.arctan2(target.y - pursuer.y, target.x - pursuer.x))
        yaw_error = (desired - pursuer.theta + np.pi) % (2.0 * np.pi) - np.pi
        w = float(np.clip(3.0 * yaw_error, -w_max, w_max))
        a = 0.4 if pursuer.speed < 1.5 else (0.0 if pursuer.speed < 2.5 else -0.4)
        outcome = env.step([[a, w]], [None])
        if all(outcome.dones):
            break
    record = env.episode_record(task="capture")
    return {
        "seed": seed,
        "captured": bool(record.get("captured", False)),
        "episode_success": bool(record.get("episode_success", False)),
        "collision_event": bool(record.get("collision_event", False)),
        "length": int(record.get("length", 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(ROOT / "artifacts/2026-08-07_positive_feedback_ladder/stage2/stage2_oracle_baseline.json"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026080701)
    args = parser.parse_args()
    config = scene_config(resolve_ladder_config(args.config), "capture")
    records: List[Dict[str, Any]] = [
        run_episode(config, int(args.seed) + idx) for idx in range(int(args.episodes))
    ]
    summary = {
        "episodes": len(records),
        "capture_rate": float(np.mean([bool(item["captured"]) for item in records])),
        "collision_rate": float(np.mean([bool(item["collision_event"]) for item in records])),
        "avg_length": float(np.mean([item["length"] for item in records])),
    }
    payload = {"kind": "stage2_simple_aw_oracle", "summary": summary, "records": records}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
