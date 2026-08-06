"""Stage 2 random/no-op baselines for the simplest (a,w) capture task."""
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


def run_episode(config: Dict[str, Any], seed: int, mode: str) -> Dict[str, Any]:
    set_global_config(config)
    env = VorAdjEnv(config, seed=seed)
    env.reset()
    rng = np.random.default_rng(seed)
    min_distance = float("inf")
    for _ in range(int(config["env"]["episode_max_length"])):
        if mode == "random":
            action = [float(rng.uniform(-0.4, 0.4)), float(rng.uniform(-np.pi / 6.0, np.pi / 6.0))]
        else:
            action = [0.0, 0.0]
        positions = np.asarray([[p.x, p.y] for p in env.pursuers if not p.deactivated], dtype=float)
        if len(positions) and env.evaders:
            target = np.asarray([env.evaders[0].x, env.evaders[0].y], dtype=float)
            min_distance = min(min_distance, float(np.min(np.linalg.norm(positions - target, axis=1))))
        outcome = env.step([action], [None])
        if all(outcome.dones):
            break
    record = env.episode_record(task="capture")
    return {
        "seed": seed,
        "mode": mode,
        "captured": bool(record.get("captured", False)),
        "episode_success": bool(record.get("episode_success", False)),
        "collision_event": bool(record.get("collision_event", False)),
        "length": int(record.get("length", 0)),
        "min_distance": float(min_distance) if np.isfinite(min_distance) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(ROOT / "artifacts/2026-08-07_positive_feedback_ladder/stage2/stage2_baselines.json"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026080701)
    args = parser.parse_args()

    config = scene_config(resolve_ladder_config(args.config), "capture")
    records: List[Dict[str, Any]] = []
    for mode in ("random", "noop"):
        for idx in range(int(args.episodes)):
            records.append(run_episode(config, int(args.seed) + idx, mode))

    summary: Dict[str, Any] = {}
    for mode in ("random", "noop"):
        subset = [item for item in records if item["mode"] == mode]
        summary[mode] = {
            "episodes": len(subset),
            "capture_rate": float(np.mean([bool(item["captured"]) for item in subset])),
            "collision_rate": float(np.mean([bool(item["collision_event"]) for item in subset])),
            "avg_length": float(np.mean([item["length"] for item in subset])),
            "avg_min_distance": float(np.mean([item["min_distance"] for item in subset if item["min_distance"] is not None])),
        }
    payload = {"kind": "stage2_simple_aw_baselines", "config": str(args.config), "summary": summary, "records": records}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
