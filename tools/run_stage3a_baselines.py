"""Stage 3A pure-coverage random/no-op baselines for CE energy comparison."""
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
DEFAULT_CONFIG = ROOT / "configs/experiments/positive_feedback_ladder_20260807/stage3a_pure_ce_aw.yaml"


def _ce_energy(env: VorAdjEnv) -> float:
    positions = np.asarray([[p.x, p.y] for p in env.pursuers if not p.deactivated], dtype=float)
    evader_positions = np.asarray([[e.x, e.y] for e in env.evaders if not e.deactivated], dtype=float)
    if len(positions) == 0:
        return float("nan")
    try:
        values = env._voradj_coverage_potentials(positions, evader_positions)
        return float(np.mean(values))
    except Exception:
        return float("nan")


def run_episode(config: Dict[str, Any], seed: int, mode: str) -> Dict[str, Any]:
    set_global_config(config)
    env = VorAdjEnv(config, seed=seed)
    env.reset()
    rng = np.random.default_rng(seed)
    initial_energy = _ce_energy(env)
    energies: List[float] = []
    for _ in range(int(config["env"]["episode_max_length"])):
        if mode == "random":
            actions = [
                [float(rng.uniform(-0.4, 0.4)), float(rng.uniform(-np.pi / 6.0, np.pi / 6.0))]
                for _ in env.pursuers
            ]
        else:
            actions = [[0.0, 0.0]] * len(env.pursuers)
        outcome = env.step(actions, [None] * len(env.evaders))
        energy = _ce_energy(env)
        if np.isfinite(energy):
            energies.append(energy)
        if all(outcome.dones):
            break
    final_energy = energies[-1] if energies else initial_energy
    min_energy = float(np.nanmin(energies)) if energies else float("nan")
    record = env.episode_record(task="pure_ce")
    return {
        "seed": seed,
        "mode": mode,
        "initial_ce_energy": initial_energy,
        "final_ce_energy": final_energy,
        "min_ce_energy": min_energy,
        "ce_energy_progress": float(initial_energy - final_energy),
        "collision_event": bool(record.get("collision_event", False)),
        "coverage_strict_success": bool(record.get("coverage_strict_success", False)),
        "length": int(record.get("length", 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(ROOT / "artifacts/2026-08-07_positive_feedback_ladder/stage3a/stage3a_baselines.json"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026080701)
    parser.add_argument("--modes", nargs="+", choices=("random", "noop"), default=("random", "noop"))
    args = parser.parse_args()
    config = scene_config(resolve_ladder_config(args.config), "pure_ce")
    records: List[Dict[str, Any]] = []
    for mode in args.modes:
        for idx in range(int(args.episodes)):
            records.append(run_episode(config, int(args.seed) + idx, mode))
    summary: Dict[str, Any] = {}
    for mode in ("random", "noop"):
        subset = [item for item in records if item["mode"] == mode]
        summary[mode] = {
            "episodes": len(subset),
            "initial_ce_energy_mean": float(np.nanmean([item["initial_ce_energy"] for item in subset])),
            "final_ce_energy_mean": float(np.nanmean([item["final_ce_energy"] for item in subset])),
            "min_ce_energy_mean": float(np.nanmean([item["min_ce_energy"] for item in subset])),
            "ce_energy_progress_mean": float(np.nanmean([item["ce_energy_progress"] for item in subset])),
            "collision_rate": float(np.mean([bool(item["collision_event"]) for item in subset])),
            "avg_length": float(np.mean([item["length"] for item in subset])),
        }
    payload = {"kind": "stage3a_pure_ce_aw_baselines", "config": str(args.config), "summary": summary, "records": records}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
