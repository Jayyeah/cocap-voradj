"""Run a deterministic Voronoi-centroid oracle for the continuous P1 contract.

This is a read-only diagnostic: it does not train, alter rewards, or change
the environment. It checks that the continuous action contract can reach the
pure-CE geometry without collisions under a scripted local target policy.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from cocap_voradj.config import ConfigManager
from cocap_voradj.dynamics.continuous_action import AccelerationActionAdapter, world_to_body
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.trainer import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/continuous_marl_20260804/p5_central_critic_contract_4v1.yaml"
DEFAULT_OUTPUT = ROOT / "artifacts/2026-08-04_continuous_marl_refactor/diagnostics/continuous_oracle_pure_ce.json"


def _config(a_max: float, max_steps: int) -> dict:
    config = copy.deepcopy(load_config(str(CONFIG_PATH)))
    config["env"]["num_pursuers"] = 4
    config["env"]["num_evaders"] = 0
    config["env"]["episode_max_length"] = int(max_steps)
    config["env"]["num_obstacles"] = 0
    config["a_max"] = float(a_max)
    config.setdefault("pursuer", {})["a_max"] = float(a_max)
    ConfigManager.get_instance().update_config(config)
    return config


def run(seed: int, a_max: float, max_steps: int) -> dict:
    config = _config(a_max, max_steps)
    env = VorAdjEnv(config, seed=int(seed))
    env.reset()
    adapter = AccelerationActionAdapter(a_max=float(a_max), decision_dt=0.5)
    for step in range(int(max_steps)):
        coverage = env._coverage_voronoi_map()
        actions = []
        for index, pursuer in enumerate(env.pursuers):
            target = np.asarray(coverage["centroids"][("pursuer", index)], dtype=float)
            delta = target - np.asarray([pursuer.x, pursuer.y], dtype=float)
            distance = float(np.linalg.norm(delta))
            world_velocity = np.zeros(2, dtype=float) if distance < 0.5 else delta / distance * min(3.0, distance)
            world_acceleration = (world_velocity - np.asarray(pursuer.velocity, dtype=float)) / 0.5
            desired_body = world_to_body(world_acceleration, float(pursuer.theta))
            action_body, _ = adapter.validate_with_diagnostics(desired_body)
            actions.append(action_body)
        outcome = env.step(np.asarray(actions, dtype=np.float32).tolist(), [None] * len(env.evaders))
        if all(outcome.dones):
            break
    record = env.episode_record(task="pure_ce")
    return {
        "seed": int(seed),
        "a_max": float(a_max),
        "max_steps": int(max_steps),
        "length": int(record["length"]),
        "episode_success": bool(record["episode_success"]),
        "collision_event": bool(record["collision_event"]),
        "coverage_strict_success": bool(record["coverage_strict_success"]),
        "coverage_cv015_success": bool(record["coverage_cv015_success"]),
        "coverage_ce_center_rms": float(record["coverage_ce_center_rms"]),
        "coverage_ce_center_max": float(record["coverage_ce_center_max"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026080410, 2026080411])
    parser.add_argument("--a-max", type=float, choices=(0.4, 0.8, 1.6), default=0.8)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    results = [run(seed, args.a_max, args.max_steps) for seed in args.seeds]
    payload = {"kind": "continuous_oracle_diagnostic", "config": str(CONFIG_PATH.relative_to(ROOT)), "results": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if all(item["coverage_strict_success"] and not item["collision_event"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
