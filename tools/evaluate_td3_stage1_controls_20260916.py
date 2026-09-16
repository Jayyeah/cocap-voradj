#!/usr/bin/env python3
"""Matched random/no-op controls for the local-TD3 Stage-1 gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from cocap_voradj.training.td3_stage1_contract import A_MAX, W_MAX, make_td3_env
from cocap_voradj.training.trainer import load_config
from tools import preflight_forward_final_scratch_20260914 as preflight
from tools.forward_final_single_task_20260915 import Telemetry, summarize
from tools.train_td3_local_aw_stage1_20260916 import (
    _capture_geometry,
    atomic_json,
    evader_actions,
)


CONFIG = ROOT / "configs/experiments/td3_local_aw_stage1_20260916/common.yaml"
SCHEMA = "td3-local-aw-stage1-controls-v1"


def evaluate_task(task: str, episodes: int, max_steps: int) -> dict:
    config = load_config(str(CONFIG))
    seed_base = int(config["evaluation"][f"{task}_seed_base"])
    by_mode = {}
    for mode in ("noop", "uniform_random"):
        records = []
        started = time.monotonic()
        for episode in range(int(episodes)):
            seed = seed_base + episode
            env, observations = make_td3_env(task, seed)
            env.total_steps = 2_000_000
            telemetry = Telemetry(env, task)
            apf_agents = [preflight.ApfAgent(evader.a, evader.w) for evader in env.evaders]
            rng = np.random.default_rng(seed + 1_000_003)
            actions_all = []
            minimum_distance = float("inf")
            ring_run = {2: 0, 3: 0}
            ring_max = {2: 0, 3: 0}
            geometric_step = None
            settled_step = None
            for step in range(int(max_steps)):
                active = [index for index, observation in enumerate(observations) if observation is not None]
                if mode == "noop":
                    selected = np.zeros((len(active), 2), dtype=np.float32)
                else:
                    selected = np.stack(
                        [
                            rng.uniform(-A_MAX, A_MAX, size=len(active)),
                            rng.uniform(-W_MAX, W_MAX, size=len(active)),
                        ],
                        axis=-1,
                    ).astype(np.float32)
                command = [None] * len(observations)
                for agent_id, action in zip(active, selected):
                    command[agent_id] = action
                outcome = env.step(command, evader_actions(env, apf_agents))
                telemetry.observe(env, outcome, active)
                actions_all.append(selected)
                geometry = _capture_geometry(env)
                if geometry["min_target_distance"] is not None:
                    minimum_distance = min(minimum_distance, float(geometry["min_target_distance"]))
                for size in (2, 3):
                    ring_run[size] = ring_run[size] + 1 if geometry["inside_capture_radius"] >= size else 0
                    ring_max[size] = max(ring_max[size], ring_run[size])
                if geometric_step is None and bool(getattr(env, "coverage_geometric_success", False)):
                    geometric_step = step + 1
                if settled_step is None and bool(getattr(env, "coverage_settled_success", False)):
                    settled_step = step + 1
                observations = list(outcome.observations)
                if all(outcome.dones):
                    break
            row = dict(seed=seed, native_done=all(outcome.dones), **telemetry.finish(env))
            flat = np.concatenate(actions_all, axis=0) if actions_all else np.zeros((0, 2))
            normalized = flat / np.asarray([A_MAX, W_MAX]) if len(flat) else flat
            row.update(
                min_target_distance=None if not np.isfinite(minimum_distance) else minimum_distance,
                repeated_ring2_max=ring_max[2],
                repeated_ring3_max=ring_max[3],
                geometric_success=geometric_step is not None,
                settled_success=settled_step is not None,
                geometric_success_step=geometric_step,
                settled_success_step=settled_step,
                final_speed_mean=float(np.mean([robot.speed for robot in env.pursuers])),
                final_speed_max=float(np.max([robot.speed for robot in env.pursuers])),
                action_a_mean=float(flat[:, 0].mean()) if len(flat) else 0.0,
                action_a_std=float(flat[:, 0].std()) if len(flat) else 0.0,
                action_w_mean=float(flat[:, 1].mean()) if len(flat) else 0.0,
                action_w_std=float(flat[:, 1].std()) if len(flat) else 0.0,
                action_saturation_ratio=float((np.abs(normalized) >= 0.99).mean()) if len(flat) else 0.0,
            )
            records.append(row)
        by_mode[mode] = {
            "summary": summarize(records),
            "extended_summary": {
                "geometric_success_rate": float(np.mean([row["geometric_success"] for row in records])),
                "settled_success_rate": float(np.mean([row["settled_success"] for row in records])),
                "normal_capture_rate": float(np.mean([row["normal_capture"] for row in records])),
                "stationary_capture_rate": float(np.mean([row["stationary_capture"] for row in records])),
                "min_target_distance_mean": float(np.mean([row["min_target_distance"] for row in records if row["min_target_distance"] is not None])) if any(row["min_target_distance"] is not None for row in records) else None,
                "repeated_ring2_max_mean": float(np.mean([row["repeated_ring2_max"] for row in records])),
                "repeated_ring3_max_mean": float(np.mean([row["repeated_ring3_max"] for row in records])),
                "final_speed_mean": float(np.mean([row["final_speed_mean"] for row in records])),
                "final_speed_max_mean": float(np.mean([row["final_speed_max"] for row in records])),
                "action_saturation_ratio": float(np.mean([row["action_saturation_ratio"] for row in records])),
            },
            "records": records,
            "wall_seconds": time.monotonic() - started,
        }
    return {
        "schema": SCHEMA,
        "task": task,
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "seed_base": seed_base,
        "modes": by_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("coverage", "capture", "all"), default="all")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/2026-09-16_td3_local_aw_stage1/controls",
    )
    args = parser.parse_args()
    tasks = ("coverage", "capture") if args.task == "all" else (args.task,)
    for task in tasks:
        report = evaluate_task(task, args.episodes, args.max_steps)
        path = args.output / f"{task}.json"
        atomic_json(path, report)
        print(json.dumps({"task": task, "path": str(path), "modes": {key: value["extended_summary"] for key, value in report["modes"].items()}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
