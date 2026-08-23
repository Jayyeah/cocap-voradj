#!/usr/bin/env python3
"""Evaluate supplied MADDPG Roundup checkpoints while preserving upstream bugs."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from open_encirclement_ctde.geometry import angular_metrics, point_in_triangle
from open_encirclement_ctde.runtime import atomic_json_dump


def collision_flags(env) -> list[bool]:
    flags = []
    for position in env.multi_current_pos:
        collided = bool(np.any(np.asarray(position) < 0.0) or np.any(np.asarray(position) > env.length))
        collided = collided or any(
            np.linalg.norm(np.asarray(position) - np.asarray(ob.position)) < ob.radius
            for ob in env.obstacles
        )
        flags.append(bool(collided))
    return flags


def run_episode(env_cls, algo_cls, seed: int, noisy: bool, make_gif: Path | None) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env = env_cls()
    actor_dims = [space.shape[0] for space in env.observation_space.values()]
    algo = algo_cls(
        actor_dims,
        sum(actor_dims),
        env.num_agents,
        2,
        fc1=128,
        fc2=128,
        alpha=1e-4,
        beta=3e-3,
        scenario="UAV_Round_up",
        chkpt_dir="tmp/maddpg/",
    )
    algo.load_checkpoint()
    obs = env.reset()
    frames = []
    max_action_abs = 0.0
    max_action_norm = 0.0
    collision_steps = 0
    containment_steps = 0
    all_radius_steps = 0
    original_capture = False
    last_distances = [float("nan")] * 3
    largest_gap = float("nan")
    for step in range(100):
        actions = algo.choose_action(obs, step, evaluate=not noisy)
        action_array = np.asarray(actions, dtype=np.float64)
        max_action_abs = max(max_action_abs, float(np.max(np.abs(action_array))))
        max_action_norm = max(max_action_norm, float(np.max(np.linalg.norm(action_array, axis=1))))
        obs, _, dones = env.step(actions)
        hunters = np.asarray(env.multi_current_pos[:3], dtype=np.float64)
        target = np.asarray(env.multi_current_pos[3], dtype=np.float64)
        contained = point_in_triangle(target, hunters)
        distances = np.linalg.norm(hunters - target, axis=1)
        largest_gap, _ = angular_metrics(hunters, target)
        containment_steps += int(contained)
        all_radius_steps += int(np.all(distances <= 0.3))
        collision_steps += int(any(collision_flags(env)[:3]))
        last_distances = distances.tolist()
        if make_gif is not None:
            frames.append(np.asarray(env.render())[..., :3])
        if any(dones):
            original_capture = True
            episode_length = step + 1
            break
    else:
        episode_length = 100
    if make_gif is not None and frames:
        make_gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(make_gif, frames, duration=0.08, loop=0)
    env.close()
    return {
        "seed": seed,
        "mode": "upstream_original_noise" if noisy else "deterministic",
        "original_capture": original_capture,
        "episode_length": episode_length,
        "truncated": not original_capture,
        "containment_fraction": containment_steps / episode_length,
        "all_within_radius_fraction": all_radius_steps / episode_length,
        "collision_fraction": collision_steps / episode_length,
        "final_distances": last_distances,
        "final_largest_angular_gap": largest_gap,
        "max_abs_action_component": max_action_abs,
        "max_action_norm": max_action_norm,
    }


def summarize(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "capture_rate": float(np.mean([row["original_capture"] for row in rows])),
        "mean_episode_length": float(np.mean([row["episode_length"] for row in rows])),
        "mean_containment_fraction": float(np.mean([row["containment_fraction"] for row in rows])),
        "mean_all_within_radius_fraction": float(np.mean([row["all_within_radius_fraction"] for row in rows])),
        "mean_collision_fraction": float(np.mean([row["collision_fraction"] for row in rows])),
        "max_abs_action_component": float(max(row["max_abs_action_component"] for row in rows)),
        "max_action_norm": float(max(row["max_action_norm"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-mode", type=int, default=20)
    parser.add_argument("--gif-count", type=int, default=2)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    output = args.output_dir.resolve()
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(snapshot))
    old_cwd = Path.cwd()
    os.chdir(snapshot)
    try:
        from maddpg import MADDPG
        from sim_env import UAVEnv

        all_rows = []
        mode_summaries = {}
        for noisy, name, seed_base in ((False, "deterministic", 7100), (True, "upstream_original_noise", 8100)):
            rows = []
            for index in range(args.episodes_per_mode):
                gif = None
                if index < args.gif_count:
                    gif = output / "gifs" / f"{name}_seed_{seed_base + index}.gif"
                row = run_episode(UAVEnv, MADDPG, seed_base + index, noisy, gif)
                rows.append(row)
                all_rows.append(row)
            mode_summaries[name] = summarize(rows)
    finally:
        os.chdir(old_cwd)
    report = {
        "schema": "upstream-raw-r0-report-v1",
        "contract": "roundup_raw_v0",
        "upstream_commit": "15309de231f639c62d2049b0ad5b07b8975309c9",
        "known_bugs_preserved": True,
        "checkpoint_contains_all_four_agents": True,
        "modes": mode_summaries,
        "episodes": all_rows,
    }
    atomic_json_dump(report, output / "report.json")
    print(json.dumps({"report": str(output / "report.json"), "modes": mode_summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
