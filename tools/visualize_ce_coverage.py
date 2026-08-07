#!/usr/bin/env python3
"""Render deterministic CE-Coverage rollouts with evaluator-identical semantics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.trainer import safe_json_dumps, set_global_config
from tools.evaluate_ce_coverage import _active_actions, _load_eval_config
from tools.rollout_voradj_visual import render_gif, snapshot_env


def _snapshot_ce(env: VorAdjEnv, step: int) -> Dict[str, Any]:
    snapshot = snapshot_env(env, "coverage", "coverage", 0, step, step)
    data = env._voronoi_map()
    metrics = env.last_distribution_metrics
    snapshot["ce_targets"] = [
        {"id": int(key[1]), "x": float(target[0]), "y": float(target[1])}
        for key, target in data.get("centroids", {}).items()
        if key[0] == "pursuer"
    ]
    snapshot["ce_center_rms"] = float(metrics.get("ce_center_rms", float("nan")))
    snapshot["ce_center_max"] = float(metrics.get("ce_center_max", float("nan")))
    active_speeds = [
        float(pursuer.speed) for pursuer in env.pursuers if not pursuer.deactivated
    ]
    snapshot["active_mean_speed"] = (
        float(np.mean(active_speeds)) if active_speeds else 0.0
    )
    return snapshot


def rollout(
    model: CoCapIQN,
    config: Dict[str, Any],
    seed: int,
    device: str,
    max_steps: int,
    post_success_steps: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    set_global_config(config)
    env = VorAdjEnv(config, seed=seed)
    observations = env.reset()
    frames = [_snapshot_ce(env, 0)]
    success_step: Optional[int] = None
    post_success_actions: List[int] = []
    post_success_speeds: List[float] = []

    for _ in range(max_steps + post_success_steps):
        actions = _active_actions(model, observations, device)
        result = env.step(actions, [])
        observations = result.observations
        frames.append(_snapshot_ce(env, int(env.episode_step)))

        if success_step is None and env.post_capture_coverage_success:
            success_step = int(env.episode_step)
        elif success_step is not None:
            post_success_actions.extend(int(action) for action in actions if action is not None)
            post_success_speeds.extend(
                float(pursuer.speed)
                for pursuer in env.pursuers
                if not pursuer.deactivated
            )
            if env.episode_step - success_step >= post_success_steps:
                break

        if all(result.dones) and success_step is None:
            break
        if success_step is None and env.episode_step >= max_steps:
            break

    success = success_step is not None
    final_status = {
        "capture_success": "false",
        "coverage_success": "true" if success else "time out",
        "episode_success": "true" if success else "false",
        "capture_success_bool": False,
        "coverage_success_bool": success,
        "episode_success_bool": success,
    }
    for frame in frames:
        frame["display_status"] = dict(final_status)

    final_metrics = env.last_distribution_metrics
    summary = {
        "seed": int(seed),
        "success": success,
        "success_step": -1 if success_step is None else int(success_step),
        "frames": len(frames),
        "post_success_observation_steps": (
            int(post_success_steps) if success_step is not None else 0
        ),
        "center_rms": float(final_metrics.get("ce_center_rms", float("inf"))),
        "center_max": float(final_metrics.get("ce_center_max", float("inf"))),
        "episode_end_mean_speed": float(frames[-1]["active_mean_speed"]),
        "post_success_mean_speed": (
            float(np.mean(post_success_speeds)) if post_success_speeds else float("inf")
        ),
        "post_success_idle_action_ratio": (
            float(np.mean(np.asarray(post_success_actions) == 4))
            if post_success_actions
            else 0.0
        ),
    }
    return frames, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--device", default="cuda:1" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--post-success-steps", type=int, default=30)
    parser.add_argument("--max-gif-frames", type=int, default=300)
    parser.add_argument("--frame-duration-ms", type=int, default=90)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint)
    output_root = Path(args.output_root)
    config = _load_eval_config(run_dir)
    model = CoCapIQN.load(str(checkpoint), device=str(args.device))
    model.eval()
    with torch.no_grad():
        frames, summary = rollout(
            model,
            config,
            seed=int(args.seed),
            device=str(args.device),
            max_steps=int(args.max_steps),
            post_success_steps=int(args.post_success_steps),
        )

    gif_path = output_root / f"rollout_ce_coverage_seed_{int(args.seed)}.gif"
    render_record = render_gif(
        frames,
        gif_path,
        int(args.max_gif_frames),
        0,
        int(args.frame_duration_ms),
        "ce coverage",
        False,
        False,
        False,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "policy_quantile_mode": "fixed_midpoint_32",
        "summary": summary,
        **render_record,
    }
    (output_root / f"summary_ce_coverage_seed_{int(args.seed)}.json").write_text(
        safe_json_dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / f"episode_ce_coverage_seed_{int(args.seed)}.json").write_text(
        json.dumps({"summary": payload, "frames": frames}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(safe_json_dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
