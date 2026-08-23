#!/usr/bin/env python3
"""CLI for corrected Roundup MAPPO/MADDPG training."""

from __future__ import annotations

import argparse
import copy
import json
import traceback
from pathlib import Path

from open_encirclement_ctde.runtime import atomic_json_dump
from open_encirclement_ctde.training import load_config, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num-env-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--rollout-length", type=int)
    parser.add_argument("--ppo-epoch", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--final-stochastic-episodes", type=int)
    parser.add_argument("--final-gif-count", type=int)
    parser.add_argument("--rolling-full-interval", type=int)
    parser.add_argument("--updates-per-transition", type=float)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config = copy.deepcopy(load_config(args.config))
    if args.num_env_steps is not None:
        config["num_env_steps"] = args.num_env_steps
    if args.device is not None:
        config["device"] = args.device
    if args.run_dir is not None:
        config["run_dir"] = str(args.run_dir)
        config["run_name"] = args.run_dir.name
    if args.n_envs is not None:
        config["n_envs"] = args.n_envs
        if config["algorithm"] == "mappo":
            config["mappo"]["n_rollout_threads"] = args.n_envs
    if args.rollout_length is not None:
        config["mappo"]["rollout_length"] = args.rollout_length
    if args.ppo_epoch is not None:
        config["mappo"]["ppo_epoch"] = args.ppo_epoch
    if args.batch_size is not None:
        config["maddpg"]["batch_size"] = args.batch_size
    if args.warmup_steps is not None:
        config["maddpg"]["warmup_steps"] = args.warmup_steps
    if args.eval_episodes is not None:
        config["eval_episodes"] = args.eval_episodes
    if args.final_stochastic_episodes is not None:
        config["final_stochastic_episodes"] = args.final_stochastic_episodes
    if args.final_gif_count is not None:
        config["final_gif_count"] = args.final_gif_count
    if args.rolling_full_interval is not None:
        config["rolling_full_interval"] = args.rolling_full_interval
    if args.updates_per_transition is not None:
        config["maddpg"]["updates_per_transition"] = args.updates_per_transition
    try:
        result = train(config, repo_root, resume=args.resume)
    except Exception as exc:
        run_dir = Path(config["run_dir"])
        if not run_dir.is_absolute():
            run_dir = repo_root / run_dir
        failure = {
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json_dump(failure, run_dir / "failure.json")
        existing = {}
        state_path = run_dir / "state.json"
        if state_path.exists():
            existing = json.loads(state_path.read_text(encoding="utf-8"))
        atomic_json_dump({**existing, **failure}, state_path)
        raise
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
