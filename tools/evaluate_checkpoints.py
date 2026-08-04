from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.trainer import set_global_config
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.control.apf import ApfAgent


def load_config(run_dir: Path) -> Dict[str, Any]:
    with (run_dir / "effective_config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def checkpoint_grid(run_dir: Path, checkpoint_steps: Optional[List[int]] = None) -> List[Path]:
    if checkpoint_steps:
        candidates = [run_dir / "checkpoints" / f"step_{step}.pt" for step in checkpoint_steps]
        final_candidates = sorted((run_dir / "checkpoints").glob("final_step_*.pt"))
        candidates.extend(final_candidates[-1:])
    else:
        candidates = [
            run_dir / "checkpoints" / "init.pt",
            run_dir / "checkpoints" / "step_1000000.pt",
            run_dir / "checkpoints" / "final_step_2000000.pt",
        ]
    return [path for path in candidates if path.is_file()]


def recovery_positions(rng: np.random.Generator, width: float, height: float) -> List[List[float]]:
    radius = 8.0
    margin = radius + 12.0
    center = rng.uniform([margin, margin], [width - margin, height - margin])
    phase = rng.uniform(0.0, 2.0 * np.pi)
    return [
        (
            center
            + radius
            * np.asarray(
                [np.cos(phase + 2.0 * np.pi * idx / 4.0), np.sin(phase + 2.0 * np.pi * idx / 4.0)]
            )
        ).tolist()
        for idx in range(4)
    ]


def scenario_config(base: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["device"] = "cpu"
    cfg.setdefault("env", {})
    if scenario == "mixed":
        cfg["env"]["num_evaders"] = 1
    else:
        cfg["env"]["num_evaders"] = 0
    if scenario == "recovery_cluster":
        cfg["env"]["pursuer_spawn_min_sep"] = 2.0
    return cfg


def evaluate_episode(
    model: CoCapIQN,
    cfg: Dict[str, Any],
    scenario: str,
    seed: int,
    device: str,
    max_steps: int,
) -> Dict[str, Any]:
    set_global_config(cfg)
    env = VorAdjEnv(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    initial_positions: Optional[List[List[float]]] = None
    if scenario == "recovery_cluster":
        initial_positions = recovery_positions(rng, env.width, env.height)
    obs_list = env.reset(initial_pursuer_positions=initial_positions)
    apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]

    for _ in range(max_steps):
        active = [idx for idx, obs in enumerate(obs_list) if obs is not None]
        actions: List[Optional[int]] = [None] * len(obs_list)
        if active:
            batch = stack_obs([obs_list[idx] for idx in active], device)
            selected = model.act(
                batch,
                mode="voradj",
                epsilon=0.0,
                deterministic_quantiles=True,
            ).detach().cpu().tolist()
            for idx, action in zip(active, selected):
                actions[idx] = int(action)

        evader_actions: List[Optional[int]] = []
        if env.evaders:
            set_global_config(cfg)
            if hasattr(env, "configure_evader_apf_agents"):
                env.configure_evader_apf_agents(apf_agents)
            for idx, evader_obs in enumerate(env.get_evader_observations_for_apf()):
                evader_actions.append(None if evader_obs is None else int(apf_agents[idx].act(evader_obs)))
        result = env.step(actions, evader_actions)
        obs_list = result.observations
        if all(result.dones):
            break

    record = env.episode_record(task=scenario)
    return {
        "seed": seed,
        "scenario": scenario,
        "truncated_by_eval": bool(env.episode_step >= max_steps and not record["episode_success"]),
        **record,
    }


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = max(len(records), 1)

    def rate(key: str) -> float:
        return sum(bool(record.get(key, False)) for record in records) / count

    def voradj_rate(key: str) -> float:
        return sum(bool(record.get("voradj_metrics", {}).get(key, False)) for record in records) / count

    return {
        "episodes": len(records),
        "episode_success_rate": rate("episode_success"),
        "captured_rate": rate("captured"),
        "fully_capture_rate": rate("fully_capture"),
        "strict_coverage_rate": rate("coverage_strict_success"),
        "post_capture_coverage_rate": voradj_rate("post_capture_coverage_success"),
        "post_capture_window_expired_rate": voradj_rate("post_capture_window_expired"),
        "collision_rate": rate("collision_event"),
        "soft_oob_rate": rate("soft_boundary_out_of_bounds_event"),
        "avg_length": float(np.mean([record["length"] for record in records])) if records else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint-step",
        action="append",
        type=int,
        dest="checkpoint_steps",
        help="Evaluate this step checkpoint; repeat for multiple steps. The run's final checkpoint is also included.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for raw_run_dir in args.run_dir:
            run_dir = Path(raw_run_dir)
            base_cfg = load_config(run_dir)
            for checkpoint in checkpoint_grid(run_dir, args.checkpoint_steps):
                model = CoCapIQN.load(str(checkpoint), device=args.device)
                model.eval()
                for scenario in ("mixed", "pure_random", "recovery_cluster"):
                    cfg = scenario_config(base_cfg, scenario)
                    records = [
                        evaluate_episode(
                            model,
                            cfg,
                            scenario,
                            args.seed + episode_idx,
                            args.device,
                            args.max_steps,
                        )
                        for episode_idx in range(args.episodes)
                    ]
                    payload = {
                        "run_dir": str(run_dir),
                        "checkpoint": str(checkpoint),
                        "scenario": scenario,
                        "max_steps": args.max_steps,
                        "policy_quantile_mode": "fixed_midpoint_32",
                        "summary": summarize(records),
                        "episodes": records,
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(json.dumps({key: payload[key] for key in ("checkpoint", "scenario", "summary")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
