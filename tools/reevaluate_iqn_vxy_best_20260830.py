#!/usr/bin/env python3
"""Fixed-midpoint-quantile 100-episode re-evaluation of IQN-VXY best checkpoints."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocap_voradj.training.continuous.formal_config import resolve_ladder_config
from tools.run_continuous_ctde_training import _screen
from tools.run_small_step_ac_migration import (
    configure_environment,
    make_components,
    policy_view,
    seed_all,
    stable_hash,
)


CONFIGS = {
    seed_index: f"configs/experiments/small_step_ac_migration_20260828/iqn_vxy9_seed{seed_index}.yaml"
    for seed_index in (1, 2, 3)
}
PROTOCOL_STEPS = {
    # A common-step comparison is needed to validate the reported 70% aggregate.
    "uniform225": {1: 225_000, 2: 225_000, 3: 225_000},
    # These are the independently selected 20-episode peaks recorded in the ledger.
    "per-seed-best": {1: 250_000, 2: 225_000, 3: 225_000},
    # Paired with uniform225 only for the reward-fidelity tail audit.
    "uniform300": {1: 300_000, 2: 300_000, 3: 300_000},
}


def specs(protocol: str) -> tuple[tuple[int, str, str, int], ...]:
    return tuple(
        (
            seed_index,
            CONFIGS[seed_index],
            (
                "artifacts/2026-08-28_small_step_ac/"
                f"iqn_vxy9_seed{seed_index}/checkpoints/step_{step:09d}.pt"
            ),
            step,
        )
        for seed_index, step in PROTOCOL_STEPS[protocol].items()
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def evaluate_one(seed_index: int, config_path: Path, checkpoint: Path, output: Path,
                 episodes: int, device: str, evaluation_seed: int, protocol: str,
                 reward_tail_windows: tuple[int, ...] = ()) -> dict:
    root_config = resolve_ladder_config(config_path)
    algorithm = str(root_config["small_step_ac"]["algorithm"])
    if algorithm != "iqn_vxy9":
        raise ValueError(f"expected iqn_vxy9, got {algorithm}")
    config = configure_environment(root_config, algorithm)
    config["seed"] = int(root_config["seed"])
    seed_all(evaluation_seed)
    trainer, _ = make_components(config, algorithm, device)
    payload = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    if payload.get("schema") != "small-step-ac-milestone-v1":
        raise ValueError(f"unexpected milestone schema: {checkpoint}")
    if payload.get("config_hash") != stable_hash(config):
        raise ValueError(f"checkpoint/config contract mismatch: {checkpoint}")
    trainer.load_state_dict(payload["trainer"])
    view = policy_view(trainer, algorithm, config)
    view.actor.eval()
    evaluation_root = copy.deepcopy(root_config)
    evaluation_root["action_mode"] = "desired_velocity_2d_body"
    evaluation_root.setdefault("env", {})["action_mode"] = "desired_velocity_2d_body"
    evaluation_root.setdefault("pursuer", {})["action_mode"] = "desired_velocity_2d_body"
    evaluation_root.setdefault("action", {}).update(
        {
            "mode": "desired_velocity_2d_body",
            "servo_acceleration_limit": float(config["action"].get("a_max", 0.4)),
        }
    )
    evaluation_root["actor"]["max_pursuers"] = int(config["actor"]["max_pursuers"])
    evaluation_root["tasks"]["capture"]["action_mode"] = "desired_velocity_2d_body"
    for key in ("env", "pursuer", "action"):
        evaluation_root["tasks"]["capture"].setdefault(key, {}).update(evaluation_root[key])
    evaluation_root["tasks"]["capture"].setdefault("actor", {})["max_pursuers"] = int(
        config["actor"]["max_pursuers"]
    )
    with torch.no_grad():
        result = _screen(
            view,
            evaluation_root,
            evaluation_seed,
            episodes,
            str(trainer.device),
            scenes=("capture",),
            max_steps=1000,
            deterministic=True,
            reward_tail_windows=reward_tail_windows,
        )
    row = {
        "schema": "iqn-vxy-fixed-midpoint-reeval-v1",
        "seed_index": seed_index,
        "training_seed": int(config["seed"]),
        "evaluation_seed": evaluation_seed,
        "episodes": episodes,
        "checkpoint_step": int(payload["step"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(config_path),
        "config_hash": payload["config_hash"],
        "quantile_contract": "fixed midpoint tau_i=(i+0.5)/N, N=action_quantile_samples",
        "selection_protocol": protocol,
        "reward_tail_windows": list(reward_tail_windows),
        "deterministic": result,
    }
    atomic_json(output, row)
    return row


def result_filename(seed_index: int, step: int, episodes: int, reward_tail_audit: bool) -> str:
    suffix = "_tailaudit" if reward_tail_audit else ""
    return f"seed{seed_index}_step{step}_fixed_tau_{episodes}ep{suffix}.json"


def aggregate_reward_groups(rows: list[dict]) -> dict[str, dict]:
    records = [
        record
        for row in rows
        for record in ((row.get("deterministic") or {}).get("capture") or {}).get("records", [])
    ]
    groups = {
        "captured": [record for record in records if bool(record.get("captured", False))],
        "failed_3plus": [
            record
            for record in records
            if not bool(record.get("captured", False)) and bool(record.get("visited_3plus_ring", False))
        ],
    }
    result: dict[str, dict] = {}
    for group_name, selected in groups.items():
        summary: dict[str, object] = {
            "episodes": len(selected),
            "mean_episode_return": float(np.mean([row["episode_return_mean"] for row in selected])) if selected else None,
            "mean_3plus_hold": float(np.mean([row["max_3plus_ring_hold_steps"] for row in selected])) if selected else None,
            "collision_rate": float(np.mean([bool(row["collision_event"]) for row in selected])) if selected else None,
            "mean_length": float(np.mean([row["length"] for row in selected])) if selected else None,
            "tail_windows": {},
        }
        windows = sorted({
            window
            for record in selected
            for window in (record.get("reward_tail_windows", {}) or {})
        }, key=int)
        for window in windows:
            tails = [record["reward_tail_windows"][window] for record in selected]
            agent_steps = sum(int(tail["agent_steps"]) for tail in tails)
            observed_steps = sum(int(tail["observed_window_steps"]) for tail in tails)
            component_names = sorted({name for tail in tails for name in tail["component_sums"]})
            component_sums = {
                name: float(sum(float(tail["component_sums"].get(name, 0.0)) for tail in tails))
                for name in component_names
            }
            role_counts: dict[str, int] = {}
            collision_type_counts: dict[str, int] = {}
            for tail in tails:
                for role, count in tail["role_agent_step_counts"].items():
                    role_counts[role] = role_counts.get(role, 0) + int(count)
                for collision_type, count in tail["collision_type_step_counts"].items():
                    collision_type_counts[collision_type] = collision_type_counts.get(collision_type, 0) + int(count)
            summary["tail_windows"][window] = {
                "episodes": len(tails),
                "observed_window_steps": observed_steps,
                "agent_steps": agent_steps,
                "component_agent_step_means": {
                    name: value / max(agent_steps, 1) for name, value in component_sums.items()
                },
                "component_episode_sum_means": {
                    name: value / max(len(tails), 1) for name, value in component_sums.items()
                },
                "role_agent_step_fractions": {
                    role: count / max(agent_steps, 1) for role, count in role_counts.items()
                },
                "collision_step_rate": sum(int(tail["collision_step_count"]) for tail in tails) / max(observed_steps, 1),
                "boundary_step_rate": sum(int(tail["boundary_step_count"]) for tail in tails) / max(observed_steps, 1),
                "collision_type_step_counts": collision_type_counts,
            }
        result[group_name] = summary
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-root", default="artifacts/2026-08-30_iqn_vxy_fixed_tau_reeval")
    parser.add_argument("--seed-index", type=int, choices=(1, 2, 3))
    parser.add_argument("--protocol", choices=tuple(PROTOCOL_STEPS), default="uniform225")
    parser.add_argument("--reward-tail-audit", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    output_root = ROOT / args.output_root
    rows = []
    if args.aggregate_only:
        for seed_index, _, _, step in specs(args.protocol):
            path = output_root / result_filename(seed_index, step, args.episodes, args.reward_tail_audit)
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    else:
        for seed_index, config_raw, checkpoint_raw, step in specs(args.protocol):
            if args.seed_index is not None and seed_index != args.seed_index:
                continue
            rows.append(
                evaluate_one(
                    seed_index,
                    ROOT / config_raw,
                    ROOT / checkpoint_raw,
                    output_root / result_filename(seed_index, step, args.episodes, args.reward_tail_audit),
                    args.episodes,
                    args.device,
                    2026083000 + seed_index * 10000 + (5_000_000 if args.reward_tail_audit else 0),
                    args.protocol,
                    (100, 200) if args.reward_tail_audit else (),
                )
            )
    if args.seed_index is None or args.aggregate_only:
        captures = [float(row["deterministic"]["capture"]["capture_rate"]) for row in rows]
        collisions = [float(row["deterministic"]["capture"]["collision_rate"]) for row in rows]
        aggregate = {
            "schema": "iqn-vxy-fixed-midpoint-aggregate-v1",
            "selection_protocol": args.protocol,
            "episodes_per_seed": args.episodes,
            "mean_capture_rate": float(np.mean(captures)),
            "mean_collision_rate": float(np.mean(collisions)),
            "seed_results": rows,
        }
        if args.reward_tail_audit:
            aggregate["reward_fidelity_groups"] = aggregate_reward_groups(rows)
        aggregate_suffix = "_tailaudit" if args.reward_tail_audit else ""
        atomic_json(
            output_root / f"aggregate_{args.protocol.replace('-', '_')}{aggregate_suffix}.json",
            aggregate,
        )
        if args.protocol == "uniform225" and not args.reward_tail_audit:
            atomic_json(output_root / "aggregate.json", aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
