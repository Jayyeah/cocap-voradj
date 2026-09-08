#!/usr/bin/env python3
"""Formal100 Gate for a distilled MAPPO-9-v2 Actor before any PPO update."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.continuous.formal_config import resolve_ladder_config, scene_config
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.small_step_ac import tensor_tree
from cocap_voradj.training.trainer import set_global_config
from tools.evaluate_iqn_corrected_capture import (
    DEFAULT_COLLISION_SEMANTICS,
    action_grid_from_config,
    current_apf_actions,
    map_action_index,
    validate_corrected_capture_contract,
)
from tools.evaluate_vxy_stage2_cross_retention_20260903 import ring_count
from cocap_voradj.evaluation.mission_events import MissionEventTracker, snapshot, summarize_events
from cocap_voradj.training.runtime_semantics import assert_runtime, initial_state_fingerprint
from tools.probe_cf3_policy_temperature import apply_collision_semantics_override
from tools.run_small_step_ac_migration import configure_environment, make_components

CHECKPOINT_SCHEMA = "mappo-iqn-distilled-actor-v1"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def max_hold(values: list[int], threshold: int) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value >= threshold else 0
        best = max(best, current)
    return best


@torch.no_grad()
def actions_for_state(teacher, actor, observations, env, device: str, stochastic: bool = False):
    active = [index for index, obs in enumerate(observations) if obs is not None]
    active_obs = [observations[index] for index in active]
    if not active:
        return [None] * len(observations), [], [], []
    teacher_batch = stack_obs(active_obs, device)
    tau = (torch.arange(32, device=device, dtype=torch.float32) + 0.5) / 32.0
    teacher_q = teacher(teacher_batch, num_tau=32, mode="voradj", tau=tau)["q_values"].mean(dim=1)
    teacher_action = teacher_q.argmax(dim=-1)
    student_batch = tensor_tree(
        {key: np.stack([np.asarray(obs[key]) for obs in active_obs]) for key in active_obs[0]},
        torch.device(device),
    )
    distribution = actor.distribution(student_batch)
    student_action = distribution.sample() if stochastic else distribution.logits.argmax(dim=-1)
    grid = action_grid_from_config(env.config)
    commands: list[np.ndarray | None] = [None] * len(observations)
    for row, agent_index in enumerate(active):
        raw = map_action_index(int(student_action[row]), grid)
        commands[agent_index] = np.asarray(env.action_adapter.validate(raw), dtype=np.float32)
    return (
        commands,
        active,
        teacher_action.cpu().numpy().astype(np.int64),
        student_action.cpu().numpy().astype(np.int64),
    )


def run_episode(teacher, actor, root_config, seed: int, device: str, stochastic: bool = False) -> dict[str, Any]:
    config = scene_config(root_config, "capture")
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=int(seed))
    observations = list(env.reset())
    initial_fingerprint = initial_state_fingerprint(env)
    actor.eval()
    assert_runtime(env, actor, categorical=True)
    torch.manual_seed(int(seed) + 1000000)
    tracker = MissionEventTracker(env.pursuers[0].dt * env.pursuers[0].N)
    tracker.observe(snapshot(env, observations), 0)
    apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    agreements: list[bool] = []
    support_agreements: list[bool] = []
    direct_agreements: list[bool] = []
    ring_counts: list[int] = []
    capture_types: list[str] = []
    collision_types: Counter[str] = Counter()
    collision = False
    for step in range(1, int(config["env"]["episode_max_length"]) + 1):
        commands, active, teacher_actions, student_actions = actions_for_state(
            teacher, actor, observations, env, device, stochastic=stochastic
        )
        outcome = env.step(commands, current_apf_actions(env, apf_agents))
        hits = teacher_actions == student_actions
        agreements.extend(bool(value) for value in hits)
        for row, agent_index in enumerate(active):
            metadata = dict(outcome.infos[agent_index].get("replay_metadata", {}) or {})
            if bool(metadata.get("support_candidate", False)):
                support_agreements.append(bool(hits[row]))
            if int(metadata.get("vct_ls_direct_enemy_count", 0) or 0) > 0:
                direct_agreements.append(bool(hits[row]))
        capture_types.extend(
            str(event.get("capture_type", "unknown"))
            for event in list(getattr(env, "last_capture_events", []) or [])
        )
        events = list(getattr(env, "last_collision_events", []) or [])
        collision = collision or bool(events)
        collision_types.update(str(event.get("type", "unknown")) for event in events)
        ring_counts.append(int(ring_count(env)))
        observations = list(outcome.observations)
        tracker.observe(snapshot(env, observations), step, env.last_capture_events)
        if all(outcome.dones):
            break
    record = env.episode_record(task="capture")
    captured = bool(record.get("captured", False))
    normal = bool(captured and any(value != "stationary" for value in capture_types))
    stationary = bool(captured and any(value == "stationary" for value in capture_types))
    return {
        "seed": int(seed),
        "initial_state_fingerprint": initial_fingerprint,
        "seed_semantics": "exact_environment_seed",
        "mission_events": tracker.finish(step),
        "captured": captured,
        "normal_capture": normal,
        "stationary_capture": stationary,
        "collision": bool(collision or record.get("collision_event", False)),
        "collision_type_counts": dict(collision_types),
        "length": int(record.get("episode_length", record.get("length", env.episode_step))),
        "visited_2plus": bool(any(value >= 2 for value in ring_counts)),
        "visited_3plus": bool(any(value >= 3 for value in ring_counts)),
        "max_3plus_hold": int(max_hold(ring_counts, 3)),
        "action_agreement": float(np.mean(agreements)) if agreements else 0.0,
        "support_action_agreement": float(np.mean(support_agreements)) if support_agreements else None,
        "direct_action_agreement": float(np.mean(direct_agreements)) if direct_agreements else None,
        "agent_steps": len(agreements),
    }


def mean_present(records, key):
    values = [row[key] for row in records if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def summarize(records):
    collision_types: Counter[str] = Counter()
    for row in records:
        collision_types.update(row["collision_type_counts"])
    return {
        "episodes": len(records),
        "safe_capture_rate": float(np.mean([row["captured"] and not row["collision"] for row in records])),
        "mission_event_summary": summarize_events([e for row in records for e in row.get("mission_events", {}).get("events", [])]),
        "capture_rate": float(np.mean([row["captured"] for row in records])),
        "normal_capture_rate": float(np.mean([row["normal_capture"] for row in records])),
        "stationary_capture_rate": float(np.mean([row["stationary_capture"] for row in records])),
        "collision_rate": float(np.mean([row["collision"] for row in records])),
        "visited_2plus_rate": float(np.mean([row["visited_2plus"] for row in records])),
        "visited_3plus_rate": float(np.mean([row["visited_3plus"] for row in records])),
        "mean_capture_length": mean_present([row for row in records if row["captured"]], "length"),
        "mean_episode_length": mean_present(records, "length"),
        "mean_action_agreement": mean_present(records, "action_agreement"),
        "mean_support_action_agreement": mean_present(records, "support_action_agreement"),
        "mean_direct_action_agreement": mean_present(records, "direct_action_agreement"),
        "max_3plus_hold": max((row["max_3plus_hold"] for row in records), default=0),
        "agent_steps": sum(row["agent_steps"] for row in records),
        "collision_type_counts": dict(collision_types),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--actor-checkpoint", required=True)
    parser.add_argument("--teacher-formal", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--policy-mode", choices=("argmax", "sample"), default="argmax")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026090301)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--min-agreement", type=float, default=0.90)
    parser.add_argument("--max-capture-drop", type=float, default=0.10)
    parser.add_argument("--max-collision-increase", type=float, default=0.05)
    parser.add_argument("--max-length-ratio", type=float, default=1.50)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes must be positive")

    teacher_path = Path(args.teacher_checkpoint).resolve()
    if sha256_file(teacher_path) != str(args.expected_teacher_sha256).lower():
        raise ValueError("teacher checkpoint SHA mismatch")
    actor_path = Path(args.actor_checkpoint).resolve()
    actor_payload = torch.load(actor_path, map_location=str(args.device), weights_only=True)
    if actor_payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported distilled actor checkpoint")

    root = resolve_ladder_config(Path(args.config).resolve())
    configured = configure_environment(root, "mappo9_v2")
    evaluation_root = apply_collision_semantics_override(root, DEFAULT_COLLISION_SEMANTICS)
    contract = validate_corrected_capture_contract(scene_config(evaluation_root, "capture"))
    trainer, _ = make_components(configured, "mappo9_v2", str(args.device))
    actor = trainer.actor
    actor.load_state_dict(actor_payload["actor_state_dict"], strict=True)
    actor.eval()
    teacher = CoCapIQN.load(str(teacher_path), device=str(args.device)).eval()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    set_global_config(configured)
    probe_env = VorAdjEnv(copy.deepcopy(configured), seed=int(args.seed))
    probe_env.reset()
    atomic_json(output_root / "runtime_preflight.json", assert_runtime(probe_env, actor, categorical=True))
    records = []
    import time
    started = time.monotonic()
    for index in range(int(args.episodes)):
        records.append(run_episode(teacher, actor, evaluation_root, int(args.seed) + index, str(args.device), stochastic=args.policy_mode == "sample"))
        elapsed = time.monotonic() - started
        atomic_json(output_root / "progress.json", {"status": "running", "completed": len(records), "total": args.episodes, "elapsed_seconds": elapsed, "eta_seconds": elapsed / len(records) * (args.episodes - len(records)), "partial_summary": summarize(records)})
        atomic_json(output_root / "records_partial.json", {"records": records})
    student = summarize(records)
    teacher_formal = json.loads(Path(args.teacher_formal).read_text(encoding="utf-8"))
    teacher_summary = dict(teacher_formal["summary"])
    teacher_capture = float(teacher_summary["capture_rate"])
    teacher_collision = float(teacher_summary["collision_rate"])
    teacher_length = float(teacher_summary["mean_episode_length"])
    checks = {
        "action_agreement": float(student["mean_action_agreement"]) >= float(args.min_agreement),
        "capture_retention": float(student["capture_rate"]) >= teacher_capture - float(args.max_capture_drop),
        "collision_retention": float(student["collision_rate"]) <= teacher_collision + float(args.max_collision_increase),
        "capture_length": (
            student["mean_capture_length"] is not None
            and float(student["mean_capture_length"]) <= teacher_length * float(args.max_length_ratio)
        ),
    }
    decision = ("STOCHASTIC_EVAL_COMPLETE_REVIEW_REQUIRED" if args.policy_mode == "sample" else ("PASS_TO_PPO_BRANCHES" if all(checks.values()) else "FAIL_STOP_BEFORE_PPO"))
    output_root = Path(args.output_root).resolve()
    report = {
        "schema": "mappo-iqn-distillation-formal-gate-v1",
        "status": "complete",
        "created_at": now(),
        "decision": decision,
        "checks": checks,
        "thresholds": {
            "min_agreement": float(args.min_agreement),
            "max_capture_drop": float(args.max_capture_drop),
            "max_collision_increase": float(args.max_collision_increase),
            "max_length_ratio": float(args.max_length_ratio),
        },
        "contract_audit": contract,
        "policy": {"teacher": "fixed_midpoint_32 epsilon=0", "student": "categorical sample, eval-mode forward" if args.policy_mode == "sample" else "deterministic categorical argmax"},
        "paired_seeds": [int(args.seed) + index for index in range(int(args.episodes))],
        "teacher": teacher_summary,
        "student": student,
        "actor_checkpoint": str(actor_path),
        "actor_checkpoint_sha256": sha256_file(actor_path),
        "records": records,
        "ppo_updates_performed": 0,
    }
    atomic_json(output_root / "gate_report.json", report)
    (output_root / "FORMAL_DONE").write_text(now() + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("decision", "checks", "student")}, indent=2, ensure_ascii=False))
    atomic_json(output_root / "progress.json", {"status": "complete", "completed": len(records), "total": args.episodes, "eta_seconds": 0, "summary": student})
    return 0 if args.policy_mode == "sample" or decision == "PASS_TO_PPO_BRANCHES" else 2


if __name__ == "__main__":
    raise SystemExit(main())
