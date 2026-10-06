#!/usr/bin/env python3
"""Parallel, read-only CPU evaluator for matched evidence checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

from cocap_voradj.models.iqn import CoCapIQN
from tools import iqn_token_matched_20260919 as matched
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file
from tools.iqn_evidence_comparison_20261006 import CONFIGS, STAGES
from tools.run_forward_final_bridge_20260908 import run_episode


SCENES = ("coverage", "capture", "mixed")
_WORKER: dict[str, Any] = {}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def gpu_snapshot() -> list[dict[str, int]]:
    try:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).splitlines()
        return [
            {"index": int(row.split(",")[0].strip()), "util_percent": int(row.split(",")[1].strip()), "free_mib": int(row.split(",")[2].strip())}
            for row in rows
        ]
    except Exception:
        return []


def choose_workers(requested: int | None = None) -> dict[str, Any]:
    cpus = max(int(os.cpu_count() or 1), 1)
    try:
        load1 = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        load1 = 0.0
    free_ram_gib = None
    try:
        import psutil
        free_ram_gib = psutil.virtual_memory().available / 1024**3
    except Exception:
        pass
    gpus = gpu_snapshot()
    try:
        compute = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).splitlines()
        training_gpu_active = bool([row.strip() for row in compute if row.strip()])
    except Exception:
        training_gpu_active = False
    reserved_cores = max(2, int(math.ceil(cpus * 0.2)))
    headroom = max(1, cpus - int(math.ceil(load1)) - reserved_cores)
    cpu_limit = min(2 if training_gpu_active else 4, headroom)
    if free_ram_gib is not None:
        cpu_limit = min(cpu_limit, max(1, int(free_ram_gib // 1.5)))
    if gpus and any(gpu["free_mib"] < 8192 for gpu in gpus):
        cpu_limit = min(cpu_limit, 1)
    if load1 > cpus * 0.75:
        cpu_limit = min(cpu_limit, 1)
    workers = max(1, min(int(requested or cpu_limit), cpu_limit))
    return {
        "workers": workers,
        "requested_workers": requested,
        "cpu_count": cpus,
        "load_1m": load1,
        "reserved_cpu_cores": reserved_cores,
        "free_ram_gib": free_ram_gib,
        "training_gpu_active": training_gpu_active,
        "gpu_snapshot": gpus,
        "policy": "CPU inference; reserve CPU headroom; cap at 2 workers during GPU training and 4 while idle",
    }


def worker_init(variant: str, stage: str, checkpoint: str, config_path: str) -> None:
    global _WORKER
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = matched.resolved(Path(config_path))
    model = CoCapIQN.load(checkpoint, device="cpu").eval()
    expected = cfg["iqn"]
    if model.config.self_feature_dim != int(expected["self_feature_dim"]):
        raise ValueError("checkpoint self-token dimension does not match evaluation config")
    if model.config.pursuer_feature_dim != int(expected["pursuer_feature_dim"]):
        raise ValueError("checkpoint friend-token dimension does not match evaluation config")
    _WORKER = {"variant": variant, "stage": stage, "cfg": cfg, "model": model, "config_path": config_path}


class EvidenceDiagnostics:
    def __init__(self, variant: str, scene: str):
        self.variant = variant
        self.scene = scene
        self.direct_count = 0
        self.one_hop_count = 0
        self.informed_count = 0
        self.evidence_slots = 0
        self.evidence_occupied = 0
        self.transitions = 0
        self.previous: dict[int, float] = {}
        self.steps = 0
        self.source_counts: dict[str, int] = {}
        self.lineage_hops: list[int] = []
        self.ages: list[int] = []
        self.z_positive = 0
        self.z_checks = 0
        self.global_availability_steps = 0
        self.global_available_steps = 0
        self.enemy_tokens = 0
        self.enemy_token_capacity = 0
        self.truncation_events = 0
        self.target_token_shortfall_events = 0
        self.active_target_capacity_overflow_events = 0
        self.capture_step: int | None = None
        self.all_zero_after_capture_step: int | None = None

    def transition(self, env, local, global_state, q, greedy, active, state, phase, step, outcome):
        del global_state, q, greedy, state, phase
        active_targets = sum(not evader.deactivated for evader in env.evaders)
        self.steps += 1
        self.global_availability_steps += 1
        self.global_available_steps += int(active_targets > 0)
        for row, index in zip(local, active):
            evidence = float(np.asarray(row["self"])[-1])
            self.evidence_slots += 1
            self.evidence_occupied += int(evidence > 0.0)
            if index in self.previous and evidence != self.previous[index]:
                self.transitions += 1
            self.previous[index] = evidence
            direct = bool(env._policy_direct_enemy_ids(int(index)))
            self.direct_count += int(direct)
            friend_rows = np.asarray(row["pursuers"], dtype=np.float32)
            one_hop = any(float(friend[-1]) > 0.0 for friend in friend_rows if np.any(friend))
            self.one_hop_count += int(direct or one_hop)
            self.informed_count += int(evidence > 0.0 or one_hop)
            if self.variant == "z05":
                meta = outcome.infos[index]["replay_metadata"]
                self.z_checks += 1
                self.z_positive += int(evidence > 0.0)
                source = str(meta.get("z_dominant_source", "zero"))
                self.source_counts[source] = self.source_counts.get(source, 0) + 1
                self.lineage_hops.append(int(meta.get("z_lineage_hops", -1)))
                self.ages.append(int(meta.get("z_source_age_steps", -1)))
        if self.variant == "global_oracle":
            target_capacity = int(env.per_cfg["max_evader_num"])
            expected_visible_targets = min(active_targets, target_capacity)
            for observation in local:
                mask = np.asarray(observation["masks"], dtype=bool)
                types = np.asarray(observation["types"], dtype=int)
                slots = (types == 2) & mask
                occupied = int(slots.sum())
                self.enemy_tokens += occupied
                self.enemy_token_capacity += target_capacity
                shortfall = occupied < expected_visible_targets
                overflow = active_targets > target_capacity
                self.target_token_shortfall_events += int(shortfall)
                self.active_target_capacity_overflow_events += int(overflow)
                self.truncation_events += int(shortfall or overflow)
        captured_now = bool(env.evaders and all(evader.deactivated and not evader.collision for evader in env.evaders))
        if captured_now and self.capture_step is None:
            self.capture_step = int(step)
        if self.variant == "z05" and self.capture_step is not None and self.all_zero_after_capture_step is None:
            if len(env.z_state) == 0 or float(np.max(env.z_state)) < 0.10:
                self.all_zero_after_capture_step = int(step) - self.capture_step

    def finish(self, env) -> dict[str, Any]:
        base = {
            "direct_visible_pursuer_count_sum": self.direct_count,
            "one_hop_informed_pursuer_count_sum": self.one_hop_count,
            "evidence_slot_occupancy": self.evidence_occupied / max(self.evidence_slots, 1),
            "evidence_transitions": self.transitions,
            "decision_steps": self.steps,
        }
        if self.variant == "local_binary":
            base.update({"direct_visible_pursuer_count_mean_per_step": self.direct_count / max(self.steps, 1), "one_hop_informed_pursuer_count_mean_per_step": self.one_hop_count / max(self.steps, 1)})
        elif self.variant == "z05":
            hops = [value for value in self.lineage_hops if value >= 0]
            ages = [value for value in self.ages if value >= 0]
            base.update({
                "z_positive_count": self.z_positive,
                "z_source_counts": self.source_counts,
                "z_lineage_hops_max": max(hops, default=-1),
                "z_lineage_hops_mean": float(np.mean(hops)) if hops else None,
                "evidence_age_mean_steps": float(np.mean(ages)) if ages else None,
                "evidence_age_p90_steps": float(np.percentile(ages, 90)) if ages else None,
                "capture_step": self.capture_step,
                "capture_after_all_z_zero_steps": self.all_zero_after_capture_step,
                "capture_after_all_z_zero_seconds": None if self.all_zero_after_capture_step is None else self.all_zero_after_capture_step * float(env.pursuers[0].dt * env.pursuers[0].N),
                "evidence_release_seconds": None if self.all_zero_after_capture_step is None else self.all_zero_after_capture_step * float(env.pursuers[0].dt * env.pursuers[0].N),
            })
        else:
            base.update({
                "active_target_global_availability_rate": self.global_available_steps / max(self.global_availability_steps, 1),
                "enemy_token_occupancy": self.enemy_tokens / max(self.enemy_token_capacity, 1),
                "enemy_tokens_visible_sum": self.enemy_tokens,
                "enemy_token_capacity_slots": self.enemy_token_capacity,
                "entity_truncation_events": self.truncation_events,
                "target_token_shortfall_events": self.target_token_shortfall_events,
                "active_target_capacity_overflow_events": self.active_target_capacity_overflow_events,
            })
        return base


def one_episode(task: tuple[str, int, int | None]) -> dict[str, Any]:
    scene, seed, max_steps = task
    variant = str(_WORKER["variant"])
    cfg = _WORKER["cfg"]
    model = _WORKER["model"]
    holder: dict[str, Any] = {}

    def factory(requested_scene: str, requested_seed: int):
        env, obs = matched.make_env(cfg, requested_scene, requested_seed)
        holder["env"] = env
        return env, obs

    diag = EvidenceDiagnostics(variant, scene)
    row = run_episode(model, scene, seed, "cpu", max_steps=max_steps, on_transition=diag.transition, env_factory=factory)
    env = holder["env"]
    record = env.episode_record(task="coverage" if scene == "coverage" else "mix")
    row["ce_max"] = float(record["coverage_ce_center_max"])
    row["area_cv"] = float(record["coverage_strict_area_cv"])
    ce_step = int(record.get("post_capture_coverage_step", -1))
    row["ce_seconds"] = None if ce_step < 0 else ce_step * float(env.pursuers[0].dt * env.pursuers[0].N)
    mission_summary = row["mission_events"]["summary"]
    ring2 = mission_summary.get("capture_region_2_to_3", {})
    ring3 = mission_summary.get("capture_region_2_to_geometry", {})
    row["ring2_seen"] = bool(ring2.get("opportunities", 0))
    row["ring3_reached"] = bool(ring2.get("completed", 0) or ring3.get("completed", 0))
    row["evidence_diagnostics"] = diag.finish(env)
    row["censored"] = not bool(row["safe_complete"] if scene == "mixed" else (row["captured"] if scene == "capture" else row["ce_success"]))
    return row


def _timing(values: list[float | None]) -> dict[str, Any]:
    items = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "n": len(items),
        "mean": float(np.mean(items)) if items else None,
        "median": float(np.percentile(items, 50)) if items else None,
        "p90": float(np.percentile(items, 90)) if items else None,
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row.get(key, False)) for row in rows) / max(len(rows), 1)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for scene in SCENES:
        rows = [row for row in records if row["scene"] == scene]
        censored = sum(bool(row["censored"]) for row in rows)
        if scene == "coverage":
            result[scene] = {
                "episodes": len(rows),
                "strict_ce_rate": _rate(rows, "ce_success"),
                "ce_rms": _timing([row["ce_rms"] for row in rows]),
                "ce_max": _timing([row["ce_max"] for row in rows]),
                "area_cv": _timing([row["area_cv"] for row in rows]),
                "collision_rate": _rate(rows, "collision"),
                "time_to_ce_seconds": _timing([row["ce_seconds"] for row in rows if row["ce_seconds"] is not None]),
                "censored_episodes": censored,
                "censored_fraction": censored / max(len(rows), 1),
            }
        elif scene == "capture":
            result[scene] = {
                "episodes": len(rows),
                "normal_capture_rate": _rate(rows, "normal_capture"),
                "capture_rate": _rate(rows, "captured"),
                "stationary_capture_rate": _rate(rows, "stationary_capture"),
                "ring2_seen_rate": _rate(rows, "ring2_seen"),
                "ring3_reached_rate": _rate(rows, "ring3_reached"),
                "capture_seconds": _timing([row["capture_seconds"] for row in rows if row["captured"]]),
                "collision_rate": _rate(rows, "collision"),
                "censored_episodes": censored,
                "censored_fraction": censored / max(len(rows), 1),
            }
        else:
            result[scene] = {
                "episodes": len(rows),
                "safe_complete_rate": _rate(rows, "safe_complete"),
                "capture_rate": _rate(rows, "captured"),
                "normal_capture_rate": _rate(rows, "normal_capture"),
                "post_capture_ce_rate": _rate(rows, "ce_success"),
                "collision_rate": _rate(rows, "collision"),
                "recovery_time_seconds": _timing([row["recovery_seconds"] for row in rows if row["recovery_seconds"] is not None]),
                "mission_time_seconds": _timing([row["mission_seconds"] for row in rows if row["safe_complete"]]),
                "censored_episodes": censored,
                "censored_fraction": censored / max(len(rows), 1),
            }
    return result


def evaluate(
    variant: str,
    stage: str,
    checkpoint: Path,
    output: Path,
    episodes: int,
    seed_base: int,
    max_steps: int | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evaluator output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cfg_path = CONFIGS[variant][stage]
    checkpoint_sha_before = sha256_file(checkpoint)
    worker_policy = choose_workers(workers)
    tasks = [
        (scene, int(seed_base + scene_index * 100_000 + index), max_steps)
        for scene_index, scene in enumerate(SCENES)
        for index in range(episodes)
    ]
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    progress_path = output / "progress.json"
    atomic_json(progress_path, {"status": "running", "completed": 0, "total": len(tasks), "workers": worker_policy})
    with ProcessPoolExecutor(
        max_workers=worker_policy["workers"],
        initializer=worker_init,
        initargs=(variant, stage, str(checkpoint), str(cfg_path)),
    ) as pool:
        futures = [pool.submit(one_episode, task) for task in tasks]
        for future in as_completed(futures):
            records.append(future.result())
            elapsed = time.monotonic() - started
            atomic_json(progress_path, {
                "status": "running",
                "completed": len(records),
                "total": len(tasks),
                "elapsed_seconds": elapsed,
                "eta_seconds": elapsed / max(len(records), 1) * (len(tasks) - len(records)),
                "workers": worker_policy,
            })
    records.sort(key=lambda row: (SCENES.index(row["scene"]), row["seed"]))
    checkpoint_sha_after = sha256_file(checkpoint)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise AssertionError("read-only evaluator observed checkpoint mutation")
    report = {
        "schema": "iqn-evidence-evaluation-v1",
        "status": "complete",
        "variant": variant,
        "stage": stage,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha_before,
        "config": str(cfg_path),
        "episodes_per_scene": episodes,
        "seed_base": seed_base,
        "seed_manifest": [row["seed"] for row in records],
        "scenes": list(SCENES),
        "paired_seed_rule": "seed_base + scene_index*100000 + episode_index",
        "summary": summarize(records),
        "evidence_diagnostics": {
            scene: {
                "episode_count": sum(row["scene"] == scene for row in records),
                "aggregate": _aggregate_diagnostics([row["evidence_diagnostics"] for row in records if row["scene"] == scene]),
            }
            for scene in SCENES
        },
        "records": records,
        "resource_policy": worker_policy,
        "evaluation_process_isolated": True,
        "optimizer_updates": 0,
        "replay_updates": 0,
        "training_rng_mutation": False,
        "checkpoint_mutation": False,
        "elapsed_seconds": time.monotonic() - started,
        "completed_at": now(),
    }
    atomic_json(output / "report.json", report)
    atomic_json(progress_path, {"status": "complete", "completed": len(records), "total": len(records), "workers": worker_policy})
    return report


def _aggregate_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key in row})
    result: dict[str, Any] = {}
    for key in keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            result[key] = float(sum(values)) if key.endswith("_sum") or key.endswith("_count") or key.endswith("_events") else _timing([float(value) for value in values])
        elif all(isinstance(value, dict) for value in values):
            merged: dict[str, int] = {}
            for value in values:
                for name, count in value.items():
                    merged[name] = merged.get(name, 0) + int(count)
            result[key] = merged
        else:
            result[key] = values
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("local_binary", "z05", "global_oracle"), required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    report = evaluate(args.variant, args.stage, args.checkpoint, args.output, args.episodes, args.seed_base, args.max_steps, args.workers)
    print(json.dumps({"status": "complete", "variant": args.variant, "stage": args.stage, "episodes_per_scene": args.episodes, "workers": report["resource_policy"]["workers"], "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
