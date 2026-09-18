#!/usr/bin/env python3
"""B3 scalar local-evidence z replacement: collection, BC, and matched rollout."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch
import torch.nn.functional as F

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.evaluation.mission_events import MissionEventTracker, snapshot
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.forward_final import CONTRACT
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.trainer import deep_update, load_config, set_global_config
from tools.collect_iqn_aw_teacher_dataset_20260903 import fixed_midpoint_q, sha256_file
from tools.evaluate_vxy_stage2_cross_retention_20260903 import ring_count
from tools.iqn_cleanup_b1_20260917 import (
    DEFAULT_TEMPERATURE,
    _stratified_train_subset,
    _temperature,
)
from tools.rollout_voradj_visual import act_evaders
from tools.run_forward_final_bridge_20260908 import atomic_json


SCHEMA = "iqn-zstate-b3-v1"
TEACHER = ROOT / "artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage1_4v1_step_2000000.pt"
TEACHER_SHA = "2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89"
CONFIG = ROOT / "configs/experiments/iqn_zstate_b3_20260918/z_state.yaml"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts/2026-09-18_iqn_zstate_b3"
B1_REPORT = ROOT / "artifacts/2026-09-17_iqn_cleanup_b1/student_report.json"
PHASE_NAMES = {0: "pre_capture", 1: "post_capture", 2: "pure_coverage"}
ROLE_NAMES = {0: "direct", 1: "pursuing_memory", 2: "support", 3: "coverage"}
SOURCE_IDS = {"zero": 0, "direct": 1, "self_decay": 2, "neighbor": 3}


def resolved_config(scene: str) -> dict[str, Any]:
    root = load_config(str(CONFIG))
    task = "voradj_coverage" if scene == "coverage" else "voradj"
    config = deep_update(root, root["tasks"][task])
    if scene == "capture":
        config = copy.deepcopy(config)
        config.setdefault("voradj", {})["capture_episode_ends_on_capture"] = True
        config["voradj"]["capture_episode_success_on_capture"] = True
    return config


def make_env(scene: str, seed: int) -> tuple[VorAdjEnv, list[Mapping[str, np.ndarray] | None]]:
    config = resolved_config(scene)
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=int(seed))
    observations = list(env.reset())
    if any(
        obs is not None
        and (obs["self"].shape != (9,) or obs["pursuers"].shape[-1] != 7)
        for obs in observations
    ):
        raise AssertionError("B3 z-token observation shape drift")
    return env, observations


def assert_contract() -> dict[str, Any]:
    actual = sha256_file(TEACHER)
    if actual != TEACHER_SHA:
        raise ValueError(f"Final teacher SHA mismatch: {actual}")
    config = resolved_config("mixed")
    b1 = load_config(str(ROOT / "configs/experiments/iqn_cleanup_emergent_b1_20260917/no_is_pursuing.yaml"))
    b1 = deep_update(b1, b1["tasks"]["voradj"])
    if config["reward"] != b1["reward"]:
        raise AssertionError("reward drift")
    if config["perception"].get("include_is_pursuing", True):
        raise AssertionError("B3 policy still exposes is_pursuing")
    if not config["perception"].get("include_z_state", False):
        raise AssertionError("B3 z policy feature is disabled")
    env, _ = make_env("mixed", 2026091800)
    decision_seconds = float(env.pursuers[0].dt * env.pursuers[0].N)
    half_life_steps = math.log(0.5) / math.log(float(config["z_state"]["lambda"]))
    return {
        "teacher_sha256": actual,
        "lambda": float(config["z_state"]["lambda"]),
        "eta": float(config["z_state"]["eta"]),
        "decision_seconds": decision_seconds,
        "half_life_steps": half_life_steps,
        "half_life_seconds": half_life_steps * decision_seconds,
        "reward_unchanged": True,
        "topology_unchanged": True,
    }


def _episode_path(dataset_root: Path, scene: str, index: int) -> Path:
    return dataset_root / "shards" / f"{scene}_{index:03d}.npz"


def _role(metadata: Mapping[str, Any]) -> int:
    direct = int(metadata.get("vct_ls_direct_enemy_count", 0) or 0) > 0
    if direct:
        return 0
    if bool(metadata.get("support_candidate", False)):
        return 2
    if bool(metadata.get("effective_pursuing", False)):
        return 1
    return 3



def _place_targeted_ring3(env: VorAdjEnv) -> list[Mapping[str, np.ndarray] | None]:
    """Create a local, geometry-only ring3 category-coverage initial state."""

    if len(env.pursuers) < 4 or not env.evaders:
        raise ValueError("ring3 coverage requires four pursuers and one evader")
    xl, xr, yb, yt = env._bounds()
    ring_radius = 9.25
    outer_radius = 18.0
    ring_angles = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
    obstacle_margin = 1.0

    def clear(point: np.ndarray, radius: float) -> bool:
        if not (xl + radius <= point[0] <= xr - radius and yb + radius <= point[1] <= yt - radius):
            return False
        return all(
            np.linalg.norm(point - np.asarray([obs.x, obs.y], dtype=float))
            > radius + float(obs.r) + obstacle_margin
            for obs in env.obstacles
        )

    chosen = None
    for cx in np.linspace(xl + outer_radius + 3.0, xr - outer_radius - 3.0, 7):
        for cy in np.linspace(yb + outer_radius + 3.0, yt - outer_radius - 3.0, 7):
            center = np.asarray([cx, cy], dtype=float)
            positions = [
                center + ring_radius * np.asarray([math.cos(angle), math.sin(angle)])
                for angle in ring_angles
            ]
            positions.append(center + outer_radius * np.asarray([math.cos(math.pi / 3.0), math.sin(math.pi / 3.0)]))
            if clear(center, float(env.evaders[0].r)) and all(
                clear(point, float(env.pursuers[index].r))
                for index, point in enumerate(positions)
            ):
                chosen = (center, positions)
                break
        if chosen is not None:
            break
    if chosen is None:
        raise RuntimeError("no obstacle-clear ring3 placement found")
    center, positions = chosen
    target = env.evaders[0]
    target.x, target.y = map(float, center)
    target.velocity = np.zeros(2, dtype=float)
    target.deactivated = False
    target.collision = False
    for index, (pursuer, point) in enumerate(zip(env.pursuers, positions)):
        pursuer.x, pursuer.y = map(float, point)
        pursuer.velocity = np.zeros(2, dtype=float)
        pursuer.deactivated = False
        pursuer.collision = False
        pursuer.theta = float(math.atan2(center[1] - point[1], center[0] - point[0]))
    env._pursuing_release_counters = [0] * len(env.pursuers)
    env._pursuing_flags_initialized = False
    env._reset_z_state()
    env._invalidate_voronoi_cache()
    data = env._capture_voronoi_map()
    env.last_task_labels = env._task_labels_from_map(data, update_effective=True)
    env._advance_z_state(data)
    observations = list(env.get_observations())
    if ring_count(env) < 3:
        raise AssertionError("targeted ring3 placement did not satisfy the diagnostic")
    return observations

def collect_episode(
    teacher: CoCapIQN,
    scene: str,
    seed: int,
    episode_id: int,
    device: str,
    targeted_ring3: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    env, observations = make_env(scene, seed)
    if targeted_ring3:
        observations = _place_targeted_ring3(env)
    from cocap_voradj.control.apf import ApfAgent

    apf = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    rows: dict[str, list[Any]] = {}
    capture_seen = False
    capture_types: list[str] = []
    for timestep in range(1, env.episode_max_length + 1):
        active = [index for index, obs in enumerate(observations) if obs is not None]
        if not active:
            break
        teacher_view = env.get_policy_observations(True, False)
        student_view = env.get_policy_observations(False, True)
        teacher_local = [teacher_view[index] for index in active]
        q_values, actions = fixed_midpoint_q(teacher, teacher_local, device)
        current_phase = 2 if scene == "coverage" else (1 if capture_seen else 0)
        current_ring = ring_count(env)
        z_snapshot = env.z_state_dict()
        commands: list[int | None] = [None] * len(env.pursuers)
        for row, index in enumerate(active):
            commands[index] = int(actions[row])
        outcome = env.step(commands, act_evaders(env, apf))
        events = list(env.last_capture_events)
        capture_types.extend(str(event.get("capture_type", "unknown")) for event in events)
        capture_seen = capture_seen or bool(events)
        for row, index in enumerate(active):
            observation = student_view[index]
            assert observation is not None
            for key, value in observation.items():
                rows.setdefault(f"obs.{key}", []).append(np.asarray(value))
            metadata = outcome.infos[index]["replay_metadata"]
            scalars = {
                "teacher_q": np.asarray(q_values[row], dtype=np.float32),
                "greedy_action": np.int64(actions[row]),
                "episode": np.int32(episode_id),
                "environment_seed": np.int64(seed),
                "timestep": np.int32(timestep),
                "agent": np.int16(index),
                "scene_id": np.uint8({"mixed": 0, "coverage": 1}[scene]),
                "phase_id": np.uint8(current_phase),
                "role_id": np.uint8(_role(metadata)),
                "direct": np.bool_(z_snapshot["last_direct"][index]),
                "support": np.bool_(metadata.get("support_candidate", False)),
                "pursuing": np.bool_(metadata.get("effective_pursuing", False)),
                "ring_count": np.int8(current_ring),
                "ring2": np.bool_(current_ring >= 2),
                "ring3": np.bool_(current_ring >= 3),
                "capture_transition": np.bool_(bool(events)),
                "z": np.float32(z_snapshot["values"][index]),
                "z_neighbor_previous_max": np.float32(z_snapshot["last_neighbor_max"][index]),
                "z_source_id": np.uint8(SOURCE_IDS[z_snapshot["last_source"][index]]),
                "z_lineage_hops": np.int16(z_snapshot["lineage_hops"][index]),
                "z_source_age_steps": np.int32(z_snapshot["source_age_steps"][index]),
                "next_z": np.float32(env.z_state[index]),
                "reward": np.float32(outcome.rewards[index]),
                "terminated": np.bool_(metadata["terminated"]),
                "truncated": np.bool_(metadata["truncated"]),
            }
            for key, value in scalars.items():
                rows.setdefault(key, []).append(value)
        observations = list(outcome.observations)
        if all(outcome.dones):
            break
    arrays = {key: np.stack(values, axis=0) for key, values in rows.items()}
    record = env.episode_record(task="coverage" if scene == "coverage" else "mix")
    return arrays, {
        "episode": int(episode_id),
        "scene": scene,
        "seed": int(seed),
        "targeted_ring3_category_coverage": bool(targeted_ring3),
        "rows": int(len(arrays["episode"])),
        "steps": int(timestep),
        "captured": bool(record.get("captured", False)),
        "ce_success": bool(env.post_capture_coverage_success),
        "collision": bool(record.get("collision_event", False)),
        "capture_types": capture_types,
        "ring2_rows": int(arrays["ring2"].sum()),
        "ring3_rows": int(arrays["ring3"].sum()),
        "direct_rows": int(arrays["direct"].sum()),
        "support_rows": int(arrays["support"].sum()),
        "post_capture_rows": int((arrays["phase_id"] == 1).sum()),
        "pure_coverage_rows": int((arrays["phase_id"] == 2).sum()),
    }


def collect_dataset(dataset_root: Path, episodes: int, seed: int, device: str) -> dict[str, Any]:
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "shards").mkdir(parents=True, exist_ok=True)
    teacher = CoCapIQN.load(str(TEACHER), device=device).eval()
    entries: list[dict[str, Any]] = []
    episode_id = 0
    for scene_offset, scene in enumerate(("mixed", "coverage")):
        for index in range(episodes):
            path = _episode_path(dataset_root, scene, index)
            episode_seed = seed + scene_offset * 100000 + index
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    summary = json.loads(str(saved["episode_summary"].item()))
                if int(summary["seed"]) != episode_seed:
                    raise ValueError(f"existing shard seed mismatch: {path}")
            else:
                arrays, summary = collect_episode(
                    teacher, scene, episode_seed, episode_id, device
                )
                arrays["episode_summary"] = np.asarray(json.dumps(summary, sort_keys=True))
                np.savez_compressed(path, **arrays)
            entries.append({
                **summary,
                "path": str(path.relative_to(dataset_root)),
                "sha256": sha256_file(path),
            })
            episode_id += 1
            atomic_json(
                dataset_root / "manifest.json",
                {
                    "schema": SCHEMA,
                    "status": "running",
                    "teacher_sha256": TEACHER_SHA,
                    "episodes_per_scene": episodes,
                    "completed": len(entries),
                    "shards": entries,
                },
            )
    if not any(int(item["ring3_rows"]) > 0 for item in entries):
        targeted_path = dataset_root / "shards" / "mixed_targeted_ring3.npz"
        targeted_seed = 2026094101
        if targeted_path.exists():
            with np.load(targeted_path, allow_pickle=False) as saved:
                summary = json.loads(str(saved["episode_summary"].item()))
        else:
            arrays, summary = collect_episode(
                teacher,
                "mixed",
                targeted_seed,
                episode_id,
                device,
                targeted_ring3=True,
            )
            arrays["episode_summary"] = np.asarray(json.dumps(summary, sort_keys=True))
            np.savez_compressed(targeted_path, **arrays)
        if int(summary["ring3_rows"]) <= 0:
            raise AssertionError("targeted ring3 shard contains no ring3 rows")
        entries.append({
            **summary,
            "path": str(targeted_path.relative_to(dataset_root)),
            "sha256": sha256_file(targeted_path),
        })

    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "teacher_sha256": TEACHER_SHA,
        "episodes_per_scene": episodes,
        "seed_base": seed,
        "row_count": int(sum(item["rows"] for item in entries)),
        "coverage": {
            key: int(sum(item[key] for item in entries))
            for key in (
                "ring2_rows",
                "ring3_rows",
                "direct_rows",
                "support_rows",
                "post_capture_rows",
                "pure_coverage_rows",
            )
        },
        "native_z_recording": True,
        "old_replay_reused": False,
        "targeted_category_coverage": [
            {
                "category": "ring3",
                "reason": "required BC coverage was absent from the fixed 20-seed natural teacher sample",
                "seed_source": "trusted C1 seed base",
                "physical_intervention": "three pursuers in legacy 8.0-10.5 outer annulus; no reward/topology/policy change",
            }
        ] if any(item.get("targeted_ring3_category_coverage", False) for item in entries) else [],
        "shards": entries,
    }
    atomic_json(dataset_root / "manifest.json", manifest)
    return manifest


def load_dataset(dataset_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads((dataset_root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("B3 dataset is not complete")
    chunks: dict[str, list[np.ndarray]] = {}
    for item in manifest["shards"]:
        path = dataset_root / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"dataset shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as shard:
            for key in shard.files:
                if key != "episode_summary":
                    chunks.setdefault(key, []).append(np.asarray(shard[key]))
    arrays = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    if len(arrays["episode"]) != int(manifest["row_count"]):
        raise ValueError("dataset row count mismatch")
    if not np.array_equal(arrays["teacher_q"].argmax(1), arrays["greedy_action"]):
        raise ValueError("teacher Q/action mismatch")
    return arrays, manifest


def observation_tree(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key.removeprefix("obs."): np.asarray(value)
        for key, value in arrays.items()
        if key.startswith("obs.")
    }


def batch_obs(obs: Mapping[str, np.ndarray], indices: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    return stack_obs(
        [{key: value[index] for key, value in obs.items()} for index in indices],
        device,
    )


def midpoint_tau(device: str, count: int) -> torch.Tensor:
    return (torch.arange(count, device=device, dtype=torch.float32) + 0.5) / float(count)


@torch.no_grad()
def predict_q(
    model: CoCapIQN,
    obs: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: str,
    batch_size: int = 8192,
) -> np.ndarray:
    values = []
    for start in range(0, len(indices), batch_size):
        take = indices[start : start + batch_size]
        output = model(
            batch_obs(obs, take, device),
            num_tau=32,
            mode="voradj",
            tau=midpoint_tau(device, 32),
        )
        values.append(output["q_values"].mean(1).detach().cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def metric_summary(
    model: CoCapIQN,
    obs: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: str,
    temperature: float,
) -> dict[str, Any]:
    student_q = predict_q(model, obs, indices, device)
    teacher_q = np.asarray(arrays["teacher_q"])[indices]
    teacher_action = teacher_q.argmax(1)
    student_action = student_q.argmax(1)
    target = torch.as_tensor(teacher_q)
    student = torch.as_tensor(student_q)
    target_lp = F.log_softmax(
        (target - target.max(1, keepdim=True).values) / temperature, dim=1
    )
    student_lp = F.log_softmax(student, dim=1)
    kl = (target_lp.exp() * (target_lp - student_lp)).sum(1).numpy()
    tdiff = teacher_q[:, :, None] - teacher_q[:, None, :]
    sdiff = student_q[:, :, None] - student_q[:, None, :]
    rank = (np.sign(tdiff) == np.sign(sdiff)).mean((1, 2))
    regret = teacher_q.max(1) - teacher_q[np.arange(len(teacher_q)), student_action]

    masks = {
        "all": np.ones(len(indices), dtype=bool),
        "pre_capture": arrays["phase_id"][indices] == 0,
        "post_capture": arrays["phase_id"][indices] == 1,
        "pure_coverage": arrays["phase_id"][indices] == 2,
        "direct": arrays["role_id"][indices] == 0,
        "pursuing_memory": arrays["role_id"][indices] == 1,
        "support": arrays["role_id"][indices] == 2,
        "coverage": arrays["role_id"][indices] == 3,
        "ring2": arrays["ring2"][indices],
        "ring3": arrays["ring3"][indices],
        "normal_capture_transition": arrays["capture_transition"][indices],
    }
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        result[name] = {
            "rows": count,
            "action_agreement": float(np.mean(student_action[mask] == teacher_action[mask])) if count else None,
            "categorical_kl": float(kl[mask].mean()) if count else None,
            "q_regret": float(regret[mask].mean()) if count else None,
            "q_ranking_agreement": float(rank[mask].mean()) if count else None,
        }
    return result


def save_student(path: Path, model: CoCapIQN, extra: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA,
            "config": dataclasses.asdict(model.config),
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "extra": dict(extra),
        },
        path,
    )


def train_student(
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    artifact_root: Path,
    device: str,
    epochs: int,
    batch_size: int,
    max_train_rows: int,
    seed: int,
) -> tuple[CoCapIQN, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    teacher = CoCapIQN.load(str(TEACHER), device=device).eval()
    student, transfer = CoCapIQN.make_z_state_student(teacher)
    student.to(device)
    obs = observation_tree(arrays)
    episodes = np.asarray(arrays["episode"])
    validation_mask = episodes % 5 == 0
    train_idx = np.flatnonzero(~validation_mask)
    valid_idx = np.flatnonzero(validation_mask)
    optimization_idx = _stratified_train_subset(arrays, train_idx, max_train_rows, seed)
    temperature = _temperature() or DEFAULT_TEMPERATURE
    optimizer = torch.optim.Adam(student.parameters(), lr=3e-4, eps=1e-5)
    teacher_q = np.asarray(arrays["teacher_q"], dtype=np.float32)
    history = []
    best_key = (-1.0, float("-inf"))
    checkpoint = artifact_root / "student_zstate.pt"
    for epoch in range(1, epochs + 1):
        student.train()
        losses = []
        order = np.random.permutation(optimization_idx)
        for start in range(0, len(order), batch_size):
            take = order[start : start + batch_size]
            output = student(
                batch_obs(obs, take, device),
                num_tau=8,
                mode="voradj",
                tau=midpoint_tau(device, 8),
            )
            student_q = output["q_values"].mean(1)
            target_q = torch.as_tensor(teacher_q[take], device=device)
            target_lp = F.log_softmax(
                (target_q - target_q.max(1, keepdim=True).values) / temperature,
                dim=1,
            )
            student_lp = F.log_softmax(student_q, dim=1)
            kl = F.kl_div(student_lp, target_lp.exp(), reduction="batchmean")
            scale = target_q.std(1, keepdim=True).clamp_min(1e-3)
            q_loss = F.smooth_l1_loss(
                (student_q - student_q.mean(1, keepdim=True)) / scale,
                (target_q - target_q.mean(1, keepdim=True)) / scale,
            )
            loss = kl + 0.25 * q_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite B3 distillation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        student.eval()
        validation = metric_summary(
            student, obs, arrays, valid_idx, device, temperature
        )
        key = (
            float(validation["all"]["action_agreement"]),
            -float(validation["all"]["categorical_kl"]),
        )
        history.append({
            "epoch": epoch,
            "mean_loss": float(np.mean(losses)),
            "validation": validation,
        })
        if key > best_key:
            best_key = key
            save_student(
                checkpoint,
                student,
                {
                    "best_epoch": epoch,
                    "teacher_sha256": TEACHER_SHA,
                    "dataset_manifest_sha256": hashlib.sha256(
                        (artifact_root / "dataset" / "manifest.json").read_bytes()
                    ).hexdigest(),
                    "transfer": transfer,
                },
            )
        atomic_json(artifact_root / "student_history.json", {"schema": SCHEMA, "history": history})
    student = CoCapIQN.load(str(checkpoint), device=device).eval()
    b1 = json.loads(B1_REPORT.read_text()) if B1_REPORT.exists() else {}
    report = {
        "schema": SCHEMA,
        "teacher_frozen": True,
        "reward_used": False,
        "rl_updates": 0,
        "scratch_rl": False,
        "rows": int(len(arrays["episode"])),
        "optimization_rows": int(len(optimization_idx)),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(valid_idx)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "temperature": float(temperature),
        "dataset_coverage": manifest["coverage"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "transfer": transfer,
        "validation": metric_summary(student, obs, arrays, valid_idx, device, temperature),
        "all_rows": metric_summary(
            student,
            obs,
            arrays,
            np.arange(len(arrays["episode"]), dtype=np.int64),
            device,
            temperature,
        ),
        "b1_all_rows": b1.get("all_rows"),
        "history": history,
    }
    atomic_json(artifact_root / "student_report.json", report)
    return student, report


def timing(values: Sequence[float | None]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    return {
        "n": len(clean),
        "mean": float(np.mean(clean)) if clean else None,
        "p50": float(np.percentile(clean, 50)) if clean else None,
        "p90": float(np.percentile(clean, 90)) if clean else None,
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for policy in ("teacher", "student"):
        result[policy] = {}
        for scene in ("coverage", "capture", "mixed"):
            rows = [row for row in records if row["policy"] == policy and row["scene"] == scene]
            if not rows:
                continue
            rate = lambda key: float(np.mean([bool(row[key]) for row in rows]))
            result[policy][scene] = {
                "episodes": len(rows),
                **{
                    f"{key}_rate": rate(key)
                    for key in (
                        "captured",
                        "normal_capture",
                        "stationary_capture",
                        "collision",
                        "ce_success",
                        "safe_complete",
                    )
                },
                "episode_seconds": timing([row["episode_seconds"] for row in rows]),
                "capture_seconds": timing([row["capture_seconds"] for row in rows]),
                "recovery_seconds": timing([row["recovery_seconds"] for row in rows]),
                "mission_seconds": timing([row["mission_seconds"] for row in rows]),
            }
    return result


@torch.no_grad()
def run_episode(
    policy: CoCapIQN,
    teacher: CoCapIQN,
    policy_name: str,
    scene: str,
    seed: int,
    device: str,
    trace_root: Path,
) -> dict[str, Any]:
    env, observations = make_env(scene, seed)
    from cocap_voradj.control.apf import ApfAgent

    apf = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    tracker = MissionEventTracker(env.pursuers[0].dt * env.pursuers[0].N)
    tracker.observe(snapshot(env, observations), 0)
    capture_step = None
    ce_step = None
    collision = False
    capture_types: list[str] = []
    agreement_rows = 0
    agreement_hits = 0
    z_steps, direct_steps, source_steps, hop_steps, role_steps = [], [], [], [], []
    direct_checks = direct_violations = 0
    support_values: list[float] = []
    coverage_values: list[float] = []
    neighbor_values: list[float] = []
    all_high_steps = all_pursue_steps = 0
    for step in range(1, env.episode_max_length + 1):
        active = [index for index, obs in enumerate(observations) if obs is not None]
        if not active:
            break
        teacher_view = env.get_policy_observations(True, False)
        student_view = env.get_policy_observations(False, True)
        teacher_q, teacher_action = fixed_midpoint_q(
            teacher, [teacher_view[index] for index in active], device
        )
        if policy_name == "teacher":
            action = teacher_action
        else:
            _q, action = fixed_midpoint_q(
                policy, [student_view[index] for index in active], device
            )
            agreement_rows += len(active)
            agreement_hits += int(np.sum(action == teacher_action))
        z_now = env.z_state.copy()
        direct_now = env._z_last_direct.copy()
        z_steps.append(z_now)
        direct_steps.append(direct_now)
        source_steps.append(np.asarray([SOURCE_IDS[value] for value in env._z_last_source], dtype=np.uint8))
        hop_steps.append(env._z_lineage_hops.copy())
        direct_checks += int(direct_now.sum())
        direct_violations += int(np.sum(direct_now & ~np.isclose(z_now, 1.0)))
        active_z = z_now[active]
        all_high_steps += int(len(active_z) > 0 and np.all(active_z >= 0.9))
        all_pursue_steps += int(len(active_z) > 0 and np.all(active_z >= 0.5))
        commands: list[int | None] = [None] * len(env.pursuers)
        for row, index in enumerate(active):
            commands[index] = int(action[row])
        outcome = env.step(commands, act_evaders(env, apf))
        role_row = np.full(len(env.pursuers), -1, dtype=np.int8)
        for index in active:
            metadata = outcome.infos[index]["replay_metadata"]
            role = _role(metadata)
            role_row[index] = role
            if role == 2:
                support_values.append(float(z_now[index]))
            elif role == 3:
                coverage_values.append(float(z_now[index]))
            if env._z_last_source[index] == "neighbor":
                neighbor_values.append(float(env.z_state[index]))
        role_steps.append(role_row)
        events = list(env.last_capture_events)
        capture_types.extend(str(event.get("capture_type", "unknown")) for event in events)
        collision = collision or bool(env.last_collision_events)
        captured = bool(env.evaders and all(evader.deactivated and not evader.collision for evader in env.evaders))
        if captured and capture_step is None:
            capture_step = step
        if env.post_capture_coverage_success and ce_step is None:
            ce_step = step
        tracker.observe(snapshot(env, outcome.observations), step, events)
        observations = list(outcome.observations)
        if all(outcome.dones):
            break
    trace_root.mkdir(parents=True, exist_ok=True)
    trace_path = trace_root / f"{policy_name}_{scene}_{seed}.npz"
    np.savez_compressed(
        trace_path,
        z=np.asarray(z_steps, dtype=np.float32),
        direct=np.asarray(direct_steps, dtype=bool),
        source=np.asarray(source_steps, dtype=np.uint8),
        hops=np.asarray(hop_steps, dtype=np.int16),
        role=np.asarray(role_steps, dtype=np.int8),
    )
    record = env.episode_record(task="coverage" if scene == "coverage" else "mix")
    captured = bool(record.get("captured", False))
    ce = bool(env.post_capture_coverage_success)
    if scene == "capture":
        safe = bool(captured and not collision and record["all_pursuers_active"])
    else:
        safe = bool(ce and (scene == "coverage" or captured) and not collision and record["all_pursuers_active"])
    dt = env.pursuers[0].dt * env.pursuers[0].N
    z_array = np.asarray(z_steps)
    post_below = None
    if capture_step is not None and len(z_array):
        for offset in range(capture_step, len(z_array)):
            if float(np.max(z_array[offset])) < 0.1:
                post_below = offset - capture_step
                break
    return {
        "policy": policy_name,
        "scene": scene,
        "seed": int(seed),
        "captured": captured,
        "normal_capture": captured and "loose" in capture_types,
        "stationary_capture": captured and "stationary" in capture_types,
        "collision": bool(collision or record.get("collision_event", False)),
        "ce_success": ce,
        "safe_complete": safe,
        "steps": int(step),
        "episode_seconds": float(step * dt),
        "capture_seconds": float(capture_step * dt) if capture_step is not None else None,
        "recovery_seconds": float((ce_step - capture_step) * dt) if ce_step is not None and capture_step is not None else None,
        "mission_seconds": float(ce_step * dt) if safe and ce_step is not None else (float(capture_step * dt) if safe and scene == "capture" and capture_step is not None else None),
        "agreement_rows": int(agreement_rows),
        "agreement_hits": int(agreement_hits),
        "trace": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "z_diagnostics": {
            "direct_checks": direct_checks,
            "direct_violations": direct_violations,
            "direct_visible_z_one_rate": 1.0 - direct_violations / max(direct_checks, 1),
            "support_z_mean": float(np.mean(support_values)) if support_values else None,
            "coverage_z_mean": float(np.mean(coverage_values)) if coverage_values else None,
            "neighbor_propagation_count": len(neighbor_values),
            "neighbor_propagation_mean": float(np.mean(neighbor_values)) if neighbor_values else None,
            "max_lineage_hop": int(max((np.max(value) for value in hop_steps), default=-1)),
            "all_swarm_z_ge_0_9_fraction": all_high_steps / max(len(z_steps), 1),
            "all_swarm_z_ge_0_5_fraction": all_pursue_steps / max(len(z_steps), 1),
            "post_capture_steps_to_max_z_lt_0_1": post_below,
            "post_capture_never_released": bool(capture_step is not None and post_below is None and scene != "capture"),
        },
        "mission_events": tracker.finish(step),
    }


def rollout(
    student: CoCapIQN,
    artifact_root: Path,
    device: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    teacher = CoCapIQN.load(str(TEACHER), device=device).eval()
    records = []
    total = episodes * 3 * 2
    atomic_json(artifact_root / "rollout_progress.json", {"status": "running", "completed": 0, "total": total})
    for policy_name, policy in (("teacher", teacher), ("student", student)):
        for scene_index, scene in enumerate(("coverage", "capture", "mixed")):
            for index in range(episodes):
                records.append(
                    run_episode(
                        policy,
                        teacher,
                        policy_name,
                        scene,
                        seed + scene_index * 100000 + index,
                        device,
                        artifact_root / "z_traces",
                    )
                )
                atomic_json(
                    artifact_root / "rollout_progress.json",
                    {"status": "running", "completed": len(records), "total": total},
                )
    student_rows = sum(row["agreement_rows"] for row in records if row["policy"] == "student")
    student_hits = sum(row["agreement_hits"] for row in records if row["policy"] == "student")
    diagnostics = [row["z_diagnostics"] for row in records if row["policy"] == "student"]
    direct_checks = sum(row["direct_checks"] for row in diagnostics)
    direct_violations = sum(row["direct_violations"] for row in diagnostics)
    report = {
        "schema": SCHEMA,
        "episodes_per_scene_policy": episodes,
        "matched_seed_base": seed,
        "teacher_sha256": TEACHER_SHA,
        "student_checkpoint_sha256": sha256_file(artifact_root / "student_zstate.pt"),
        "summary": summarize_records(records),
        "student_on_student_trajectory_teacher_agreement": {
            "rows": student_rows,
            "action_agreement": student_hits / max(student_rows, 1),
        },
        "z_diagnostics": {
            "direct_checks": direct_checks,
            "direct_violations": direct_violations,
            "direct_visible_z_one_rate": 1.0 - direct_violations / max(direct_checks, 1),
            "support_z_mean": float(np.mean([row["support_z_mean"] for row in diagnostics if row["support_z_mean"] is not None])),
            "coverage_z_mean": float(np.mean([row["coverage_z_mean"] for row in diagnostics if row["coverage_z_mean"] is not None])),
            "neighbor_propagation_count": int(sum(row["neighbor_propagation_count"] for row in diagnostics)),
            "neighbor_propagation_mean": float(np.mean([row["neighbor_propagation_mean"] for row in diagnostics if row["neighbor_propagation_mean"] is not None])),
            "max_lineage_hop": int(max(row["max_lineage_hop"] for row in diagnostics)),
            "mean_all_swarm_z_ge_0_9_fraction": float(np.mean([row["all_swarm_z_ge_0_9_fraction"] for row in diagnostics])),
            "mean_all_swarm_z_ge_0_5_fraction": float(np.mean([row["all_swarm_z_ge_0_5_fraction"] for row in diagnostics])),
            "post_capture_release_steps": [
                row["post_capture_steps_to_max_z_lt_0_1"]
                for row in diagnostics
                if row["post_capture_steps_to_max_z_lt_0_1"] is not None
            ],
            "post_capture_never_release_episodes": int(sum(row["post_capture_never_released"] for row in diagnostics)),
        },
        "typical_trace_selection": {
            scene: next(
                (row["trace"] for row in records if row["policy"] == "student" and row["scene"] == scene and row["safe_complete"]),
                next(row["trace"] for row in records if row["policy"] == "student" and row["scene"] == scene),
            )
            for scene in ("coverage", "capture", "mixed")
        },
        "records": records,
    }
    atomic_json(artifact_root / "rollout_report.json", report)
    atomic_json(artifact_root / "rollout_progress.json", {"status": "complete", "completed": total, "total": total})
    return report


def classify(student_report: Mapping[str, Any], rollout_report: Mapping[str, Any] | None) -> str:
    b3 = student_report["all_rows"]
    b1 = student_report.get("b1_all_rows") or {}
    gain = float(b3["all"]["action_agreement"]) - float(b1.get("action_agreement", 0.0))
    if rollout_report is None:
        return "B3_PENDING_ROLLOUT"
    summary = rollout_report["summary"]
    student = summary["student"]
    teacher = summary["teacher"]
    capture_ok = student["capture"]["captured_rate"] >= teacher["capture"]["captured_rate"] - 0.15
    coverage_ok = student["coverage"]["safe_complete_rate"] >= teacher["coverage"]["safe_complete_rate"] - 0.15
    mixed_ok = student["mixed"]["safe_complete_rate"] >= teacher["mixed"]["safe_complete_rate"] - 0.15
    decay_ok = rollout_report["z_diagnostics"]["post_capture_never_release_episodes"] == 0
    if gain >= 0.20 and capture_ok and coverage_ok and mixed_ok and decay_ok:
        return "B3_PASS"
    if gain >= 0.10 and capture_ok:
        return "B3_PARTIAL"
    return "B3_FAIL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("collect", "train", "rollout", "all"), default="all")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset-episodes", type=int, default=10)
    parser.add_argument("--dataset-seed", type=int, default=2026091801)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-train-rows", type=int, default=65536)
    parser.add_argument("--rollout-episodes", type=int, default=20)
    parser.add_argument("--rollout-seed", type=int, default=2026092801)
    parser.add_argument("--seed", type=int, default=2026091801)
    args = parser.parse_args()
    if min(args.dataset_episodes, args.epochs, args.batch_size, args.max_train_rows, args.rollout_episodes) <= 0:
        raise ValueError("all budgets must be positive")
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    contract = assert_contract()
    atomic_json(artifact_root / "contract.json", {"schema": SCHEMA, **contract})
    dataset_root = artifact_root / "dataset"
    manifest = None
    arrays = None
    student_report = None
    rollout_report = None
    if args.stage in {"collect", "all"}:
        manifest = collect_dataset(dataset_root, args.dataset_episodes, args.dataset_seed, args.device)
    if args.stage in {"train", "all"}:
        arrays, manifest = load_dataset(dataset_root)
        _student, student_report = train_student(
            arrays,
            manifest,
            artifact_root,
            args.device,
            args.epochs,
            args.batch_size,
            args.max_train_rows,
            args.seed,
        )
    if args.stage in {"rollout", "all"}:
        checkpoint = artifact_root / "student_zstate.pt"
        student = CoCapIQN.load(str(checkpoint), device=args.device).eval()
        if not student.config.include_z_state or student.config.include_is_pursuing:
            raise ValueError("rollout checkpoint is not a B3 z-state student")
        rollout_report = rollout(
            student,
            artifact_root,
            args.device,
            args.rollout_episodes,
            args.rollout_seed,
        )
        if student_report is None:
            student_report = json.loads((artifact_root / "student_report.json").read_text())
    if student_report is not None:
        final = {
            "schema": SCHEMA,
            "status": "complete" if rollout_report is not None else "offline_complete",
            "head_at_run": os.popen("git rev-parse HEAD").read().strip(),
            "contract": contract,
            "dataset_manifest": str(dataset_root / "manifest.json"),
            "student_report": str(artifact_root / "student_report.json"),
            "rollout_report": str(artifact_root / "rollout_report.json") if rollout_report is not None else None,
            "classification": classify(student_report, rollout_report),
            "rl_fine_tune_launched": False,
            "scratch_rl": False,
        }
        atomic_json(artifact_root / "B3_FINAL_AUDIT.json", final)
        print(json.dumps(final, ensure_ascii=False, indent=2))
    elif manifest is not None:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
