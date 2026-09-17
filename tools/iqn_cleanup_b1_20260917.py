#!/usr/bin/env python3
"""B0/B1 Final IQN role-bit cleanup audit.

This tool is deliberately limited to the B1 information ablation:

* the Final IQN teacher is frozen;
* the existing trusted Final teacher dataset is reused;
* the student sees self/friend geometry without is_pursuing;
* the first learning stage is Q/action distillation only;
* reward, topology, z_i, and scratch RL are never changed or launched.

The role-free friend ordering is physical-only so token order cannot preserve
the removed role bit as an indirect shortcut.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch
import torch.nn.functional as F

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.forward_final import CONTRACT, check_env
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.trainer import deep_update, load_config, set_global_config
from tools.collect_iqn_aw_teacher_dataset_20260903 import fixed_midpoint_q, sha256_file
from tools.rollout_voradj_visual import act_evaders
from tools.run_forward_final_bridge_20260908 import atomic_json, summarize
from cocap_voradj.evaluation.mission_events import MissionEventTracker, snapshot


SCHEMA = "iqn-cleanup-b1-no-is-pursuing-v1"
TEACHER = ROOT / "artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage1_4v1_step_2000000.pt"
TEACHER_SHA = "2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89"


def _default_source_dataset() -> Path:
    """Find the trusted dataset in this worktree or its sibling artifact worktree."""

    candidates = (
        ROOT / "artifacts/2026-09-08_forward_final/c1_dataset",
        ROOT.parent / "cocap-voradj-small-step-ac/artifacts/2026-09-08_forward_final/c1_dataset",
    )
    for candidate in candidates:
        if (candidate / "manifest.json").exists() and (candidate / "shards").is_dir():
            return candidate
    return candidates[0]


SOURCE_DATASET = _default_source_dataset()
NO_BIT_CONFIG = ROOT / "configs/experiments/iqn_cleanup_emergent_b1_20260917/no_is_pursuing.yaml"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts/2026-09-17_iqn_cleanup_b1"
DEFAULT_TEMPERATURE = 0.0031884169327952754
PHASES = {"pre_capture": 0, "post_capture": 1, "pure_coverage": 2}
ROLES = {"direct": 0, "pursuing_memory": 1, "support": 2, "coverage": 3}


def _hash_json(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_teacher() -> None:
    actual = sha256_file(TEACHER)
    if actual != TEACHER_SHA:
        raise ValueError(f"Final teacher SHA mismatch: expected {TEACHER_SHA}, got {actual}")


def _load_dataset(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "forward-final-teacher-dataset-v1":
        raise ValueError("B1 requires the trusted forward-final teacher dataset")
    if manifest.get("teacher_sha256") != TEACHER_SHA:
        raise ValueError("teacher dataset is not paired with the Final stage1 teacher")
    if manifest.get("status") != "complete":
        raise ValueError("teacher dataset is not complete")
    chunks: dict[str, list[np.ndarray]] = {}
    for shard in manifest["shards"]:
        path = root / shard["path"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"dataset shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                chunks.setdefault(key, []).append(np.asarray(data[key]))
    arrays = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    rows = int(manifest["row_count"])
    if len(arrays["episode"]) != rows or not np.array_equal(
        arrays["teacher_q"].argmax(axis=1), arrays["greedy_action"]
    ):
        raise ValueError("teacher dataset row/Q/action contract failed")
    return arrays, {
        "source_root": str(root),
        "source_manifest_sha256": _hash_json(root / "manifest.json"),
        "source_manifest": manifest,
        "rows": rows,
    }


def _role_free_view(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remove role columns and canonicalize friends by physical features."""

    self_obs = np.asarray(arrays["local_obs.self"], dtype=np.float32)
    friends = np.asarray(arrays["local_obs.pursuers"], dtype=np.float32)
    evaders = np.asarray(arrays["local_obs.evaders"], dtype=np.float32)
    obstacles = np.asarray(arrays["local_obs.obstacles"], dtype=np.float32)
    masks = np.asarray(arrays["local_obs.masks"], dtype=bool)
    types = np.asarray(arrays["local_obs.types"], dtype=np.int64)
    if self_obs.shape[1] < 2 or friends.shape[2] < 2:
        raise ValueError("teacher observation is too small for B1 role removal")
    max_p = friends.shape[1]
    friend_start = 1
    friend_end = friend_start + max_p
    role_free_friends = friends[:, :, :-1].copy()
    role_free_masks = masks.copy()
    role_free_types = types.copy()
    for row in range(len(friends)):
        valid = masks[row, friend_start:friend_end] & (types[row, friend_start:friend_end] == 1)
        order = sorted(
            range(max_p),
            key=lambda index: (
                0 if valid[index] else 1,
                tuple(float(value) for value in friends[row, index, :-1]),
                index,
            ),
        )
        role_free_friends[row] = friends[row, order, :-1]
        role_free_masks[row, friend_start:friend_end] = masks[row, friend_start:friend_end][order]
        role_free_types[row, friend_start:friend_end] = types[row, friend_start:friend_end][order]
    return {
        "self": self_obs[:, :-1].copy(),
        "pursuers": role_free_friends,
        "evaders": evaders.copy(),
        "obstacles": obstacles.copy(),
        "masks": role_free_masks,
        "types": role_free_types,
    }


def _batch(obs: Mapping[str, np.ndarray], indices: Sequence[int], device: str) -> dict[str, torch.Tensor]:
    selected = {key: value[np.asarray(indices)] for key, value in obs.items()}
    return stack_obs(
        [
            {key: selected[key][row] for key in selected}
            for row in range(len(np.asarray(indices)))
        ],
        device,
    )


def _midpoint_tau(device: str, num_tau: int = 32) -> torch.Tensor:
    return (torch.arange(num_tau, device=device, dtype=torch.float32) + 0.5) / float(num_tau)


@torch.no_grad()
def _predict_q(model: CoCapIQN, obs: Mapping[str, np.ndarray], indices: Sequence[int], device: str) -> np.ndarray:
    batch = _batch(obs, indices, device)
    q = model(batch, num_tau=32, mode="voradj", tau=_midpoint_tau(device))["q_values"].mean(dim=1)
    return q.detach().cpu().numpy().astype(np.float32)


def _metric_rows(
    model: CoCapIQN,
    obs: Mapping[str, np.ndarray],
    teacher_q: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: str,
    temperature: float,
    batch_size: int = 8192,
) -> dict[str, Any]:
    predictions: list[np.ndarray] = []
    kls: list[np.ndarray] = []
    regrets: list[np.ndarray] = []
    ranks: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        take = indices[start : start + batch_size]
        student_q = _predict_q(model, obs, take, device)
        target = torch.as_tensor(teacher_q[take], dtype=torch.float32)
        target_lp = F.log_softmax((target - target.max(dim=1, keepdim=True).values) / temperature, dim=1)
        student = torch.as_tensor(student_q, dtype=torch.float32)
        student_lp = F.log_softmax(student, dim=1)
        predictions.append(student_q.argmax(axis=1))
        kls.append((target_lp.exp() * (target_lp - student_lp)).sum(dim=1).numpy())
        pred = student_q.argmax(axis=1)
        regrets.append(
            (teacher_q[take].max(axis=1) - teacher_q[take, pred]).astype(np.float32)
        )
        tdiff = teacher_q[take, :, None] - teacher_q[take, None, :]
        sdiff = student_q[:, :, None] - student_q[:, None, :]
        ranks.append((np.sign(tdiff) == np.sign(sdiff)).mean(axis=(1, 2)))
    predicted = np.concatenate(predictions)
    kl = np.concatenate(kls)
    regret = np.concatenate(regrets)
    rank = np.concatenate(ranks)

    def summarize_mask(mask: np.ndarray) -> dict[str, Any]:
        count = int(mask.sum())
        teacher_actions = teacher_q[indices][mask].argmax(axis=1)
        return {
            "rows": count,
            "action_agreement": float(np.mean(predicted[mask] == teacher_actions)) if count else None,
            "categorical_kl": float(kl[mask].mean()) if count else None,
            "q_regret": float(regret[mask].mean()) if count else None,
            "q_pairwise_rank_agreement": float(rank[mask].mean()) if count else None,
        }

    result = {
        "rows": int(len(indices)),
        "action_agreement": float(np.mean(predicted == teacher_q[indices].argmax(axis=1))),
        "categorical_kl": float(kl.mean()),
        "q_regret": float(regret.mean()),
        "q_pairwise_rank_agreement": float(rank.mean()),
        "phase": {},
        "role": {},
    }
    for name, value in PHASES.items():
        result["phase"][name] = summarize_mask(arrays["phase_id"][indices] == value)
    for name, value in ROLES.items():
        result["role"][name] = summarize_mask(arrays["role_id"][indices] == value)
    result["macro_phase_action_agreement"] = float(
        np.mean([item["action_agreement"] for item in result["phase"].values() if item["rows"]])
    )
    return result


def _split_indices(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.asarray(arrays["episode"], dtype=np.int64)
    validation = (episodes // 2) % 5 == 0
    train = np.flatnonzero(~validation)
    valid = np.flatnonzero(validation)
    if not len(train) or not len(valid) or set((episodes[train] // 2).tolist()) & set((episodes[valid] // 2).tolist()):
        raise ValueError("B1 episode-pair split is invalid")
    return train, valid


def _stratified_train_subset(
    arrays: Mapping[str, np.ndarray],
    train_idx: np.ndarray,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    """Bound CPU optimization work while retaining every observed phase/role."""

    if max_rows <= 0 or len(train_idx) <= max_rows:
        return train_idx
    rng = np.random.default_rng(seed)
    strata: dict[tuple[int, int], list[int]] = {}
    phases = np.asarray(arrays["phase_id"])
    roles = np.asarray(arrays["role_id"])
    for index in train_idx.tolist():
        strata.setdefault((int(phases[index]), int(roles[index])), []).append(int(index))
    buckets = [np.asarray(values, dtype=np.int64) for values in strata.values() if values]
    if max_rows < len(buckets):
        raise ValueError("max_train_rows must cover every non-empty phase/role stratum")
    selected: list[int] = []
    remaining: list[int] = []
    quota, extra = divmod(int(max_rows), len(buckets))
    for bucket in buckets:
        take = min(len(bucket), quota + (1 if extra > 0 else 0))
        extra -= int(take > quota)
        chosen = rng.choice(bucket, size=take, replace=False)
        selected.extend(int(value) for value in chosen)
        if take < len(bucket):
            remaining.extend(int(value) for value in np.setdiff1d(bucket, chosen, assume_unique=False))
    if len(selected) < max_rows:
        selected.extend(int(value) for value in rng.choice(
            np.asarray(remaining, dtype=np.int64),
            size=max_rows - len(selected),
            replace=False,
        ))
    return np.asarray(sorted(selected), dtype=np.int64)


def _temperature() -> float:
    report = ROOT / "artifacts/2026-09-08_forward_final/c2_distillation/report.json"
    if report.exists():
        value = float(json.loads(report.read_text()).get("temperature", DEFAULT_TEMPERATURE))
        if value > 0:
            return value
    return DEFAULT_TEMPERATURE


def _save_student(path: Path, student: CoCapIQN, extra: Mapping[str, Any]) -> None:
    payload = {
        "schema": SCHEMA,
        "contract": CONTRACT,
        "teacher_sha256": TEACHER_SHA,
        "config": dataclasses.asdict(student.config),
        "state_dict": {key: value.detach().cpu() for key, value in student.state_dict().items()},
        "extra": dict(extra),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_student(
    arrays: Mapping[str, np.ndarray],
    role_free: Mapping[str, np.ndarray],
    source_meta: Mapping[str, Any],
    artifact_root: Path,
    device: str,
    epochs: int,
    batch_size: int,
    seed: int,
    max_train_rows: int,
) -> tuple[CoCapIQN, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    teacher = CoCapIQN.load(str(TEACHER), device=device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student, transfer = CoCapIQN.make_no_is_pursuing_student(teacher)
    student.to(device)
    temperature = _temperature()
    train_idx, valid_idx = _split_indices(arrays)
    full_train_rows = len(train_idx)
    train_idx = _stratified_train_subset(arrays, train_idx, max_train_rows, seed)
    history: list[dict[str, Any]] = []
    best_key = (-1.0, float("-inf"))
    best_payload: dict[str, Any] | None = None
    optimizer = torch.optim.Adam(student.parameters(), lr=3e-4, eps=1e-5)
    teacher_q = np.asarray(arrays["teacher_q"], dtype=np.float32)
    for epoch in range(1, int(epochs) + 1):
        student.train()
        order = np.random.permutation(train_idx)
        losses = []
        for start in range(0, len(order), int(batch_size)):
            take = order[start : start + int(batch_size)]
            batch = _batch(role_free, take, device)
            # Eight deterministic quadrature points are sufficient for the
            # optimization pass; all reported policy metrics use the Final
            # 32-quantile evaluation contract below.
            output = student(batch, num_tau=8, mode="voradj", tau=_midpoint_tau(device, 8))
            student_q = output["q_values"].mean(dim=1)
            target_q = torch.as_tensor(teacher_q[take], dtype=torch.float32, device=device)
            target_lp = F.log_softmax(
                (target_q - target_q.max(dim=1, keepdim=True).values) / temperature,
                dim=1,
            )
            student_lp = F.log_softmax(student_q, dim=1)
            kl = F.kl_div(student_lp, target_lp.exp(), reduction="batchmean")
            q_scale = target_q.detach().std(dim=1, keepdim=True).clamp_min(1e-3)
            q_loss = F.smooth_l1_loss(
                (student_q - student_q.mean(dim=1, keepdim=True)) / q_scale,
                (target_q - target_q.mean(dim=1, keepdim=True)) / q_scale,
            )
            loss = kl + 0.25 * q_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite B1 distillation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        student.eval()
        validation = _metric_rows(
            student, role_free, teacher_q, arrays, valid_idx, device, temperature
        )
        key = (validation["macro_phase_action_agreement"], -validation["categorical_kl"])
        row = {
            "epoch": epoch,
            "mean_loss": float(np.mean(losses)),
            "validation": validation,
        }
        history.append(row)
        if key > best_key:
            best_key = key
            best_payload = {
                "epoch": epoch,
                "validation": validation,
                "temperature": temperature,
            }
            _save_student(
                artifact_root / "student_no_is_pursuing.pt",
                student,
                {
                    "best_epoch": epoch,
                    "dataset_manifest_sha256": source_meta["source_manifest_sha256"],
                    "transfer": transfer,
                    "distillation": {
                        "objective": "KL(teacher_Q||student_Q) + 0.25 centered-Q smooth-L1",
                        "teacher_frozen": True,
                        "environment_reward_used": False,
                        "epochs": int(epochs),
                        "batch_size": int(batch_size),
                        "temperature": temperature,
                        "ppo_updates": 0,
                        "scratch_rl": False,
                    },
                },
            )
        atomic_json(
            artifact_root / "student_history.json",
            {"schema": SCHEMA, "status": "running", "history": history},
        )
    if best_payload is None:
        raise RuntimeError("student training did not produce a checkpoint")
    student = CoCapIQN.load(str(artifact_root / "student_no_is_pursuing.pt"), device=device).eval()
    report = {
        "schema": SCHEMA,
        "contract": CONTRACT,
        "teacher_sha256": TEACHER_SHA,
        "dataset_manifest_sha256": source_meta["source_manifest_sha256"],
        "rows": int(len(teacher_q)),
        "train_rows": int(len(train_idx)),
        "full_train_rows": int(full_train_rows),
        "max_train_rows": int(max_train_rows),
        "validation_rows": int(len(valid_idx)),
        "student_checkpoint": str(artifact_root / "student_no_is_pursuing.pt"),
        "student_checkpoint_sha256": sha256_file(artifact_root / "student_no_is_pursuing.pt"),
        "student_config": dataclasses.asdict(student.config),
        "transfer": transfer,
        "distillation": {
            "teacher_frozen": True,
            "environment_reward_used": False,
            "ppo_updates": 0,
            "scratch_rl": False,
            "temperature": temperature,
            "optimization_num_tau": 8,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
        },
        "best": best_payload,
        "train": _metric_rows(student, role_free, teacher_q, arrays, train_idx, device, temperature),
        "validation": _metric_rows(student, role_free, teacher_q, arrays, valid_idx, device, temperature),
        "all_rows": _metric_rows(
            student,
            role_free,
            teacher_q,
            arrays,
            np.arange(len(teacher_q), dtype=np.int64),
            device,
            temperature,
        ),
        "history": history,
    }
    atomic_json(artifact_root / "student_report.json", report)
    return student, report


@torch.no_grad()
def _q_for_observations(model: CoCapIQN, observations: Sequence[Mapping[str, np.ndarray]], device: str) -> tuple[np.ndarray, np.ndarray]:
    return fixed_midpoint_q(model, observations, device)


def _no_bit_scene_config(scene: str) -> dict[str, Any]:
    if scene not in {"mixed", "coverage"}:
        raise ValueError(scene)
    root = load_config(str(NO_BIT_CONFIG))
    task = "voradj" if scene == "mixed" else "voradj_coverage"
    return deep_update(root, root["tasks"][task])


def _make_no_bit_env(scene: str, seed: int) -> tuple[VorAdjEnv, list[Mapping[str, np.ndarray] | None]]:
    config = _no_bit_scene_config(scene)
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=int(seed))
    observations = env.reset()
    check_env(env)
    no_bit = env.get_policy_observations(False)
    if any(
        item is not None
        and (item["self"].shape[-1] != 8 or item["pursuers"].shape[-1] != 6)
        for item in no_bit
    ):
        raise AssertionError("no-bit environment observation shape drift")
    return env, no_bit


@torch.no_grad()
def _run_no_bit_episode(
    student: CoCapIQN,
    teacher: CoCapIQN,
    scene: str,
    seed: int,
    device: str,
    max_steps: int | None = None,
) -> dict[str, Any]:
    env, observations = _make_no_bit_env(scene, seed)
    initial = snapshot(env, observations)
    tracker = MissionEventTracker(env.pursuers[0].dt * env.pursuers[0].N)
    tracker.observe(initial, 0)
    from cocap_voradj.control.apf import ApfAgent
    apf = [ApfAgent(e.a, e.w) for e in env.evaders]
    capture_step = None
    ce_step = None
    capture_types: list[str] = []
    collision = False
    boundary = False
    prefix_collision = False
    agreement = {"rows": 0, "hits": 0, "groups": {}}
    horizon = min(env.episode_max_length, max_steps) if max_steps else env.episode_max_length
    for step in range(1, horizon + 1):
        active = [index for index, value in enumerate(observations) if value is not None]
        teacher_observations = env.get_policy_observations(True)
        student_observations = env.get_policy_observations(False)
        teacher_local = [teacher_observations[index] for index in active]
        student_local = [student_observations[index] for index in active]
        teacher_q, teacher_action = _q_for_observations(teacher, teacher_local, device)
        _student_q, student_action = _q_for_observations(student, student_local, device)
        state = snapshot(env, observations)
        phase = "pure_coverage" if scene == "coverage" else ("post_capture" if capture_step is not None else "pre_capture")
        hits = np.asarray(student_action) == np.asarray(teacher_action)
        agreement["rows"] += len(active)
        agreement["hits"] += int(hits.sum())
        commands: list[int | None] = [None] * len(observations)
        for row, index in enumerate(active):
            group = agreement["groups"].setdefault(
                f"phase/{phase}", {"rows": 0, "hits": 0}
            )
            group["rows"] += 1
            group["hits"] += int(hits[row])
            commands[index] = int(student_action[row])
        outcome = env.step(commands, act_evaders(env, apf))
        events = list(env.last_capture_events)
        capture_types.extend(str(event.get("capture_type", "unknown")) for event in events)
        collision_events = list(env.last_collision_events)
        collision = collision or bool(collision_events)
        boundary = boundary or any(event.get("type") == "boundary" for event in collision_events)
        if capture_step is None:
            prefix_collision = prefix_collision or bool(collision_events)
        captured = bool(env.evaders and all(evader.deactivated and not evader.collision for evader in env.evaders))
        if captured and capture_step is None:
            capture_step = step
        if env.post_capture_coverage_success and ce_step is None:
            ce_step = step
        after = snapshot(env, outcome.observations)
        tracker.observe(after, step, events)
        observations = outcome.observations
        if all(outcome.dones):
            break
    record = env.episode_record(task="coverage" if scene == "coverage" else "mix")
    collision = collision or bool(record.get("collision_event", False))
    boundary = boundary or bool(record.get("boundary_collision_event", False))
    captured = bool(record["captured"])
    ce = bool(env.post_capture_coverage_success)
    safe = bool(ce and (scene == "coverage" or captured) and not collision and record["all_pursuers_active"])
    dt = env.pursuers[0].dt * env.pursuers[0].N
    agreement["action_agreement"] = agreement["hits"] / max(agreement["rows"], 1)
    return {
        "scene": scene,
        "seed": int(seed),
        "contract": CONTRACT,
        "policy_mode": "no_is_pursuing_student_greedy",
        "captured": captured,
        "normal_capture": captured and "loose" in capture_types,
        "stationary_capture": captured and "stationary" in capture_types,
        "capture_types": capture_types,
        "collision": collision,
        "boundary": boundary,
        "capture_prefix_collision": prefix_collision,
        "ce_success": ce,
        "safe_complete": safe,
        "length": step,
        "episode_seconds": step * dt,
        "capture_seconds": capture_step * dt if capture_step is not None else None,
        "recovery_seconds": (ce_step - capture_step) * dt if ce_step is not None and capture_step is not None else None,
        "mission_seconds": ce_step * dt if safe and ce_step is not None else None,
        "detection_to_capture_seconds": None,
        "mission_events": tracker.finish(step),
        "policy_diagnostics": agreement,
        "no_is_pursuing_observation": True,
        "reward_role_metadata_retained": True,
    }


def rollout_student(
    student: CoCapIQN,
    artifact_root: Path,
    device: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    teacher = CoCapIQN.load(str(TEACHER), device=device).eval()
    records = []
    atomic_json(
        artifact_root / "rollout_progress.json",
        {"status": "running", "completed": 0, "total": 2 * int(episodes)},
    )
    for pair in range(int(episodes)):
        records.append(_run_no_bit_episode(student, teacher, "mixed", seed + pair, device))
        records.append(_run_no_bit_episode(student, teacher, "coverage", seed + pair + 100000, device))
        atomic_json(
            artifact_root / "rollout_progress.json",
            {"status": "running", "completed": len(records), "total": 2 * int(episodes)},
        )
    summary = summarize(records)
    agreement_rows = sum(int(row["policy_diagnostics"]["rows"]) for row in records)
    agreement_hits = sum(int(row["policy_diagnostics"]["hits"]) for row in records)
    report = {
        "schema": SCHEMA,
        "contract": CONTRACT,
        "teacher_sha256": TEACHER_SHA,
        "student_checkpoint_sha256": sha256_file(artifact_root / "student_no_is_pursuing.pt"),
        "episodes_per_scene": int(episodes),
        "seed_base": int(seed),
        "ppo_updates": 0,
        "scratch_rl": False,
        "observation_contract": {
            "self_feature_dim": 8,
            "pursuer_feature_dim": 6,
            "removed": ["self.is_pursuing", "friend.is_pursuing", "role_dependent_friend_ordering"],
            "reward_role_state": "retained in env/replay_metadata/evaluator only",
        },
        "summary": summary,
        "student_on_student_trajectory_teacher_agreement": {
            "rows": agreement_rows,
            "action_agreement": agreement_hits / max(agreement_rows, 1),
        },
        "records": records,
    }
    atomic_json(artifact_root / "rollout_report.json", report)
    atomic_json(
        artifact_root / "rollout_progress.json",
        {"status": "complete", "completed": len(records), "total": len(records)},
    )
    return report


def _physical_vectors(role_free: Mapping[str, np.ndarray]) -> np.ndarray:
    parts = [
        role_free["self"],
        role_free["pursuers"].reshape(len(role_free["self"]), -1),
        role_free["evaders"].reshape(len(role_free["self"]), -1),
        role_free["obstacles"].reshape(len(role_free["self"]), -1),
        role_free["masks"].astype(np.float32),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def aliasing_audit(
    teacher: CoCapIQN,
    arrays: Mapping[str, np.ndarray],
    role_free: Mapping[str, np.ndarray],
    artifact_root: Path,
    device: str,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    pursuing = np.asarray(arrays["pursuing"], dtype=bool)
    left = np.flatnonzero(~pursuing)
    right = np.flatnonzero(pursuing)
    if not len(left) or not len(right):
        return {"schema": SCHEMA, "status": "INSUFFICIENT_CROSS_ROLE_ROWS", "rows": int(len(arrays["episode"]))}
    vectors = _physical_vectors(role_free)
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        return {"schema": SCHEMA, "status": "SCIPY_UNAVAILABLE", "error": str(exc)}
    tree = cKDTree(vectors[right])
    distance, neighbor = tree.query(vectors[left], k=1)
    rmse = distance / np.sqrt(vectors.shape[1])
    paired_right = right[neighbor]
    actions = np.asarray(arrays["greedy_action"])
    mismatch = actions[left] != actions[paired_right]
    # RMSE is averaged over the 159-dimensional role-free physical vector.
    # 0.05 is retained as a strict reference; 0.10 is the calibrated
    # near-observation threshold used for the primary aliasing diagnosis.
    threshold = 0.10
    close = rmse <= threshold
    close_mismatch = close & mismatch
    categories: dict[str, int] = {}
    for source, target, is_close, is_mismatch in zip(left, paired_right, close, mismatch):
        if not is_close or not is_mismatch:
            continue
        direct_changed = not np.array_equal(arrays["direct_target_mask"][source], arrays["direct_target_mask"][target])
        neighbor_changed = not np.array_equal(arrays["neighbor_target_mask"][source], arrays["neighbor_target_mask"][target])
        if direct_changed:
            category = "target_recent_memory_or_local_target_evidence"
        elif neighbor_changed:
            category = "neighbor_target_evidence"
        else:
            category = "role_history_or_release_timer"
        categories[category] = categories.get(category, 0) + 1

    sample = rng.choice(len(arrays["episode"]), size=min(2048, len(arrays["episode"])), replace=False)
    teacher_obs = {
        "self": np.asarray(arrays["local_obs.self"])[sample].copy(),
        "pursuers": np.asarray(arrays["local_obs.pursuers"])[sample].copy(),
        "evaders": np.asarray(arrays["local_obs.evaders"])[sample].copy(),
        "obstacles": np.asarray(arrays["local_obs.obstacles"])[sample].copy(),
        "masks": np.asarray(arrays["local_obs.masks"])[sample].copy(),
        "types": np.asarray(arrays["local_obs.types"])[sample].copy(),
    }
    base_q = _predict_q(teacher, teacher_obs, np.arange(len(sample)), device)
    own_toggle = {key: value.copy() for key, value in teacher_obs.items()}
    own_toggle["self"][:, -1] = 1.0 - own_toggle["self"][:, -1]
    toggled_q = _predict_q(teacher, own_toggle, np.arange(len(sample)), device)
    friend_zero = {key: value.copy() for key, value in teacher_obs.items()}
    friend_zero["pursuers"][:, :, -1] = 0.0
    friend_q = _predict_q(teacher, friend_zero, np.arange(len(sample)), device)
    counterfactual = {
        "rows": int(len(sample)),
        "own_self_bit_toggle_action_flip_rate": float(np.mean(base_q.argmax(1) != toggled_q.argmax(1))),
        "own_self_bit_toggle_mean_abs_q_delta": float(np.abs(base_q - toggled_q).mean()),
        "friend_role_zeroing_action_flip_rate": float(np.mean(base_q.argmax(1) != friend_q.argmax(1))),
        "friend_role_zeroing_mean_abs_q_delta": float(np.abs(base_q - friend_q).mean()),
        "note": "counterfactual teacher sensitivity; no role bit is restored to the student",
    }
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "teacher_sha256": TEACHER_SHA,
        "rows": int(len(arrays["episode"])),
        "cross_role_rows": {"without_pursuing": int(len(left)), "with_pursuing": int(len(right))},
        "physical_vector": {
            "dimensions": int(vectors.shape[1]),
            "definition": "self/friend/evader/obstacle continuous features plus token masks; role columns removed; friend tokens physical-lexicographically sorted",
        },
        "nearest_cross_role_rmse": {
            "p10": float(np.percentile(rmse, 10)),
            "p25": float(np.percentile(rmse, 25)),
            "p50": float(np.percentile(rmse, 50)),
            "p90": float(np.percentile(rmse, 90)),
            "threshold": threshold,
            "close_pairs": int(close.sum()),
            "close_pair_fraction": float(close.mean()),
            "close_action_mismatches": int(close_mismatch.sum()),
            "close_action_mismatch_rate": float(close_mismatch.sum() / max(close.sum(), 1)),
            "threshold_curve": {
                str(value): {
                    "pairs": int((rmse <= value).sum()),
                    "pair_fraction": float((rmse <= value).mean()),
                    "action_mismatches": int(((rmse <= value) & mismatch).sum()),
                    "action_mismatch_rate": float(((rmse <= value) & mismatch).sum() / max((rmse <= value).sum(), 1)),
                }
                for value in (0.05, 0.10, 0.15, 0.20)
            },
        },
        "aliasing_categories": categories,
        "counterfactual_role_sensitivity": counterfactual,
        "interpretation": {
            "policy_shortcut": "own self role, friend role values, and role-dependent friend ordering are absent from the student view",
            "hidden_state_test": "close cross-role physical pairs with different teacher actions indicate information not represented by instantaneous role-free local geometry",
            "classification_rule": "direct target mask difference suggests local target evidence/history; neighbor mask difference suggests neighbor target evidence; otherwise role history/release timer is the leading proxy",
        },
    }
    atomic_json(artifact_root / "aliasing_audit.json", report)
    return report


def classify_b1(
    student_report: Mapping[str, Any],
    rollout_report: Mapping[str, Any] | None,
    alias_report: Mapping[str, Any],
) -> str:
    validation = student_report.get("all_rows", student_report["validation"])
    phase_scores = [float(value["action_agreement"]) for value in validation["phase"].values() if value["rows"]]
    overall = float(validation["action_agreement"])
    rollout_summary = (rollout_report or {}).get("summary", {})
    mixed = rollout_summary.get("mixed", {})
    coverage = rollout_summary.get("coverage", {})
    phase_ok = bool(phase_scores) and min(phase_scores) >= 0.85
    rollout_ok = bool(mixed) and bool(coverage) and min(
        float(mixed.get("safe_complete_rate", 0.0)),
        float(coverage.get("safe_complete_rate", 0.0)),
        float(mixed.get("captured_rate", 0.0)),
    ) >= 0.90
    close_alias = int(alias_report.get("nearest_cross_role_rmse", {}).get("close_action_mismatches", 0))
    if overall >= 0.90 and phase_ok and rollout_ok:
        return "B1_PASS_REDUNDANT_SHORTCUT"
    if overall >= 0.80 and phase_ok and (not rollout_report or mixed.get("safe_complete_rate", 0.0) >= 0.75):
        return "B1_PARTIAL_INFORMATION_GAP" if close_alias else "B1_PARTIAL_INFORMATION_GAP"
    return "B1_FAIL_HIDDEN_STATE_REQUIRED" if close_alias else "B1_FAIL_HIDDEN_STATE_REQUIRED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=SOURCE_DATASET)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-train-rows", type=int, default=32768)
    parser.add_argument("--student-checkpoint", type=Path, default=None)
    parser.add_argument("--rollout-episodes", type=int, default=20)
    parser.add_argument("--rollout-seed", type=int, default=2026091701)
    parser.add_argument("--seed", type=int, default=2026091701)
    parser.add_argument("--skip-rollout", action="store_true")
    args = parser.parse_args()
    if args.device != "cpu":
        raise ValueError("B1 default is CPU to avoid interfering with existing GPU processes")
    if args.epochs <= 0 or args.batch_size <= 0 or args.rollout_episodes <= 0 or args.max_train_rows <= 0:
        raise ValueError("epochs, batch size, rollout episodes, and max train rows must be positive")
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    _assert_teacher()
    arrays, source_meta = _load_dataset(args.dataset_root.resolve())
    role_free = _role_free_view(arrays)
    atomic_json(
        artifact_root / "dataset_manifest.json",
        {
            "schema": SCHEMA,
            "contract": CONTRACT,
            "teacher_sha256": TEACHER_SHA,
            "source": source_meta,
            "role_free_view": {
                "self_shape": list(role_free["self"].shape),
                "pursuer_shape": list(role_free["pursuers"].shape),
                "removed_columns": ["self[-1]=is_pursuing", "pursuers[:,-1]=friend.is_pursuing"],
                "friend_order": "physical lexicographic feature order; no role key",
            },
            "reward_unchanged": True,
            "topology_unchanged": True,
            "z_i": False,
            "scratch_rl": False,
        },
    )
    if args.student_checkpoint is None:
        student, student_report = train_student(
            arrays,
            role_free,
            source_meta,
            artifact_root,
            args.device,
            args.epochs,
            args.batch_size,
            args.seed,
            args.max_train_rows,
        )
    else:
        checkpoint = args.student_checkpoint.resolve()
        student = CoCapIQN.load(str(checkpoint), device=args.device).eval()
        if student.config.include_is_pursuing:
            raise ValueError("--student-checkpoint must be a no-is-pursuing student")
        report_path = artifact_root / "student_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"student report missing beside checkpoint: {report_path}")
        student_report = json.loads(report_path.read_text())
        expected_sha = str(student_report.get("student_checkpoint_sha256", ""))
        if expected_sha and sha256_file(checkpoint) != expected_sha:
            raise ValueError("student checkpoint does not match student_report.json")
    alias_report = aliasing_audit(
        CoCapIQN.load(str(TEACHER), device=args.device).eval(),
        arrays,
        role_free,
        artifact_root,
        args.device,
        args.seed,
    )
    rollout_report = None
    if not args.skip_rollout:
        rollout_report = rollout_student(
            student,
            artifact_root,
            args.device,
            args.rollout_episodes,
            args.rollout_seed,
        )
    else:
        existing_rollout = artifact_root / "rollout_report.json"
        if existing_rollout.exists():
            rollout_report = json.loads(existing_rollout.read_text())
    classification = classify_b1(student_report, rollout_report, alias_report)
    final = {
        "schema": SCHEMA,
        "status": "complete",
        "head_at_run": os.popen("git rev-parse HEAD").read().strip(),
        "teacher": {
            "path": str(TEACHER),
            "sha256": TEACHER_SHA,
            "config": dataclasses.asdict(CoCapIQN.load(str(TEACHER), device=args.device).config),
        },
        "student_report": str(artifact_root / "student_report.json"),
        "aliasing_report": str(artifact_root / "aliasing_audit.json"),
        "rollout_report": str(artifact_root / "rollout_report.json") if rollout_report else None,
        "b1_classification": classification,
        "next_stage_allowed": classification == "B1_PASS_REDUNDANT_SHORTCUT",
        "scope_guard": {
            "b2_local_voronoi": False,
            "b3_b4_z_i": False,
            "reward_change": False,
            "scratch_long_training": False,
            "actor_critic": False,
        },
    }
    atomic_json(artifact_root / "B1_FINAL_AUDIT.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
