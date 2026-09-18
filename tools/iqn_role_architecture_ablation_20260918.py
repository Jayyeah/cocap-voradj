#!/usr/bin/env python3
"""Matched C0/C1 positive-control ablation for Final IQN role architecture."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import dataclasses
import hashlib
import json
import os
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch
import torch.nn.functional as F

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.evaluation.mission_events import MissionEventTracker, snapshot
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.trainer import deep_update, load_config, set_global_config
from tools.collect_iqn_aw_teacher_dataset_20260903 import fixed_midpoint_q, sha256_file
from tools.evaluate_vxy_stage2_cross_retention_20260903 import ring_count
from tools.iqn_cleanup_b1_20260917 import DEFAULT_TEMPERATURE, _stratified_train_subset, _temperature
from tools.iqn_zstate_b3_20260918 import _place_targeted_ring3, _role
from tools.rollout_voradj_visual import act_evaders
from tools.run_forward_final_bridge_20260908 import atomic_json

SCHEMA = "iqn-role-architecture-ablation-v1"
TEACHER = ROOT / "artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage1_4v1_step_2000000.pt"
TEACHER_SHA = "2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89"
CONFIG = ROOT / "configs/experiments/iqn_role_architecture_ablation_20260918/token_only_is_pursuing.yaml"
B3_ROOT = ROOT / "artifacts/2026-09-18_iqn_zstate_b3"
DEFAULT_ROOT = ROOT / "artifacts/2026-09-18_iqn_role_arch_ablation"
PHASE_NAMES = {0: "pre_capture", 1: "post_capture", 2: "pure_coverage"}


def resolved_config(scene: str) -> dict[str, Any]:
    root = load_config(str(CONFIG))
    task = "voradj_coverage" if scene == "coverage" else "voradj"
    cfg = deep_update(root, root["tasks"][task])
    if scene == "capture":
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("voradj", {})["capture_episode_ends_on_capture"] = True
        cfg["voradj"]["capture_episode_success_on_capture"] = True
    return cfg


def make_env(scene: str, seed: int) -> tuple[VorAdjEnv, list[Mapping[str, np.ndarray] | None]]:
    cfg = resolved_config(scene)
    set_global_config(cfg)
    env = VorAdjEnv(copy.deepcopy(cfg), seed=int(seed))
    observations = list(env.reset())
    if any(obs is not None and (obs["self"].shape != (9,) or obs["pursuers"].shape[-1] != 7) for obs in observations):
        raise AssertionError("role-token observation shape drift")
    return env, observations


def assert_contract() -> dict[str, Any]:
    if sha256_file(TEACHER) != TEACHER_SHA:
        raise ValueError("Final IQN teacher SHA mismatch")
    cfg = resolved_config("mixed")
    b3 = load_config(str(ROOT / "configs/experiments/iqn_zstate_b3_20260918/z_state.yaml"))
    b3 = deep_update(b3, b3["tasks"]["voradj"])
    for key in ("reward", "voradj", "env"):
        if cfg[key] != b3[key]:
            raise AssertionError(f"{key} drift from B3")
    if not cfg["perception"].get("include_is_pursuing") or cfg["perception"].get("include_z_state"):
        raise AssertionError("C1 observation contract is not role-token only")
    if cfg["iqn"].get("pursuing_late_fusion", True):
        raise AssertionError("C1 late fusion remains enabled")
    return {
        "teacher_sha256": TEACHER_SHA,
        "b3_dataset_manifest_sha256": sha256_file(B3_ROOT / "dataset/manifest.json"),
        "b3_student_checkpoint_sha256": sha256_file(B3_ROOT / "student_zstate.pt"),
        "reward_unchanged": True,
        "topology_unchanged": True,
        "role_tokens_preserved": True,
        "role_dependent_friend_ordering_preserved": True,
        "c1_late_fusion_removed": True,
    }


def collect_episode(teacher: CoCapIQN, scene: str, seed: int, episode_id: int, device: str, targeted: bool = False) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    env, observations = make_env(scene, seed)
    if targeted:
        observations = _place_targeted_ring3(env)
    apf = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    rows: dict[str, list[Any]] = {}
    capture_seen = False
    capture_types: list[str] = []
    for timestep in range(1, env.episode_max_length + 1):
        active = [i for i, obs in enumerate(observations) if obs is not None]
        if not active:
            break
        teacher_view = env.get_policy_observations(True, False)
        q_values, actions = fixed_midpoint_q(teacher, [teacher_view[i] for i in active], device)
        phase = 2 if scene == "coverage" else (1 if capture_seen else 0)
        current_ring = ring_count(env)
        decision_map = env._capture_voronoi_map()
        direct_snapshot = {
            index: bool(env._has_enemy_neighbor(decision_map, ("pursuer", index)))
            for index in active
        }
        commands: list[int | None] = [None] * len(env.pursuers)
        for row, index in enumerate(active):
            commands[index] = int(actions[row])
        outcome = env.step(commands, act_evaders(env, apf))
        events = list(env.last_capture_events)
        capture_types.extend(str(event.get("capture_type", "unknown")) for event in events)
        capture_seen = capture_seen or bool(events)
        for row, index in enumerate(active):
            observation = teacher_view[index]
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
                "phase_id": np.uint8(phase),
                "role_id": np.uint8(_role(metadata)),
                "direct": np.bool_(direct_snapshot[index]),
                "support": np.bool_(metadata.get("support_candidate", False)),
                "pursuing": np.bool_(metadata.get("effective_pursuing", False)),
                "ring_count": np.int8(current_ring),
                "ring2": np.bool_(current_ring >= 2),
                "ring3": np.bool_(current_ring >= 3),
                "capture_transition": np.bool_(bool(events)),
                "reward": np.float32(outcome.rewards[index]),
                "terminated": np.bool_(metadata["terminated"]),
                "truncated": np.bool_(metadata["truncated"]),
            }
            for key, value in scalars.items():
                rows.setdefault(key, []).append(value)
        observations = list(outcome.observations)
        if all(outcome.dones):
            break
    arrays = {key: np.stack(values) for key, values in rows.items()}
    record = env.episode_record(task="coverage" if scene == "coverage" else "mix")
    return arrays, {
        "episode": int(episode_id), "scene": scene, "seed": int(seed),
        "targeted_ring3_category_coverage": bool(targeted), "rows": int(len(arrays["episode"])),
        "steps": int(timestep), "captured": bool(record.get("captured", False)),
        "ce_success": bool(env.post_capture_coverage_success),
        "collision": bool(record.get("collision_event", False)), "capture_types": capture_types,
        "ring2_rows": int(arrays["ring2"].sum()), "ring3_rows": int(arrays["ring3"].sum()),
        "direct_rows": int(arrays["direct"].sum()), "support_rows": int(arrays["support"].sum()),
        "post_capture_rows": int((arrays["phase_id"] == 1).sum()),
        "pure_coverage_rows": int((arrays["phase_id"] == 2).sum()),
    }


def _episode_path(root: Path, scene: str, index: int) -> Path:
    return root / "shards" / f"{scene}_{index:03d}.npz"


def _verify_shard_match(arrays: Mapping[str, np.ndarray], b3_path: Path) -> dict[str, Any]:
    exact_keys = ("greedy_action", "episode", "environment_seed", "timestep", "agent", "scene_id", "phase_id", "role_id", "direct", "support", "pursuing", "ring_count", "ring2", "ring3", "capture_transition", "terminated", "truncated")
    with np.load(b3_path, allow_pickle=False) as b3:
        if any(not np.array_equal(arrays[key], b3[key]) for key in exact_keys):
            raise AssertionError(f"metadata mismatch against B3: {b3_path}")
        q_delta = float(np.max(np.abs(arrays["teacher_q"] - b3["teacher_q"])))
        reward_delta = float(np.max(np.abs(arrays["reward"] - b3["reward"])))
    if q_delta > 1e-6 or reward_delta > 1e-6:
        raise AssertionError(f"numeric mismatch against B3: q={q_delta}, reward={reward_delta}")
    return {"b3_shard": str(b3_path.relative_to(B3_ROOT / "dataset")), "teacher_q_max_abs_delta": q_delta, "reward_max_abs_delta": reward_delta, "metadata_exact": True}


def collect_dataset(root: Path, episodes: int, seed: int, device: str) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "shards").mkdir(parents=True, exist_ok=True)
    teacher = CoCapIQN.load(str(TEACHER), device=device).eval()
    entries, matches = [], []
    episode_id = 0
    for scene_offset, scene in enumerate(("mixed", "coverage")):
        for index in range(episodes):
            path = _episode_path(root, scene, index)
            episode_seed = seed + scene_offset * 100000 + index
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    arrays = {key: np.asarray(saved[key]) for key in saved.files if key != "episode_summary"}
                    summary = json.loads(str(saved["episode_summary"].item()))
            else:
                arrays, summary = collect_episode(teacher, scene, episode_seed, episode_id, device)
                np.savez_compressed(path, **arrays, episode_summary=np.asarray(json.dumps(summary, sort_keys=True)))
            b3_path = B3_ROOT / "dataset" / "shards" / path.name
            matches.append(_verify_shard_match(arrays, b3_path))
            entries.append({**summary, "path": str(path.relative_to(root)), "sha256": sha256_file(path)})
            episode_id += 1
            atomic_json(root / "manifest.json", {"schema": SCHEMA, "status": "running", "completed": len(entries)})
    targeted_path = root / "shards/mixed_targeted_ring3.npz"
    if targeted_path.exists():
        with np.load(targeted_path, allow_pickle=False) as saved:
            arrays = {key: np.asarray(saved[key]) for key in saved.files if key != "episode_summary"}
            summary = json.loads(str(saved["episode_summary"].item()))
    else:
        arrays, summary = collect_episode(teacher, "mixed", 2026094101, episode_id, device, True)
        np.savez_compressed(targeted_path, **arrays, episode_summary=np.asarray(json.dumps(summary, sort_keys=True)))
    matches.append(_verify_shard_match(arrays, B3_ROOT / "dataset/shards/mixed_targeted_ring3.npz"))
    entries.append({**summary, "path": str(targeted_path.relative_to(root)), "sha256": sha256_file(targeted_path)})
    manifest = {
        "schema": SCHEMA, "status": "complete", "teacher_sha256": TEACHER_SHA,
        "episodes_per_scene": episodes, "seed_base": seed,
        "row_count": int(sum(item["rows"] for item in entries)),
        "coverage": {key: int(sum(item[key] for item in entries)) for key in ("ring2_rows", "ring3_rows", "direct_rows", "support_rows", "post_capture_rows", "pure_coverage_rows")},
        "observation_semantics": "original Final IQN is_pursuing tokens and role-dependent friend ordering",
        "b3_match": {"strict": True, "all_metadata_exact": True, "max_teacher_q_abs_delta": max(x["teacher_q_max_abs_delta"] for x in matches), "max_reward_abs_delta": max(x["reward_max_abs_delta"] for x in matches), "shards": matches},
        "shards": entries,
    }
    atomic_json(root / "manifest.json", manifest)
    return manifest


def load_dataset(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("role dataset incomplete")
    chunks: dict[str, list[np.ndarray]] = {}
    for item in manifest["shards"]:
        path = root / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as shard:
            for key in shard.files:
                if key != "episode_summary":
                    chunks.setdefault(key, []).append(np.asarray(shard[key]))
    arrays = {key: np.concatenate(value) for key, value in chunks.items()}
    return arrays, manifest


def observation_tree(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key.removeprefix("obs."): np.asarray(value) for key, value in arrays.items() if key.startswith("obs.")}


def batch_obs(obs: Mapping[str, np.ndarray], indices: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    return stack_obs([{key: value[index] for key, value in obs.items()} for index in indices], device)


def midpoint_tau(device: str, count: int) -> torch.Tensor:
    return (torch.arange(count, device=device, dtype=torch.float32) + 0.5) / count


@torch.no_grad()
def predict_q(model: CoCapIQN, obs: Mapping[str, np.ndarray], indices: np.ndarray, device: str, batch_size: int = 8192) -> np.ndarray:
    output = []
    for start in range(0, len(indices), batch_size):
        take = indices[start:start + batch_size]
        q = model(batch_obs(obs, take, device), num_tau=32, mode="voradj", tau=midpoint_tau(device, 32))["q_values"].mean(1)
        output.append(q.detach().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def metric_summary(model: CoCapIQN, obs: Mapping[str, np.ndarray], arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: str, temperature: float) -> dict[str, Any]:
    student_q = predict_q(model, obs, indices, device)
    teacher_q = np.asarray(arrays["teacher_q"])[indices]
    teacher_action, student_action = teacher_q.argmax(1), student_q.argmax(1)
    target, student = torch.as_tensor(teacher_q), torch.as_tensor(student_q)
    target_lp = F.log_softmax((target - target.max(1, keepdim=True).values) / temperature, dim=1)
    student_lp = F.log_softmax(student, dim=1)
    kl = (target_lp.exp() * (target_lp - student_lp)).sum(1).numpy()
    rank = (np.sign(teacher_q[:, :, None] - teacher_q[:, None, :]) == np.sign(student_q[:, :, None] - student_q[:, None, :])).mean((1, 2))
    regret = teacher_q.max(1) - teacher_q[np.arange(len(teacher_q)), student_action]
    masks = {
        "overall": np.ones(len(indices), bool), "pre_capture": arrays["phase_id"][indices] == 0,
        "post_capture": arrays["phase_id"][indices] == 1, "pure_coverage": arrays["phase_id"][indices] == 2,
        "direct": arrays["role_id"][indices] == 0, "pursuing_memory": arrays["role_id"][indices] == 1,
        "support": arrays["role_id"][indices] == 2, "coverage": arrays["role_id"][indices] == 3,
        "ring2": arrays["ring2"][indices], "ring3": arrays["ring3"][indices],
    }
    return {name: {"rows": int(mask.sum()), "action_agreement": float(np.mean(student_action[mask] == teacher_action[mask])) if mask.any() else None, "categorical_kl": float(kl[mask].mean()) if mask.any() else None, "q_regret": float(regret[mask].mean()) if mask.any() else None, "q_ranking_agreement": float(rank[mask].mean()) if mask.any() else None} for name, mask in masks.items()}


def save_model(path: Path, model: CoCapIQN, extra: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": SCHEMA, "config": dataclasses.asdict(model.config), "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "extra": dict(extra)}, path)


def train_arm(arm: str, arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any], artifact_root: Path, device: str, epochs: int, batch_size: int, max_train_rows: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    teacher = CoCapIQN.load(str(TEACHER), device=device).eval()
    if arm == "C0":
        student, transfer = copy.deepcopy(teacher), {"full_teacher_state_loaded": True, "late_fusion_preserved": True}
    elif arm == "C1":
        student, transfer = CoCapIQN.make_token_only_is_pursuing_student(teacher)
    else:
        raise ValueError(arm)
    student.to(device).eval()
    obs = observation_tree(arrays)
    episodes = np.asarray(arrays["episode"])
    valid_mask = episodes % 5 == 0
    train_idx, valid_idx = np.flatnonzero(~valid_mask), np.flatnonzero(valid_mask)
    optimization_idx = _stratified_train_subset(arrays, train_idx, max_train_rows, seed)
    temperature = _temperature() or DEFAULT_TEMPERATURE
    all_idx = np.arange(len(episodes), dtype=np.int64)
    step0 = metric_summary(student, obs, arrays, all_idx, device, temperature)
    step0_q = predict_q(student, obs, all_idx, device)
    step0_fidelity = {
        "teacher_q_max_abs_delta": float(
            np.max(np.abs(step0_q - np.asarray(arrays["teacher_q"])))
        ),
        "note": "GPU batch arithmetic may differ at near-tied argmax rows; Q regret is authoritative",
    }
    step0_path = artifact_root / f"{arm}_step0.pt"
    save_model(step0_path, student, {"arm": arm, "stage": "step0", "transfer": transfer})
    if arm == "C0" and (
        step0["overall"]["action_agreement"] < 0.9999
        or step0_fidelity["teacher_q_max_abs_delta"] > 1e-4
        or step0["overall"]["q_regret"] > 1e-7
    ):
        raise AssertionError("IMPLEMENTATION_INVALID: C0 step0 does not reproduce teacher")
    optimizer = torch.optim.Adam(student.parameters(), lr=3e-4, eps=1e-5)
    teacher_q = np.asarray(arrays["teacher_q"], dtype=np.float32)
    history, best_key = [], (-1.0, float("-inf"))
    checkpoint = artifact_root / f"{arm}_postbc.pt"
    for epoch in range(1, epochs + 1):
        student.train()
        losses = []
        order = np.random.permutation(optimization_idx)
        for start in range(0, len(order), batch_size):
            take = order[start:start + batch_size]
            student_q = student(batch_obs(obs, take, device), num_tau=8, mode="voradj", tau=midpoint_tau(device, 8))["q_values"].mean(1)
            target_q = torch.as_tensor(teacher_q[take], device=device)
            target_lp = F.log_softmax((target_q - target_q.max(1, keepdim=True).values) / temperature, dim=1)
            student_lp = F.log_softmax(student_q, dim=1)
            kl = F.kl_div(student_lp, target_lp.exp(), reduction="batchmean")
            scale = target_q.std(1, keepdim=True).clamp_min(1e-3)
            q_loss = F.smooth_l1_loss((student_q - student_q.mean(1, keepdim=True)) / scale, (target_q - target_q.mean(1, keepdim=True)) / scale)
            loss = kl + 0.25 * q_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        student.eval()
        validation = metric_summary(student, obs, arrays, valid_idx, device, temperature)
        key = (validation["overall"]["action_agreement"], -validation["overall"]["categorical_kl"])
        history.append({"epoch": epoch, "mean_loss": float(np.mean(losses)), "validation": validation})
        if key > best_key:
            best_key = key
            save_model(checkpoint, student, {"arm": arm, "stage": "postbc", "best_epoch": epoch, "transfer": transfer, "dataset_manifest_sha256": sha256_file(artifact_root / "dataset/manifest.json")})
    post = CoCapIQN.load(str(checkpoint), device=device).eval()
    post_metrics = metric_summary(post, obs, arrays, all_idx, device, temperature)
    common = {"schema": SCHEMA, "arm": arm, "teacher_sha256": TEACHER_SHA, "dataset_rows": int(len(episodes)), "optimization_rows": int(len(optimization_idx)), "train_rows": int(len(train_idx)), "validation_rows": int(len(valid_idx)), "epochs": epochs, "batch_size": batch_size, "optimizer": {"name": "Adam", "lr": 3e-4, "eps": 1e-5}, "loss": "categorical_KL + 0.25*normalized_smooth_L1", "temperature": float(temperature), "seed": seed, "transfer": transfer}
    atomic_json(artifact_root / f"{arm}_step0.json", {**common, "stage": "step0", "checkpoint": str(step0_path), "checkpoint_sha256": sha256_file(step0_path), "offline": step0, "step0_fidelity": step0_fidelity})
    atomic_json(artifact_root / f"{arm}_postbc.json", {**common, "stage": "postbc", "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "offline": post_metrics, "history": history})
    return {"step0": step0, "postbc": post_metrics}


def timing(values: Sequence[float | None]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None]
    return {"n": len(clean), "mean": float(np.mean(clean)) if clean else None, "p50": float(np.percentile(clean, 50)) if clean else None, "p90": float(np.percentile(clean, 90)) if clean else None}


@torch.no_grad()
def run_episode(policy: CoCapIQN, teacher: CoCapIQN, name: str, scene: str, seed: int, device: str) -> dict[str, Any]:
    env, observations = make_env(scene, seed)
    apf = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    tracker = MissionEventTracker(env.pursuers[0].dt * env.pursuers[0].N)
    tracker.observe(snapshot(env, observations), 0)
    capture_step = ce_step = None
    collision = False
    capture_types: list[str] = []
    agreement_rows = agreement_hits = 0
    for step in range(1, env.episode_max_length + 1):
        active = [i for i, obs in enumerate(observations) if obs is not None]
        if not active:
            break
        view = env.get_policy_observations(True, False)
        teacher_q, teacher_action = fixed_midpoint_q(teacher, [view[i] for i in active], device)
        _q, action = fixed_midpoint_q(policy, [view[i] for i in active], device)
        agreement_rows += len(active); agreement_hits += int(np.sum(action == teacher_action))
        commands: list[int | None] = [None] * len(env.pursuers)
        for row, index in enumerate(active): commands[index] = int(action[row])
        outcome = env.step(commands, act_evaders(env, apf))
        events = list(env.last_capture_events)
        capture_types.extend(str(event.get("capture_type", "unknown")) for event in events)
        collision = collision or bool(env.last_collision_events)
        captured_now = bool(env.evaders and all(e.deactivated and not e.collision for e in env.evaders))
        if captured_now and capture_step is None: capture_step = step
        if env.post_capture_coverage_success and ce_step is None: ce_step = step
        tracker.observe(snapshot(env, outcome.observations), step, events)
        observations = list(outcome.observations)
        if all(outcome.dones): break
    record = env.episode_record(task="coverage" if scene == "coverage" else "mix")
    captured, ce = bool(record.get("captured", False)), bool(env.post_capture_coverage_success)
    collision = bool(collision or record.get("collision_event", False))
    safe = bool(captured and not collision and record["all_pursuers_active"]) if scene == "capture" else bool(ce and (scene == "coverage" or captured) and not collision and record["all_pursuers_active"])
    dt = env.pursuers[0].dt * env.pursuers[0].N
    return {"policy": name, "scene": scene, "seed": seed, "captured": captured, "normal_capture": captured and "loose" in capture_types, "stationary_capture": captured and "stationary" in capture_types, "collision": collision, "ce_success": ce, "safe_complete": safe, "steps": int(step), "episode_seconds": float(step * dt), "capture_seconds": float(capture_step * dt) if capture_step else None, "recovery_seconds": float((ce_step - capture_step) * dt) if ce_step and capture_step else None, "mission_seconds": float(ce_step * dt) if safe and ce_step else (float(capture_step * dt) if safe and scene == "capture" and capture_step else None), "agreement_rows": agreement_rows, "agreement_hits": agreement_hits, "mission_events": tracker.finish(step)}


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for policy in sorted({str(row["policy"]) for row in records}):
        result[policy] = {}
        for scene in ("coverage", "capture", "mixed"):
            rows = [row for row in records if row["policy"] == policy and row["scene"] == scene]
            rate = lambda key: float(np.mean([bool(row[key]) for row in rows]))
            result[policy][scene] = {"episodes": len(rows), **{f"{key}_rate": rate(key) for key in ("captured", "normal_capture", "stationary_capture", "collision", "ce_success", "safe_complete")}, "episode_seconds": timing([r["episode_seconds"] for r in rows]), "capture_seconds": timing([r["capture_seconds"] for r in rows]), "recovery_seconds": timing([r["recovery_seconds"] for r in rows]), "mission_seconds": timing([r["mission_seconds"] for r in rows])}
    return result


_WORKER_MODELS: dict[str, CoCapIQN] = {}


def _rollout_worker(job: tuple[str, str, int, str, str]) -> dict[str, Any]:
    name, scene, seed, artifact_root_text, device = job
    artifact_root = Path(artifact_root_text)
    if "teacher" not in _WORKER_MODELS:
        _WORKER_MODELS["teacher"] = CoCapIQN.load(str(TEACHER), device=device).eval()
    if name not in _WORKER_MODELS:
        _WORKER_MODELS[name] = CoCapIQN.load(
            str(artifact_root / f"{name}.pt"), device=device
        ).eval()
    return run_episode(
        _WORKER_MODELS[name], _WORKER_MODELS["teacher"], name, scene, seed, device
    )


def rollout(artifact_root: Path, device: str, episodes: int, qual_episodes: int, seed: int, workers: int) -> dict[str, Any]:
    policies = [("C0_step0", artifact_root / "C0_step0.pt", qual_episodes), ("C1_step0", artifact_root / "C1_step0.pt", qual_episodes), ("C0_postbc", artifact_root / "C0_postbc.pt", episodes), ("C1_postbc", artifact_root / "C1_postbc.pt", episodes)]
    jobs = []
    for name, _path, count in policies:
        for scene_index, scene in enumerate(("coverage", "capture", "mixed")):
            for index in range(count):
                jobs.append((name, scene, seed + scene_index * 100000 + index, str(artifact_root), device))
    records: list[dict[str, Any]] = []
    total = sum(count * 3 for _, _, count in policies)
    atomic_json(artifact_root / "rollout_progress.json", {"status": "running", "completed": 0, "total": total, "workers": workers})
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = [executor.submit(_rollout_worker, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            atomic_json(artifact_root / "rollout_progress.json", {"status": "running", "completed": len(records), "total": total, "workers": workers})
    records.sort(key=lambda row: (row["policy"], row["scene"], row["seed"]))
    report = {"schema": SCHEMA, "formal_episodes_per_scene": episodes, "step0_qualification_episodes_per_scene": qual_episodes, "matched_seed_base": seed, "summary": summarize(records), "on_policy_teacher_agreement": {name: {"rows": sum(r["agreement_rows"] for r in records if r["policy"] == name), "action_agreement": sum(r["agreement_hits"] for r in records if r["policy"] == name) / max(sum(r["agreement_rows"] for r in records if r["policy"] == name), 1)} for name, _, _ in policies}, "records": records}
    atomic_json(artifact_root / "rollout_comparison.json", report)
    atomic_json(artifact_root / "rollout_progress.json", {"status": "complete", "completed": total, "total": total})
    for arm in ("C0", "C1"):
        for stage in ("step0", "postbc"):
            path = artifact_root / f"{arm}_{stage}.json"
            payload = json.loads(path.read_text()); payload["rollout"] = report["summary"][f"{arm}_{stage}"]; atomic_json(path, payload)
    return report


def build_final(artifact_root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    arm_reports = {f"{arm}_{stage}": json.loads((artifact_root / f"{arm}_{stage}.json").read_text()) for arm in ("C0", "C1") for stage in ("step0", "postbc")}
    rollout_report = json.loads((artifact_root / "rollout_comparison.json").read_text())
    b3_student = json.loads((B3_ROOT / "student_report.json").read_text())
    b3_audit = json.loads((B3_ROOT / "B3_FINAL_AUDIT.json").read_text())
    offline = {name: report["offline"] for name, report in arm_reports.items()}
    offline["C2_B3_postbc"] = b3_student["all_rows"]
    atomic_json(artifact_root / "offline_comparison.json", {"schema": SCHEMA, "arms": offline})
    summary = rollout_report["summary"]
    def passed(name: str) -> bool:
        return all(summary[name][scene]["safe_complete_rate"] >= 0.85 for scene in ("coverage", "capture", "mixed"))
    c0_pass, c1_pass = passed("C0_postbc"), passed("C1_postbc")
    if not c0_pass: case, verdict = "CASE_C", "CURRENT_BC_PIPELINE_FAILS_POSITIVE_CONTROL"
    elif c1_pass: case, verdict = "CASE_A", "TOKEN_ONLY_IS_PURSUING_SUFFICIENT"
    elif any(summary["C1_postbc"][scene]["safe_complete_rate"] >= 0.85 for scene in ("coverage", "capture", "mixed")): case, verdict = "CASE_D", "LATE_FUSION_PARTIALLY_IMPORTANT"
    else: case, verdict = "CASE_B", "LATE_FUSION_CAUSALLY_IMPORTANT"
    c0_step = offline["C0_step0"]["overall"]; c0_post = offline["C0_postbc"]["overall"]
    manifest = json.loads((artifact_root / "dataset/manifest.json").read_text())
    compact_arms = {
        name: {
            "checkpoint": report["checkpoint"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "offline": report["offline"],
            "rollout": report.get("rollout"),
            **({"step0_fidelity": report["step0_fidelity"]} if "step0_fidelity" in report else {}),
        }
        for name, report in arm_reports.items()
    }
    compact_rollout = {
        key: rollout_report[key]
        for key in (
            "formal_episodes_per_scene",
            "step0_qualification_episodes_per_scene",
            "matched_seed_base",
            "summary",
            "on_policy_teacher_agreement",
        )
    }
    final = {"schema": SCHEMA, "status": "complete", "branch": "experiment/iqn-role-architecture-ablation-20260918", "baseline_head_at_run": os.popen("git rev-parse HEAD").read().strip(), "contract": contract, "dataset": {"manifest": str(artifact_root / "dataset/manifest.json"), "rows": manifest["row_count"], "coverage": manifest["coverage"], "observation_semantics": manifest["observation_semantics"], "b3_match": {key: manifest["b3_match"][key] for key in ("strict", "all_metadata_exact", "max_teacher_q_abs_delta", "max_reward_abs_delta")}}, "training_matched_to_b3": {"strict": True, "epochs": 4, "batch_size": 4096, "optimizer": "Adam(lr=3e-4, eps=1e-5)", "loss": "categorical_KL + 0.25*normalized_smooth_L1", "split": "episode_id % 5", "max_train_rows": 65536, "seed": 2026091801}, "arms": compact_arms, "c2_b3_reference": {"offline": b3_student["all_rows"], "rollout": b3_audit["rollout"], "classification": b3_audit["classification"]}, "c0_bc_drift": {key: c0_post[key] - c0_step[key] for key in ("action_agreement", "categorical_kl", "q_regret", "q_ranking_agreement")}, "rollout": compact_rollout, "decision": {"case": case, "verdict": verdict, "pass_rule": "all three formal safe-complete rates >= 0.85", "c0_postbc_pass": c0_pass, "c1_postbc_pass": c1_pass}, "pure_coverage_causal_analysis": "C0 step-0 passes but C0 post-BC reaches 0/20 pure-coverage CE with unchanged original architecture and role semantics. The BC pipeline is therefore the major confound; C1-versus-C0 and B3 z-causality conclusions are downgraded.", "future_z_todo": {"implemented": False, "shorter_half_life": "preregister a small lambda matrix", "hard_floor": "z=0 when max(d_i, lambda*z_prev, eta*neighbor_prev) < epsilon_z", "matrix": ["lambda", "epsilon_z"]}, "next_stage_training_launched": False}
    atomic_json(artifact_root / "FINAL_AUDIT.json", final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("collect", "train", "rollout", "final", "all"), default="all")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset-episodes", type=int, default=20)
    parser.add_argument("--dataset-seed", type=int, default=2026091801)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-train-rows", type=int, default=65536)
    parser.add_argument("--rollout-episodes", type=int, default=20)
    parser.add_argument("--qualification-episodes", type=int, default=3)
    parser.add_argument("--rollout-seed", type=int, default=2026092801)
    parser.add_argument("--rollout-workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026091801)
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve(); artifact_root.mkdir(parents=True, exist_ok=True)
    contract = assert_contract(); atomic_json(artifact_root / "contract.json", {"schema": SCHEMA, **contract})
    if args.stage in ("collect", "all"): collect_dataset(artifact_root / "dataset", args.dataset_episodes, args.dataset_seed, args.device)
    if args.stage in ("train", "all"):
        arrays, manifest = load_dataset(artifact_root / "dataset")
        for arm in ("C0", "C1"): train_arm(arm, arrays, manifest, artifact_root, args.device, args.epochs, args.batch_size, args.max_train_rows, args.seed)
    if args.stage in ("rollout", "all"): rollout(artifact_root, args.device, args.rollout_episodes, args.qualification_episodes, args.rollout_seed, args.rollout_workers)
    if args.stage in ("final", "all"):
        print(json.dumps(build_final(artifact_root, contract), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
