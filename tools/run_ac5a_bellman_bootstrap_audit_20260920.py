#!/usr/bin/env python3
"""AC-5A: one-step Bellman bootstrap target audit.

The actor and both critics are frozen.  AC-4B anchors and full empirical
returns are read-only inputs.  This runner reconstructs the exact anchor
snapshots from the AC-4B seed/step manifest, executes only each forced first
step, evaluates the frozen BC policy on the resulting next state, and writes
small prediction matrices plus aggregate diagnostics.  No continuation
trajectory or optimizer state is written.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path("/home/yjq/rl/CoCap1/cocap-voradj-critic-audit")
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.evaluation.mission_events import snapshot as mission_snapshot
from tools.collect_ac2b_canonical_bc_bank_20260920 import ACTION_GRID
from tools.distill_forward_final_actor_20260908 import load_actor
from tools.run_ac3a_critic_sampler_audit_20260920 import make_config, sha256_file
from tools.run_ac4a_counterfactual_aw9_pilot_20260920 import (
    ACTION_SCALE,
    ACTOR_LOADED,
    ACTOR_REQUESTED,
    ACTOR_SHA256,
    GAMMA,
    MAX_AGENTS,
    MAX_NEIGHBORS,
    atomic_json,
    canonical_actions,
    canonical_path_from_state,
    compare_observations,
    current_phase_class,
    json_safe,
    pad_observations,
    restore_env_state,
    save_env_state,
    step_once,
    tensor_state_sha256,
)
from tools.run_ac4b_formal_counterfactual_aw9_20260920 import (
    aggregate_ranking,
    ranking_row,
)
from tools.run_critic_identifiability_audit_20260917 import forward, make_model, to_tensor
from tools.run_forward_final_bridge_20260908 import act_evaders, make_env


OUT = ROOT / "artifacts/2026-09-20_ac5a"
AC4B_DIR = ROOT / "artifacts/2026-09-20_ac4b"
AC4B_MANIFEST = AC4B_DIR / "anchor_manifest.json"
AC4B_RETURNS = AC4B_DIR / "counterfactual_returns.npz"
AC4B_SUMMARY = AC4B_DIR / "summary.json"
AC3B_SUMMARY = ROOT / "artifacts/2026-09-20_ac3b/summary.json"
GAMMA = 0.99
SNAPSHOT_STATES = 5
SNAPSHOT_CONTINUATION_STEPS = 5
MARGIN_TOLERANCE = 1e-3
CRITIC_PATHS = {
    "LQ": ROOT / "artifacts/2026-09-20_ac3a/checkpoints/lq_natural_seed2026091711.pt",
    "NQ": ROOT / "artifacts/2026-09-20_ac3a/checkpoints/nq_natural_seed2026091711.pt",
}
CRITIC_SHA = {
    "LQ": "e8b52a32040937c2a4060e359546d1a427aee6c955dcaec501827038fac27552",
    "NQ": "53a194b17d9e963c5a63fce137d44725df19dc59420b624a5657c87d290a5052",
}
CRITIC_SEED = {"LQ": 2026091711, "NQ": 2026091711}
PHASES = ("pursuing", "pre_capture_cover", "early_recovery", "recovery_pure")


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def describe(values: list[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "mean": None, "median": None, "sd": None, "iqr": None, "q1": None, "q3": None}
    q1, q3 = np.quantile(array, [0.25, 0.75])
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sd": float(np.std(array)),
        "iqr": float(q3 - q1),
        "q1": float(q1),
        "q3": float(q3),
    }


def regression_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(truth)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    pred = pred[valid]
    truth = truth[valid]
    if pred.size == 0:
        return {"n": 0, "rmse": None, "mae": None, "bias": None, "pearson": None, "spearman": None}
    error = pred - truth
    pearson = None
    spearman = None
    if pred.size > 1 and np.std(pred) > 0 and np.std(truth) > 0:
        pearson = float(np.corrcoef(pred, truth)[0, 1])
        pred_rank = np.argsort(np.argsort(pred, kind="mergesort"), kind="mergesort").astype(float)
        truth_rank = np.argsort(np.argsort(truth, kind="mergesort"), kind="mergesort").astype(float)
        spearman = float(np.corrcoef(pred_rank, truth_rank)[0, 1])
    return {
        "n": int(pred.size),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "pearson": pearson,
        "spearman": spearman,
    }


def load_ac4b_inputs() -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads(AC4B_MANIFEST.read_text())
    if manifest.get("status") != "complete" or len(manifest.get("anchors", [])) != 64:
        raise RuntimeError("AC4B_ANCHOR_MANIFEST_FAIL")
    with np.load(AC4B_RETURNS, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    required = {"returns", "lq_predictions", "nq_predictions", "bc_actions", "semantic_class"}
    if not required.issubset(arrays) or arrays["returns"].shape != (64, 9):
        raise RuntimeError("AC4B_RETURN_MATRIX_FAIL")
    summary = json.loads(AC4B_SUMMARY.read_text())
    if summary.get("branches", {}).get("total") != 576:
        raise RuntimeError("AC4B_SUMMARY_BRANCH_COUNT_FAIL")
    return manifest["anchors"], arrays, summary


def replay_ac4b_anchors(actor: torch.nn.Module, manifest_anchors: list[dict[str, Any]], output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    for row in manifest_anchors:
        key = (str(row["scene"]), int(row["seed"]), int(row["transition_step"]), int(row["focal_agent_id"]))
        targets.setdefault(key, []).append(row)
    result: dict[int, dict[str, Any]] = {}
    max_obs_error = 0.0
    action_mismatches = []
    semantic_mismatches = []
    episodes = []
    for scene in ("mixed", "pure_coverage"):
        seeds = sorted({int(row["seed"]) for row in manifest_anchors if row["scene"] == scene})
        for seed in seeds:
            env_scene = "coverage" if scene == "pure_coverage" else "mixed"
            env, observations = make_env(env_scene, seed)
            apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
            capture_step = None
            found_this_episode = 0
            episode_targets = [row for row in manifest_anchors if row["scene"] == scene and int(row["seed"]) == seed]
            target_keys = {
                (scene, seed, int(row["transition_step"]), int(row["focal_agent_id"])) for row in episode_targets
            }
            for _ in range(env.episode_max_length):
                if not any(value is not None for value in observations):
                    break
                actions, policy_meta, _ = canonical_actions(actor, observations)
                transition_step = int(env.episode_step + 1)
                active = set(int(value) for value in policy_meta["active"].tolist())
                for focal in sorted(active):
                    key = (scene, seed, transition_step, focal)
                    if key not in target_keys:
                        continue
                    for row in targets[key]:
                        phase, semantic_class, _ = current_phase_class(scene, observations, focal, capture_step, transition_step)
                        if semantic_class != row["semantic_class"] or phase != row["phase"]:
                            semantic_mismatches.append({"anchor_id": row["anchor_id"], "expected": [row["phase"], row["semantic_class"]], "actual": [phase, semantic_class]})
                        max_local_error = 0.0
                        reference = row["local_observation"]
                        for name, value in reference.items():
                            current = np.asarray(observations[focal][name], dtype=np.float64)
                            expected = np.asarray(value, dtype=np.float64)
                            max_local_error = max(max_local_error, float(np.max(np.abs(current - expected), initial=0.0)))
                        max_obs_error = max(max_obs_error, max_local_error)
                        expected_actions = np.asarray(row["joint_bc_action_indices"], dtype=np.int64)
                        if not np.array_equal(actions, expected_actions):
                            action_mismatches.append(int(row["anchor_id"]))
                        result[int(row["anchor_id"])] = {
                            "manifest": row,
                            "scene": scene,
                            "focal": focal,
                            "bundle": save_env_state(env, observations, capture_step),
                            "joint_action_indices": actions.copy(),
                            "original_bc_action": int(actions[focal]),
                        }
                        found_this_episode += 1
                if found_this_episode == len(episode_targets):
                    break
                step = step_once(env, observations, actions, apf_agents, scene, capture_step)
                observations = step["observations"]
                capture_step = step["capture_step"]
                if bool(step["done"].all()):
                    break
            episodes.append({"scene": scene, "seed": seed, "anchors_found": found_this_episode, "anchors_expected": len(episode_targets)})
    if len(result) != len(manifest_anchors):
        missing = sorted(set(range(len(manifest_anchors))) - set(result))
        raise RuntimeError(f"AC4B_ANCHOR_REPLAY_MISSING: {missing}")
    if action_mismatches or semantic_mismatches or max_obs_error > 1e-6:
        raise RuntimeError(json.dumps({"AC4B_ANCHOR_REPLAY_IDENTITY_FAIL": {"action_mismatches": action_mismatches, "semantic_mismatches": semantic_mismatches, "max_obs_error": max_obs_error}}, ensure_ascii=False))
    anchors = [result[index] for index in range(len(manifest_anchors))]
    atomic_json(output / "progress.json", {"status": "ac4b_anchor_replay_complete", "anchors": len(anchors), "max_observation_error": max_obs_error})
    return anchors, {"episodes": episodes, "max_observation_error": max_obs_error, "action_mismatches": 0, "semantic_mismatches": 0}


def snapshot_gate(anchors: list[dict[str, Any]], actor: torch.nn.Module) -> dict[str, Any]:
    rows = []
    max_obs_error = 0.0
    max_reward_error = 0.0
    for index, anchor in enumerate(anchors[:SNAPSHOT_STATES]):
        first = canonical_path_from_state(anchor["bundle"], actor, anchor["scene"], anchor["focal"], forced_action=anchor["original_bc_action"], max_steps=1 + SNAPSHOT_CONTINUATION_STEPS)
        second = canonical_path_from_state(anchor["bundle"], actor, anchor["scene"], anchor["focal"], forced_action=anchor["original_bc_action"], max_steps=1 + SNAPSHOT_CONTINUATION_STEPS)
        if len(first["trace"]) != len(second["trace"]) or len(first["trace"]) < 1 + SNAPSHOT_CONTINUATION_STEPS:
            raise RuntimeError("SNAPSHOT_RESTORE_FAIL")
        event_match = True
        obs_error = 0.0
        reward_error = 0.0
        for left, right in zip(first["trace"], second["trace"]):
            obs_error = max(obs_error, compare_observations(left["observations"], right["observations"]))
            reward_error = max(reward_error, float(np.max(np.abs(left["reward"] - right["reward"]), initial=0.0)))
            event_match = event_match and left["event_signature"] == right["event_signature"]
            event_match = event_match and left["phase"] == right["phase"]
            event_match = event_match and np.array_equal(left["done"], right["done"])
            event_match = event_match and np.array_equal(left["truncated"], right["truncated"])
        max_obs_error = max(max_obs_error, obs_error)
        max_reward_error = max(max_reward_error, reward_error)
        if obs_error > 1e-7 or reward_error > 1e-9 or not event_match:
            raise RuntimeError("SNAPSHOT_RESTORE_FAIL")
        rows.append({"anchor_id": anchor["manifest"]["anchor_id"], "steps_compared": len(first["trace"]), "max_observation_abs_error": obs_error, "max_reward_abs_error": reward_error, "event_done_match": True})
    return {"status": "PASS", "states_checked": len(rows), "continuation_steps_after_first": SNAPSHOT_CONTINUATION_STEPS, "max_observation_abs_error": max_obs_error, "max_reward_abs_error": max_reward_error, "max_error": max(max_obs_error, max_reward_error), "states": rows}


def one_step_branch(anchor: Mapping[str, Any], actor: torch.nn.Module, forced_action: int) -> dict[str, Any]:
    env, observations = restore_env_state(anchor["bundle"])
    apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    actions = np.asarray(anchor["joint_action_indices"], dtype=np.int64).copy()
    actions[int(anchor["focal"])] = int(forced_action)
    step = step_once(env, observations, actions, apf_agents, anchor["scene"], anchor["bundle"].get("capture_step"))
    next_observations = step["observations"]
    next_actions, _, _ = canonical_actions(actor, next_observations)
    next_state = mission_snapshot(env, next_observations)
    local_full, _ = pad_observations(next_observations)
    focal = int(anchor["focal"])
    neighbor_ids = np.full((MAX_NEIGHBORS,), -1, dtype=np.int16)
    neighbor_mask = np.zeros((MAX_NEIGHBORS,), dtype=bool)
    ids = sorted(next_state["friends"].get(focal, set()))[:MAX_NEIGHBORS]
    neighbor_ids[: len(ids)] = ids
    neighbor_mask[: len(ids)] = True
    joint_aw = np.zeros((MAX_AGENTS, 2), dtype=np.float32)
    joint_aw[: len(next_actions)] = ACTION_GRID[np.clip(next_actions, 0, 8)]
    done = np.asarray(step["done"], dtype=bool)
    truncated = np.asarray(step["truncated"], dtype=bool)
    terminal = bool(done.all() or truncated.all())
    return {
        "immediate_reward": float(step["outcome"].rewards[focal]),
        "terminal": terminal,
        "done": done,
        "truncated": truncated,
        "next_focal_active": next_observations[focal] is not None,
        "next_action_index": int(next_actions[focal]),
        "next_local_full": local_full,
        "next_neighbor_ids": neighbor_ids,
        "next_neighbor_mask": neighbor_mask,
        "next_joint_aw": joint_aw,
        "event_signature": step["event_signature"],
    }


def build_next_batch(records: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    focal = np.asarray([int(row["focal"]) for row in records], dtype=np.int64)
    obs = {key: np.stack([np.asarray(row["next_local_full"][key])[int(row["focal"])] for row in records]) for key in records[0]["next_local_full"]}
    neighbor_obs = {}
    neighbor_actions = []
    masks = []
    for key in records[0]["next_local_full"]:
        neighbor_obs[key] = np.stack([
            np.asarray(row["next_local_full"][key])[np.maximum(np.asarray(row["next_neighbor_ids"]), 0)] for row in records
        ])
    for row in records:
        ids = np.maximum(np.asarray(row["next_neighbor_ids"], dtype=np.int64), 0)
        actions = np.asarray(row["next_joint_aw"])[ids].copy()
        actions[~np.asarray(row["next_neighbor_mask"], dtype=bool)] = 0.0
        neighbor_actions.append(actions)
        masks.append(np.asarray(row["next_neighbor_mask"], dtype=bool))
    action_aw = np.stack([ACTION_GRID[int(row["next_action_index"])] for row in records])
    return {
        "obs": {key: to_tensor(value, device) for key, value in obs.items()},
        "action": to_tensor(action_aw / ACTION_SCALE, device),
        "neighbor_obs": {key: to_tensor(value, device) for key, value in neighbor_obs.items()},
        "neighbor_action": to_tensor(np.stack(neighbor_actions) / ACTION_SCALE, device),
        "neighbor_mask": to_tensor(np.stack(masks), device),
        "agent": torch.as_tensor(focal, device=device, dtype=torch.long),
    }


def load_lq_nq(device: torch.device) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    config = make_config()
    models = {}
    provenance = {}
    for kind in ("LQ", "NQ"):
        path = CRITIC_PATHS[kind]
        actual = sha256_file(path)
        if actual != CRITIC_SHA[kind]:
            raise RuntimeError(f"CRITIC_CHECKPOINT_IDENTITY_FAIL_{kind}")
        payload = torch.load(path, map_location=device, weights_only=False)
        model = make_model(kind, config, device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        model.requires_grad_(False)
        before = tensor_state_sha256(model.state_dict())
        models[kind] = model
        provenance[kind] = {"path": str(path), "sha256": actual, "seed": CRITIC_SEED[kind], "state_sha256_before": before}
    return models, provenance


def predict_tails(records: list[dict[str, Any]], models: Mapping[str, torch.nn.Module], device: torch.device) -> dict[str, np.ndarray]:
    valid = [(index, row) for index, row in enumerate(records) if not row["terminal"] and row["next_focal_active"]]
    if any(not row["next_focal_active"] and not row["terminal"] for row in records):
        raise RuntimeError("NEXT_STATE_FOCAL_MISSING")
    output = {kind: np.full(len(records), np.nan, dtype=np.float64) for kind in models}
    if not valid:
        return output
    batch = build_next_batch([row for _, row in valid], device)
    with torch.no_grad():
        for kind, model in models.items():
            values = forward(kind, model, batch).detach().cpu().numpy().astype(np.float64)
            values *= float(NORMALIZATION["train_std"])
            values += float(NORMALIZATION["train_mean"])
            for (index, _), value in zip(valid, values):
                output[kind][index] = float(value)
    return output


def ranking_report(predictions: np.ndarray, empirical: np.ndarray, anchors: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [ranking_row(predictions[index], empirical[index], int(anchor["original_bc_action"])) for index, anchor in enumerate(anchors) for _ in [0]]
    return {
        "overall": aggregate_ranking(rows),
        "per_anchor": rows,
        "by_phase": {
            phase: aggregate_ranking([row for row, anchor in zip(rows, anchors) if anchor["manifest"]["semantic_class"] == phase])
            for phase in PHASES
        },
    }


def tail_partition_metrics(prediction: np.ndarray, empirical_tail: np.ndarray, records: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> dict[str, Any]:
    valid = np.isfinite(prediction) & np.isfinite(empirical_tail)
    all_bc = []
    all_alt = []
    for index, record in enumerate(records):
        anchor = anchors[int(record["anchor_id"])]
        (all_bc if int(record["action"]) == int(anchor["original_bc_action"]) else all_alt).append(index)
    masks = {
        "all": valid,
        "bc_action": valid & np.asarray([index in all_bc for index in range(len(records))]),
        "alternatives": valid & np.asarray([index in all_alt for index in range(len(records))]),
    }
    return {name: regression_metrics(prediction, empirical_tail, mask) for name, mask in masks.items()}


def phase_tail_metrics(prediction: np.ndarray, empirical_tail: np.ndarray, records: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for phase in PHASES:
        phase_mask = np.asarray([anchors[int(record["anchor_id"])]["manifest"]["semantic_class"] == phase for record in records])
        bc_mask = np.asarray([int(record["action"]) == int(anchors[int(record["anchor_id"])]["original_bc_action"]) for record in records])
        valid = np.isfinite(prediction) & np.isfinite(empirical_tail)
        result[phase] = {
            "all": regression_metrics(prediction, empirical_tail, valid & phase_mask),
            "bc_action": regression_metrics(prediction, empirical_tail, valid & phase_mask & bc_mask),
            "alternatives": regression_metrics(prediction, empirical_tail, valid & phase_mask & ~bc_mask),
        }
    return result


def delta_metrics(direct: Mapping[str, Any], bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "top1": ("top1_agreement", "mean"),
        "top3": ("top3_overlap", "mean"),
        "spearman": ("spearman", "mean"),
        "regret": ("regret", "mean"),
        "sign": ("sign_accuracy_vs_bc", "mean"),
        "margin_sign": ("margin_filtered_sign_accuracy", "mean"),
    }
    return {name: float(bootstrap[key][field] - direct[key][field]) for name, (key, field) in keys.items()}


def ranking_classification(direct: Mapping[str, Any], bootstrap: Mapping[str, Any], tail: Mapping[str, Any]) -> list[str]:
    labels = []
    primary_deltas = [
        bootstrap["top1_agreement"]["mean"] - direct["top1_agreement"]["mean"],
        bootstrap["top3_overlap"]["mean"] - direct["top3_overlap"]["mean"],
        bootstrap["spearman"]["mean"] - direct["spearman"]["mean"],
        bootstrap["sign_accuracy_vs_bc"]["mean"] - direct["sign_accuracy_vs_bc"]["mean"],
    ]
    improved = sum(value > 0 for value in primary_deltas)
    degraded = sum(value < 0 for value in primary_deltas)
    if improved >= 3:
        labels.append("BOOTSTRAP_SIGNAL_IMPROVED")
    elif degraded >= 3:
        labels.append("BOOTSTRAP_SIGNAL_DEGRADED")
    else:
        labels.append("BOOTSTRAP_SIGNAL_PRESERVED")
    bc_rmse = tail["bc_action"]["rmse"]
    alt_rmse = tail["alternatives"]["rmse"]
    if bc_rmse is not None and alt_rmse is not None and alt_rmse > bc_rmse:
        labels.extend(["ON_POLICY_TAIL_OK", "COUNTERFACTUAL_NEXT_STATE_EXTRAPOLATION"])
    else:
        labels.append("ON_POLICY_TAIL_WEAK")
        labels.append("NO_CLEAR_EXTRAPOLATION_GAP")
    return labels


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    before_disk = shutil.disk_usage(ROOT)

    actor_sha = sha256_file(ACTOR_LOADED)
    if actor_sha != ACTOR_SHA256:
        raise RuntimeError("BC_CHECKPOINT_IDENTITY_FAIL")
    actor, _ = load_actor(ACTOR_LOADED, "cpu")
    actor.eval()
    actor.requires_grad_(False)
    actor_hash_before = tensor_state_sha256(actor.state_dict())
    if not np.array_equal(np.asarray(actor.action_grid.cpu(), dtype=np.float32), ACTION_GRID):
        raise RuntimeError("ACTION_MAPPING_FAIL")
    manifest_anchors, ac4b_arrays, ac4b_summary = load_ac4b_inputs()
    global NORMALIZATION
    ac3b = json.loads(AC3B_SUMMARY.read_text())
    NORMALIZATION = ac3b["contract"]["normalization"]
    anchors, replay = replay_ac4b_anchors(actor, manifest_anchors, output)
    gate = snapshot_gate(anchors, actor)

    records = []
    immediate = np.zeros((64, 9), dtype=np.float64)
    terminal = np.zeros((64, 9), dtype=bool)
    next_action = np.zeros((64, 9), dtype=np.int64)
    for anchor_index, anchor in enumerate(anchors):
        for action in range(9):
            outcome = one_step_branch(anchor, actor, action)
            immediate[anchor_index, action] = outcome["immediate_reward"]
            terminal[anchor_index, action] = outcome["terminal"]
            next_action[anchor_index, action] = outcome["next_action_index"]
            records.append({"anchor_id": anchor_index, "action": action, "focal": int(anchor["focal"]), **outcome})
        atomic_json(output / "progress.json", {"status": "first_step_branching", "anchors_completed": anchor_index + 1, "anchors_total": len(anchors)})

    device = torch.device(args.device)
    models, provenance = load_lq_nq(device)
    direct = {"LQ": np.asarray(ac4b_arrays["lq_predictions"], dtype=np.float64), "NQ": np.asarray(ac4b_arrays["nq_predictions"], dtype=np.float64)}
    empirical = np.asarray(ac4b_arrays["returns"], dtype=np.float64)
    tail_flat = predict_tails(records, models, device)
    tails = {kind: values.reshape(64, 9) for kind, values in tail_flat.items()}
    bootstrap = {kind: immediate + GAMMA * np.where(terminal, 0.0, tails[kind]) for kind in models}
    empirical_tail = np.where(terminal, np.nan, (empirical - immediate) / GAMMA)

    direct_rank = {kind: ranking_report(direct[kind], empirical, anchors) for kind in models}
    bootstrap_rank = {kind: ranking_report(bootstrap[kind], empirical, anchors) for kind in models}
    direct_vs_bootstrap = {kind: delta_metrics(direct_rank[kind]["overall"], bootstrap_rank[kind]["overall"]) for kind in models}
    direct_errors = {kind: regression_metrics(direct[kind].reshape(-1), empirical.reshape(-1)) for kind in models}
    bootstrap_errors = {kind: regression_metrics(bootstrap[kind].reshape(-1), empirical.reshape(-1)) for kind in models}
    tail_errors = {kind: tail_partition_metrics(tail_flat[kind], empirical_tail.reshape(-1), records, anchors) for kind in models}
    phase_tail = {kind: phase_tail_metrics(tail_flat[kind], empirical_tail.reshape(-1), records, anchors) for kind in models}
    phase_difference = {}
    for kind in models:
        phase_difference[kind] = {}
        for phase in PHASES:
            phase_mask = np.asarray([anchors[int(record["anchor_id"])]["manifest"]["semantic_class"] == phase for record in records])
            direct_phase = ranking_report(direct[kind], empirical, anchors)["by_phase"][phase]
            bootstrap_phase = ranking_report(bootstrap[kind], empirical, anchors)["by_phase"][phase]
            phase_difference[kind][phase] = {
                "direct_spearman": direct_phase["spearman"]["mean"],
                "bootstrap_spearman": bootstrap_phase["spearman"]["mean"],
                "direct_sign": direct_phase["sign_accuracy_vs_bc"]["mean"],
                "bootstrap_sign": bootstrap_phase["sign_accuracy_vs_bc"]["mean"],
                "tail_rmse_all": phase_tail[kind][phase]["all"]["rmse"],
                "tail_rmse_bc_action": phase_tail[kind][phase]["bc_action"]["rmse"],
                "tail_rmse_alternatives": phase_tail[kind][phase]["alternatives"]["rmse"],
            }
    contribution = {}
    for phase in PHASES:
        mask = np.asarray([anchors[index // 9]["manifest"]["semantic_class"] == phase for index in range(576)])
        phase_immediate = immediate.reshape(-1)[mask]
        phase_tail_emp = np.where(np.isfinite(empirical_tail.reshape(-1)[mask]), empirical.reshape(-1)[mask] - immediate.reshape(-1)[mask], 0.0)
        contribution[phase] = {
            "n": int(np.sum(mask)),
            "var_immediate_reward": float(np.var(phase_immediate)),
            "var_gamma_empirical_tail": float(np.var(phase_tail_emp)),
            "mean_abs_immediate_reward": float(np.mean(np.abs(phase_immediate))),
            "mean_abs_gamma_empirical_tail": float(np.mean(np.abs(phase_tail_emp))),
        }
    terminal_rows = [
        {"anchor_id": int(index // 9), "action": int(index % 9), "immediate_reward": float(immediate.reshape(-1)[index]), "done": True}
        for index in np.flatnonzero(terminal.reshape(-1))
    ]
    ac4b_direct_match = {}
    for kind in models:
        old = ac4b_summary["ranking"][kind]["overall"]
        new = direct_rank[kind]["overall"]
        ac4b_direct_match[kind] = {
            "max_abs_metric_difference": max(
                abs(float(old[key]["mean"]) - float(new[key]["mean"]))
                for key in ("top3_overlap", "spearman", "regret", "sign_accuracy_vs_bc", "margin_filtered_sign_accuracy")
            ),
            "top1_count_matches": int(old["top1_count"]) == int(new["top1_count"]),
        }
    classifications = {kind: ranking_classification(direct_rank[kind]["overall"], bootstrap_rank[kind]["overall"], tail_errors[kind]) for kind in models}
    actor_hash_after = tensor_state_sha256(actor.state_dict())
    critic_hash_after = {kind: tensor_state_sha256(model.state_dict()) for kind, model in models.items()}
    arrays = {
        "empirical_returns": empirical,
        "direct_lq": direct["LQ"],
        "direct_nq": direct["NQ"],
        "bootstrap_lq": bootstrap["LQ"],
        "bootstrap_nq": bootstrap["NQ"],
        "immediate_rewards": immediate,
        "terminal": terminal,
        "next_action_indices": next_action,
        "empirical_tail": empirical_tail,
        "lq_tail_q": tails["LQ"],
        "nq_tail_q": tails["NQ"],
        "bc_actions": np.asarray(ac4b_arrays["bc_actions"], dtype=np.int64),
    }
    np.savez_compressed(output / "branch_bootstrap_predictions.npz", **arrays)
    after_disk = shutil.disk_usage(ROOT)
    summary = {
        "schema": "ac5a-one-step-bellman-bootstrap-target-audit-v1",
        "status": "complete",
        "git_head_at_run": git_head(),
        "contract": {
            "source": "AC4B formal 64-anchor x 9-action counterfactual matrix",
            "anchors": 64,
            "branches": 576,
            "gamma": GAMMA,
            "environment_contract": ac4b_summary["contract"]["environment_contract"],
            "actor_checkpoint_requested": str(ACTOR_REQUESTED),
            "actor_checkpoint_loaded": str(ACTOR_LOADED),
            "actor_sha256": actor_sha,
            "target": "r_t + gamma*(1-done)*Q(next-state, frozen BC argmax)",
            "empirical_reference": str(AC4B_RETURNS),
            "margin_tolerance": MARGIN_TOLERANCE,
        },
        "ac4b_anchor_replay": replay,
        "snapshot_gate": gate,
        "branch_partition": {"total": 576, "bc_action": 64, "alternatives": 512, "terminal_first_step": len(terminal_rows), "terminal_rows": terminal_rows},
        "next_state_policy": {"focal_and_neighbors_recomputed_from_counterfactual_next_state": True, "anchor_t1_actions_reused": False},
        "critic_provenance": provenance,
        "direct_q": direct_rank,
        "bootstrap_q": bootstrap_rank,
        "bootstrap_minus_direct": direct_vs_bootstrap,
        "direct_return_errors": direct_errors,
        "bootstrap_return_errors": bootstrap_errors,
        "tail_errors": tail_errors,
        "tail_errors_by_phase": phase_tail,
        "phase_conditioned": phase_difference,
        "immediate_vs_tail_contribution": contribution,
        "distribution_shift": {
            kind: {
                "alternative_over_bc_tail_rmse_ratio": (tail_errors[kind]["alternatives"]["rmse"] / tail_errors[kind]["bc_action"]["rmse"]) if tail_errors[kind]["bc_action"]["rmse"] else None,
                "absolute_rmse_difference": tail_errors[kind]["alternatives"]["rmse"] - tail_errors[kind]["bc_action"]["rmse"],
            }
            for kind in models
        },
        "ac4b_direct_reproduction": ac4b_direct_match,
        "classification": classifications,
        "forbidden_executed": {"actor_optimizer_steps": 0, "lq_optimizer_steps": 0, "nq_optimizer_steps": 0, "critic_training": False, "bootstrap_training": False, "iterative_fqe": False, "target_network_training": False, "PPO": False, "GAE": False, "TD3": False, "MADDPG": False, "SAC": False},
        "state_hashes": {"actor_before": actor_hash_before, "actor_after": actor_hash_after, "actor_bit_exact": actor_hash_before == actor_hash_after, "critics_before": {kind: provenance[kind]["state_sha256_before"] for kind in models}, "critics_after": critic_hash_after, "critics_bit_exact": all(provenance[kind]["state_sha256_before"] == critic_hash_after[kind] for kind in models)},
        "artifacts": {"source_anchor_manifest": str(AC4B_MANIFEST), "source_returns": str(AC4B_RETURNS), "branch_bootstrap_predictions": str(output / "branch_bootstrap_predictions.npz"), "branch_bootstrap_predictions_sha256": sha256_file(output / "branch_bootstrap_predictions.npz")},
        "storage": {"disk_free_bytes_before": int(before_disk.free), "disk_free_bytes_after": int(after_disk.free), "output_bytes": 0},
    }
    atomic_json(output / "summary.json", summary)
    summary["storage"]["output_bytes"] = int(sum(path.stat().st_size for path in output.rglob("*") if path.is_file()))
    summary["storage"]["disk_free_bytes_after"] = int(shutil.disk_usage(ROOT).free)
    atomic_json(output / "summary.json", summary)
    summary["storage"]["output_bytes"] = int(sum(path.stat().st_size for path in output.rglob("*") if path.is_file()))
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "progress.json", {"status": "complete", "branches": 576})
    for model in models.values():
        del model
    del actor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps({"status": "complete", "snapshot_gate": gate["status"], "branches": 576, "terminal_first_step": len(terminal_rows), "output_bytes": summary["storage"]["output_bytes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
