#!/usr/bin/env python3
"""AC-5B1: paired MC-control versus iterative Bellman LQ audit.

This runner deliberately fits only LQ.  It uses the canonical BC argmax rows
from the existing AC-2B bank, starts every arm from the same AC-3A LQ state,
and compares a fixed realized-MC target with four frozen-target Bellman
blocks.  The Actor is loaded only for deterministic next-action evaluation;
it is never optimized and no trajectory bank is written.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path("/home/yjq/rl/CoCap1/cocap-voradj-critic-audit")
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from cocap_voradj.models.critic_identifiability import parameter_count
from cocap_voradj.training.small_step_ac import tensor_tree
from tools.collect_ac2b_canonical_bc_bank_20260920 import ACTION_GRID, EVENT_TO_ID, PHASE_TO_ID
from tools.distill_forward_final_actor_20260908 import load_actor
from tools.run_ac3a_critic_sampler_audit_20260920 import AC3Rows, make_config, sha256_file
from tools.run_ac4b_formal_counterfactual_aw9_20260920 import aggregate_ranking, ranking_row
from tools.run_critic_identifiability_audit_20260917 import (
    ACTION_SCALE,
    calibration,
    forward,
    make_model,
    predict,
    to_tensor,
)


OUT = ROOT / "artifacts/2026-09-20_ac5b1"
BANK_PATH = ROOT / "artifacts/2026-09-20_ac2b/canonical_bc_critic_bank_v2.npz"
BANK_MANIFEST = ROOT / "artifacts/2026-09-20_ac2b/bank_manifest.json"
BANK_SHA256 = "07e08d17ae6283d785e0d2e00307f330b13a88dcd0f8ce83f92f0d451ad7808a"
INITIAL_LQ_PATH = ROOT / "artifacts/2026-09-20_ac3a/checkpoints/lq_natural_seed2026091711.pt"
INITIAL_LQ_SHA256 = "e8b52a32040937c2a4060e359546d1a427aee6c955dcaec501827038fac27552"
EXPECTED_INITIAL_STATE_SHA256 = "825b11e6753a9eb541dd265b087b74e4ed786fbc0430d04ab6fbd9802900eada"
ACTOR_REQUESTED = ROOT / "artifacts/2026-09-08_forward_final/c2_distillation/actor_epoch_030.pt"
ACTOR_LOADED = ROOT / "artifacts/2026-09-17_critic_identifiability_audit/frozen_policy/actor_epoch_030.pt"
ACTOR_SHA256 = "7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd"
AC4B_ANCHOR_MANIFEST = ROOT / "artifacts/2026-09-20_ac4b/anchor_manifest.json"
AC4B_RETURNS = ROOT / "artifacts/2026-09-20_ac4b/counterfactual_returns.npz"
AC4B_SUMMARY = ROOT / "artifacts/2026-09-20_ac4b/summary.json"
GAMMA = 0.99
TRAIN_MEAN = 15.788457870483398
TRAIN_STD = 52.9477653503418
SEEDS = (2026091711, 2026091712)
CONDITIONS = ("mc_control", "iterative_bellman")
BLOCKS = 4
UPDATES_PER_BLOCK = 300
BATCH_SIZE = 256
CHECKPOINT_POINTS = (0, 300, 600, 900, 1200)
PHASE_NAMES = ("pure_coverage", "pre_capture", "post_capture")
EVENT_METRIC_NAMES = ("early_recovery", "late_recovery", "normal_capture", "stationary_capture", "collision")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def load_bank() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    actual = sha256_file(BANK_PATH)
    if actual != BANK_SHA256:
        raise RuntimeError(f"AC5B1_BANK_IDENTITY_FAIL: {actual} != {BANK_SHA256}")
    manifest = json.loads(BANK_MANIFEST.read_text())
    if manifest["status"] != "formal_bank_complete" or manifest["bank"]["sha256"] != actual:
        raise RuntimeError("AC5B1_BANK_MANIFEST_MISMATCH")
    with np.load(BANK_PATH, allow_pickle=False) as source:
        bank = {name: source[name] for name in source.files}
    required = {
        "next_local_self", "next_local_pursuers", "next_local_evaders", "next_local_obstacles",
        "next_local_masks", "next_local_types", "done", "reward", "mc_return",
        "policy_mode_id", "split_id", "episode_id", "active_mask",
    }
    missing = sorted(required.difference(bank))
    if missing:
        raise RuntimeError(f"AC5B1_BANK_SCHEMA_MISSING: {missing}")
    return bank, manifest


def make_argmax_rows(bank: Mapping[str, np.ndarray], device: torch.device) -> AC3Rows:
    rows = AC3Rows(bank, device)
    keep = bank["policy_mode_id"][rows.transition] == 0
    rows.transition = rows.transition[keep]
    rows.agent = rows.agent[keep]
    rows.split = bank["split_id"][rows.transition]
    rows.target = bank["mc_return"][rows.transition, rows.agent].astype(np.float32)
    rows.primary = bank["primary_event"][rows.transition]
    rows.semantic_class = bank["replay_semantic_class"][rows.transition, rows.agent].astype(np.int8)
    return rows


def episode_contract(bank: Mapping[str, np.ndarray]) -> dict[str, Any]:
    records = []
    for episode_id in np.unique(bank["episode_id"]):
        take = bank["episode_id"] == episode_id
        records.append({
            "episode_id": int(episode_id),
            "seed": int(bank["episode_seed"][take][0]),
            "scene": "mixed" if int(bank["scene_id"][take][0]) == 0 else "pure_coverage",
            "policy_mode": "bc_argmax" if int(bank["policy_mode_id"][take][0]) == 0 else "bc_sample",
            "split": ("train", "validation", "test")[int(bank["split_id"][take][0])],
        })
    argmax = [row for row in records if row["policy_mode"] == "bc_argmax"]
    expected = {
        "train": {"episodes": 32, "mixed": 24, "pure_coverage": 8},
        "validation": {"episodes": 4, "mixed": 3, "pure_coverage": 1},
        "test": {"episodes": 4, "mixed": 3, "pure_coverage": 1},
    }
    actual = {}
    for split, wanted in expected.items():
        chosen = [row for row in argmax if row["split"] == split]
        actual[split] = {
            "episodes": len(chosen),
            "mixed": sum(row["scene"] == "mixed" for row in chosen),
            "pure_coverage": sum(row["scene"] == "pure_coverage" for row in chosen),
            "seeds": [row["seed"] for row in chosen],
            "episode_ids": [row["episode_id"] for row in chosen],
        }
        if actual[split]["episodes"] != wanted["episodes"] or actual[split]["mixed"] != wanted["mixed"] or actual[split]["pure_coverage"] != wanted["pure_coverage"]:
            raise RuntimeError(f"AC5B1_ARGMAX_EPISODE_SPLIT_FAIL: {split}: {actual[split]}")
    return {"all_episode_records": records, "argmax": actual}


def load_initial_state(config: Mapping[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], str]:
    actual = sha256_file(INITIAL_LQ_PATH)
    if actual != INITIAL_LQ_SHA256:
        raise RuntimeError(f"AC5B1_INITIAL_LQ_IDENTITY_FAIL: {actual} != {INITIAL_LQ_SHA256}")
    payload = torch.load(INITIAL_LQ_PATH, map_location="cpu", weights_only=False)
    if payload.get("schema") != "cocap-ac3a-critic-sampler-v1" or payload.get("kind") != "LQ":
        raise RuntimeError("AC5B1_INITIAL_LQ_PAYLOAD_FAIL")
    norm = payload.get("normalization", {})
    if abs(float(norm.get("train_mean", 0.0)) - TRAIN_MEAN) > 1e-6 or abs(float(norm.get("train_std", 0.0)) - TRAIN_STD) > 1e-6:
        raise RuntimeError("AC5B1_NORMALIZATION_CONTRACT_FAIL")
    model = make_model("LQ", config, device)
    model.load_state_dict(payload["state_dict"], strict=True)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    state_hash = tensor_state_sha256(state)
    if state_hash != EXPECTED_INITIAL_STATE_SHA256:
        raise RuntimeError(f"AC5B1_INITIAL_STATE_HASH_FAIL: {state_hash}")
    del model
    return state, state_hash


def local_obs(bank: Mapping[str, np.ndarray], prefix: str, transition: np.ndarray, agent: np.ndarray) -> dict[str, np.ndarray]:
    return {
        key: bank[prefix + key][transition, agent]
        for key in ("self", "pursuers", "evaders", "obstacles", "masks", "types")
    }


@torch.no_grad()
def actor_argmax(actor: torch.nn.Module, observations: Mapping[str, np.ndarray], device: torch.device, batch_size: int = 4096) -> np.ndarray:
    total = len(next(iter(observations.values())))
    result = np.empty(total, dtype=np.int64)
    actor.eval()
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        batch = tensor_tree({key: value[start:stop] for key, value in observations.items()}, device)
        logits = actor.distribution(batch).logits
        if not torch.isfinite(logits).all():
            raise RuntimeError("AC5B1_ACTOR_NONFINITE_LOGITS")
        result[start:stop] = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
    return result


def actor_contract(bank: Mapping[str, np.ndarray], rows: AC3Rows, actor: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    before = tensor_state_sha256(actor.state_dict())
    if sha256_file(ACTOR_LOADED) != ACTOR_SHA256:
        raise RuntimeError("AC5B1_ACTOR_IDENTITY_FAIL")
    grid = actor.action_grid.detach().cpu().numpy().astype(np.float32)
    if not np.array_equal(grid, ACTION_GRID.astype(np.float32)):
        raise RuntimeError("AC5B1_ACTION_GRID_FAIL")
    current_obs = local_obs(bank, "local_", rows.transition, rows.agent)
    current_actions = actor_argmax(actor, current_obs, device)
    stored_actions = bank["action_index"][rows.transition, rows.agent].astype(np.int64)
    current_match = bool(np.array_equal(current_actions, stored_actions))
    if not current_match:
        raise RuntimeError("AC5B1_BANK_ACTOR_ACTION_MISMATCH")
    next_obs = local_obs(bank, "next_local_", rows.transition, rows.agent)
    next_actions = actor_argmax(actor, next_obs, device)
    after = tensor_state_sha256(actor.state_dict())
    if before != after:
        raise RuntimeError("AC5B1_ACTOR_STATE_CHANGED")
    return {
        "checkpoint_requested": str(ACTOR_REQUESTED),
        "checkpoint_loaded": str(ACTOR_LOADED),
        "checkpoint_sha256": ACTOR_SHA256,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "actor_updates": 0,
        "action_grid": grid.tolist(),
        "current_bank_argmax_match": current_match,
        "next_action_indices": next_actions,
    }


def metric_with_loss(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    metric = calibration(target, prediction)
    metric["mse_normalized"] = float(np.mean(((prediction - TRAIN_MEAN) / TRAIN_STD - (target - TRAIN_MEAN) / TRAIN_STD) ** 2)) if len(target) else None
    return metric


def phase_and_event_metrics(rows: AC3Rows, indices: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    transition = rows.transition[indices]
    target = rows.target[indices]
    result: dict[str, Any] = {"phases": {}, "events": {}}
    for name, phase_id in PHASE_TO_ID.items():
        take = rows.bank["phase_id"][transition] == phase_id
        result["phases"][name] = metric_with_loss(target[take], prediction[take])
    for name in EVENT_METRIC_NAMES:
        take = rows.bank["transition_event_flags"][transition, EVENT_TO_ID[name]]
        result["events"][name] = metric_with_loss(target[take], prediction[take])
    return result


def q_scale(prediction: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(np.mean(prediction)),
        "std": float(np.std(prediction)),
        "min": float(np.min(prediction)),
        "max": float(np.max(prediction)),
    }


def build_anchor_batch() -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[str]]:
    manifest = json.loads(AC4B_ANCHOR_MANIFEST.read_text())
    anchors = manifest["anchors"]
    with np.load(AC4B_RETURNS, allow_pickle=False) as source:
        returns = source["returns"].astype(np.float64)
        bc_actions = source["bc_actions"].astype(np.int64)
        stored_predictions = source["lq_predictions"].astype(np.float64)
    if len(anchors) != 64 or returns.shape != (64, 9) or stored_predictions.shape != (64, 9):
        raise RuntimeError("AC5B1_AC4B_FIXED_BRANCH_SHAPE_FAIL")
    local = {key: np.asarray([anchor["local_observation"][key] for anchor in anchors], dtype=np.float32) for key in ("self", "pursuers", "evaders", "obstacles", "masks", "types")}
    observations = {key: np.repeat(value, 9, axis=0) for key, value in local.items()}
    actions = np.tile(ACTION_GRID, (len(anchors), 1)).astype(np.float32) / ACTION_SCALE
    batch = {
        "obs": observations,
        "action": actions,
    }
    phases = [str(anchor["semantic_class"]) for anchor in anchors]
    return batch, returns, bc_actions, phases, stored_predictions


@torch.no_grad()
def ranking_eval(model: torch.nn.Module, batch: Mapping[str, Any], returns: np.ndarray, bc_actions: np.ndarray, phases: list[str], device: torch.device) -> dict[str, Any]:
    tensor_batch = {
        "obs": {key: to_tensor(value, device) for key, value in batch["obs"].items()},
        "action": to_tensor(batch["action"], device),
    }
    predicted = forward("LQ", model, tensor_batch).cpu().numpy().astype(np.float64) * TRAIN_STD + TRAIN_MEAN
    predicted = predicted.reshape(64, 9)
    rows = [ranking_row(predicted[index], returns[index], int(bc_actions[index])) for index in range(64)]
    overall = aggregate_ranking(rows)
    by_phase = {phase: aggregate_ranking([row for row, value in zip(rows, phases) if value == phase]) for phase in sorted(set(phases))}
    return {"overall": overall, "by_phase": by_phase, "predictions": predicted}


def prediction_metrics(model: torch.nn.Module, rows: AC3Rows, indices: np.ndarray, device: torch.device, ranking_inputs: tuple[Any, ...]) -> dict[str, Any]:
    normalized = predict("LQ", model, rows, indices, BATCH_SIZE)
    prediction = normalized * TRAIN_STD + TRAIN_MEAN
    batch, returns, bc_actions, phases, _ = ranking_inputs
    ranking = ranking_eval(model, batch, returns, bc_actions, phases, device)
    ranking.pop("predictions", None)
    score = {
        "mc_test": metric_with_loss(rows.target[indices], prediction),
        "phase_conditioned": phase_and_event_metrics(rows, indices, prediction),
        "q_scale_test": q_scale(prediction),
        "ranking": ranking,
    }
    return score


@torch.no_grad()
def bellman_targets(
    model: torch.nn.Module,
    actor_actions: np.ndarray,
    rows: AC3Rows,
    indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    bank = rows.bank
    output = np.empty(len(indices), dtype=np.float64)
    model.eval()
    for start in range(0, len(indices), BATCH_SIZE):
        chosen = indices[start:start + BATCH_SIZE]
        transition = rows.transition[chosen]
        agent = rows.agent[chosen]
        obs = local_obs(bank, "next_local_", transition, agent)
        batch_obs = {key: to_tensor(value, device) for key, value in obs.items()}
        action = to_tensor(ACTION_GRID[actor_actions[chosen]] / ACTION_SCALE, device)
        q_next = model(batch_obs, action).cpu().numpy().astype(np.float64) * TRAIN_STD + TRAIN_MEAN
        reward = bank["reward"][transition, agent].astype(np.float64)
        done = bank["done"][transition, agent].astype(np.float64)
        output[start:start + len(chosen)] = reward + GAMMA * (1.0 - done) * q_next
    return output


def target_drift_stats(current: np.ndarray, mc: np.ndarray, previous: np.ndarray | None) -> dict[str, Any]:
    error = current - mc
    result = {
        "rows": int(len(current)),
        "vs_mc_rmse": float(np.sqrt(np.mean(error ** 2))),
        "vs_mc_mae": float(np.mean(np.abs(error))),
        "vs_mc_bias": float(np.mean(error)),
        "vs_mc_pearson": safe_float(np.corrcoef(mc, current)[0, 1]) if np.std(mc) > 0 and np.std(current) > 0 else None,
        "target_mean": float(np.mean(current)),
        "target_std": float(np.std(current)),
        "target_min": float(np.min(current)),
        "target_max": float(np.max(current)),
    }
    if previous is None:
        result["delta_from_previous"] = None
    else:
        delta = current - previous
        result["delta_from_previous"] = {
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "mae": float(np.mean(np.abs(delta))),
            "mean": float(np.mean(delta)),
            "max_abs": float(np.max(np.abs(delta))),
        }
    return result


def initial_model(state: Mapping[str, torch.Tensor], config: Mapping[str, Any], seed: int, device: torch.device) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = make_model("LQ", config, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def train_arm(
    condition: str,
    seed: int,
    state: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    rows: AC3Rows,
    actor_actions: np.ndarray,
    actor: torch.nn.Module,
    ranking_inputs: tuple[Any, ...],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    started = time.monotonic()
    model = initial_model(state, config, seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    rng = np.random.default_rng(seed)
    points: list[dict[str, Any]] = []
    mc_targets = rows.target[test_indices].astype(np.float64)
    point0 = prediction_metrics(model, rows, test_indices, device, ranking_inputs)
    points.append({"update": 0, **point0, "bellman_target_vs_mc": None})
    previous_target = None
    updates = 0
    for block in range(1, BLOCKS + 1):
        target_model = None
        td_target = None
        if condition == "iterative_bellman":
            target_model = initial_model(model.state_dict(), config, seed + block, device)
            target_model.requires_grad_(False)
            target_model.eval()
            td_target = bellman_targets(target_model, actor_actions, rows, train_indices, device)
        for _ in range(UPDATES_PER_BLOCK):
            chosen = rng.choice(train_indices, size=BATCH_SIZE, replace=True)
            batch = rows.batch(chosen)
            if condition == "mc_control":
                target_value = rows.target[chosen].astype(np.float32)
            else:
                target_value = bellman_targets(target_model, actor_actions, rows, chosen, device).astype(np.float32)
            target = to_tensor((target_value - TRAIN_MEAN) / TRAIN_STD, device)
            model.train()
            loss = torch.mean((forward("LQ", model, batch) - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5))
            optimizer.step()
            updates += 1
        model.eval()
        metric = prediction_metrics(model, rows, test_indices, device, ranking_inputs)
        points.append({
            "update": updates,
            **metric,
            "last_train_loss_normalized": float(loss.detach().cpu()),
            "last_gradient_norm": gradient,
            "bellman_target_vs_mc": None if td_target is None else target_drift_stats(td_target, rows.target[train_indices].astype(np.float64), previous_target),
        })
        if td_target is not None:
            previous_target = td_target
        if target_model is not None:
            del target_model
        atomic_json(output / "progress.json", {
            "status": "fitting",
            "condition": condition,
            "seed": seed,
            "completed_updates": updates,
            "total_updates": BLOCKS * UPDATES_PER_BLOCK,
            "elapsed_seconds": time.monotonic() - started,
        })
    if updates != BLOCKS * UPDATES_PER_BLOCK:
        raise RuntimeError("AC5B1_UPDATE_COUNT_FAIL")
    checkpoint = output / "checkpoints" / f"lq_{condition}_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "cocap-ac5b1-lq-control-v1",
        "condition": condition,
        "seed": seed,
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "initial_checkpoint_sha256": INITIAL_LQ_SHA256,
        "normalization": {"train_mean": TRAIN_MEAN, "train_std": TRAIN_STD},
        "optimizer": {"name": "Adam", "learning_rate": 1e-4, "weight_decay": 1e-6, "batch_size": BATCH_SIZE, "grad_clip": 0.5},
        "updates": updates,
        "checkpoint_kind": "final",
    }, checkpoint)
    result = {
        "condition": condition,
        "seed": seed,
        "parameter_count": int(parameter_count(model)),
        "updates": updates,
        "actor_updates": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "points": points,
        "elapsed_seconds": time.monotonic() - started,
        "final_state_sha256": tensor_state_sha256(model.state_dict()),
    }
    del optimizer, model
    gc.collect()
    return result


def compare_initial_parity(
    state: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    rows: AC3Rows,
    test_indices: np.ndarray,
    ranking_inputs: tuple[Any, ...],
    device: torch.device,
) -> dict[str, Any]:
    model = initial_model(state, config, SEEDS[0], device)
    ranking = ranking_eval(model, ranking_inputs[0], ranking_inputs[1], ranking_inputs[2], ranking_inputs[3], device)
    stored = ranking_inputs[4]
    max_prediction_abs_diff = float(np.max(np.abs(ranking["predictions"] - stored)))
    reference = json.loads(AC4B_SUMMARY.read_text())["ranking"]["LQ"]["overall"]
    metric_diffs = {
        key: safe_float(ranking["overall"][key]["mean"] - reference[key]["mean"])
        for key in ("top1_agreement", "top3_overlap", "spearman", "regret", "sign_accuracy_vs_bc", "margin_filtered_sign_accuracy")
    }
    pass_gate = max_prediction_abs_diff <= 1e-4 and all(value is not None and abs(value) <= 1e-6 for value in metric_diffs.values())
    if not pass_gate:
        raise RuntimeError(f"AC5B1_ITERATION0_PARITY_FAIL: max_prediction_abs_diff={max_prediction_abs_diff}, metric_diffs={metric_diffs}")
    test = prediction_metrics(model, rows, test_indices, device, ranking_inputs)
    result = {
        "status": "PASS",
        "initial_checkpoint_sha256": INITIAL_LQ_SHA256,
        "initial_state_sha256": tensor_state_sha256(model.state_dict()),
        "ac4b_lq_state_sha256": "825b11e6753a9eb541dd265b087b74e4ed786fbc0430d04ab6fbd9802900eada",
        "max_ac4b_prediction_abs_diff": max_prediction_abs_diff,
        "ranking_metric_diffs_vs_ac4b": metric_diffs,
        "test_metrics": test,
    }
    del model
    return result


def aggregate_curve(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for update in CHECKPOINT_POINTS:
        points = [next(point for point in run["points"] if point["update"] == update) for run in runs]
        def mean_metric(path: tuple[str, ...]) -> dict[str, Any]:
            values = []
            for point in points:
                value: Any = point
                for key in path:
                    value = value[key]
                if value is not None and math.isfinite(float(value)):
                    values.append(float(value))
            return {"mean": float(np.mean(values)) if values else None, "seed_values": values}
        result[str(update)] = {
            "mc_test": {key: mean_metric(("mc_test", key)) for key in ("rmse", "mae", "explained_variance", "pearson", "spearman", "decile_calibration_error")},
            "ranking": {key: mean_metric(("ranking", "overall", key, "mean")) for key in ("top1_agreement", "top3_overlap", "spearman", "regret", "sign_accuracy_vs_bc", "margin_filtered_sign_accuracy")},
            "early_recovery": {key: mean_metric(("phase_conditioned", "events", "early_recovery", key)) for key in ("rmse", "mae", "spearman")},
            "early_recovery_sign_proxy": mean_metric(("phase_conditioned", "events", "early_recovery", "spearman")),
            "q_scale_test": {key: mean_metric(("q_scale_test", key)) for key in ("mean", "std", "min", "max")},
            "bellman_target_vs_mc": [point["bellman_target_vs_mc"] for point in points],
        }
    return result


def classify(runs_by_condition: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    labels = {}
    for condition, runs in runs_by_condition.items():
        points = [point for run in runs for point in run["points"]]
        finite = all(math.isfinite(float(point["mc_test"]["rmse"])) for point in points)
        initial_std = float(np.mean([run["points"][0]["q_scale_test"]["std"] for run in runs]))
        final_std = float(np.mean([run["points"][-1]["q_scale_test"]["std"] for run in runs]))
        initial_rmse = float(np.mean([run["points"][0]["mc_test"]["rmse"] for run in runs]))
        final_rmse = float(np.mean([run["points"][-1]["mc_test"]["rmse"] for run in runs]))
        scale_ratio = final_std / max(initial_std, 1e-9)
        rmse_ratio = final_rmse / max(initial_rmse, 1e-9)
        if not finite or scale_ratio > 5.0 or rmse_ratio > 3.0:
            label = "MC_CONTROL_UNSTABLE" if condition == "mc_control" else "ITERATIVE_BOOTSTRAP_DIVERGED"
        else:
            label = "MC_CONTROL_STABLE" if condition == "mc_control" else "ITERATIVE_BOOTSTRAP_STABLE"
        labels[condition] = {
            "classification": label,
            "finite": finite,
            "initial_test_rmse_mean": initial_rmse,
            "final_test_rmse_mean": final_rmse,
            "initial_q_std_mean": initial_std,
            "final_q_std_mean": final_std,
            "final_to_initial_q_std_ratio": scale_ratio,
            "final_to_initial_rmse_ratio": rmse_ratio,
            "rule": "finite curves and final Q-std ratio <=5 and final RMSE ratio <=3",
        }
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.device != "cpu":
        raise RuntimeError("AC5B1_CPU_ONLY")
    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device("cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    disk_before = shutil.disk_usage(ROOT)
    bank, bank_manifest = load_bank()
    config = make_config()
    config["training"] = {**config["training"], "batch_size": BATCH_SIZE, "updates": BLOCKS * UPDATES_PER_BLOCK, "learning_rate": 1e-4, "weight_decay": 1e-6, "grad_clip": 0.5}
    rows = make_argmax_rows(bank, device)
    episodes = episode_contract(bank)
    train_indices, test_indices = rows.indices(0), rows.indices(2)
    if len(train_indices) != 28172 or len(test_indices) != 3564:
        raise RuntimeError(f"AC5B1_ARGMAX_ROW_SPLIT_FAIL: train={len(train_indices)}, test={len(test_indices)}")
    state, state_hash = load_initial_state(config, device)
    actor, _ = load_actor(ACTOR_LOADED, "cpu")
    actor.requires_grad_(False)
    actor_meta = actor_contract(bank, rows, actor, device)
    ranking_inputs = build_anchor_batch()
    initial_parity = compare_initial_parity(state, config, rows, test_indices, ranking_inputs, device)
    all_runs: list[dict[str, Any]] = []
    runs_by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    started = time.monotonic()
    for condition in CONDITIONS:
        for seed in SEEDS:
            run = train_arm(condition, seed, state, config, rows, actor_meta["next_action_indices"], actor, ranking_inputs, train_indices, test_indices, OUT, device)
            runs_by_condition[condition].append(run)
            all_runs.append(run)
            atomic_json(OUT / "progress.json", {"status": "run_complete", "completed_runs": len(all_runs), "total_runs": 4, "last": {"condition": condition, "seed": seed}, "elapsed_seconds": time.monotonic() - started})
    actor_after = tensor_state_sha256(actor.state_dict())
    if actor_after != actor_meta["state_sha256_before"]:
        raise RuntimeError("AC5B1_ACTOR_CHANGED_AFTER_FITS")
    curves = {
        "schema": "ac5b1-learning-curves-v1",
        "points": list(CHECKPOINT_POINTS),
        "runs": all_runs,
        "aggregate": {condition: aggregate_curve(runs) for condition, runs in runs_by_condition.items()},
    }
    atomic_json(OUT / "learning_curves.json", curves)
    classifications = classify(runs_by_condition)
    disk_after = shutil.disk_usage(ROOT)
    output_bytes = int(sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()))
    actor_summary = {key: value for key, value in actor_meta.items() if key != "next_action_indices"}
    summary = {
        "schema": "ac5b1-mc-vs-iterative-bellman-lq-v1",
        "status": "complete",
        "scope": "LQ_only; no NQ/CQ/V/Actor/PPO",
        "bank": {"path": str(BANK_PATH), "sha256": sha256_file(BANK_PATH), "manifest_sha256": bank_manifest["bank"]["sha256"], "argmax_active_rows": int(len(rows.transition)), "train_rows": int(len(train_indices)), "test_rows": int(len(test_indices))},
        "initial_lq": {"path": str(INITIAL_LQ_PATH), "checkpoint_sha256": INITIAL_LQ_SHA256, "state_sha256": state_hash, "ac4b_state_sha256": EXPECTED_INITIAL_STATE_SHA256, "common_to_all_arms": True},
        "argmax_episode_split": episodes["argmax"],
        "normalization": {"mean": TRAIN_MEAN, "std": TRAIN_STD, "frozen": True},
        "actor": actor_summary,
        "contract": {"architecture": "historical A1 LocalQ, LegacyVorAdjFeatureBackbone", "hidden_dim": 256, "num_heads": 8, "num_layers": 4, "dropout": 0.0, "optimizer": "Adam", "learning_rate": 1e-4, "weight_decay": 1e-6, "batch_size": BATCH_SIZE, "grad_clip": 0.5, "updates_per_arm": BLOCKS * UPDATES_PER_BLOCK, "outer_blocks": BLOCKS, "updates_per_outer_block": UPDATES_PER_BLOCK, "sampler": "natural uniform active-agent train-row sampling with replacement", "seeds": list(SEEDS), "gamma": GAMMA, "mc_target": "exact individual realized full-episode MC return", "bellman_target": "individual reward + gamma*(1-done)*Q_target(next local observation, frozen canonical BC argmax action)", "target_copy": "copy previous online Q only between outer blocks"},
        "iteration_0_parity": initial_parity,
        "runs_by_condition": runs_by_condition,
        "aggregate_learning_curves": curves["aggregate"],
        "classifications": classifications,
        "forbidden_executed": {"nq_updates": 0, "cq_updates": 0, "v_updates": 0, "actor_updates": 0, "ppo_updates": 0, "new_trajectory_bank": False, "bc_updates": 0},
        "storage": {"disk_free_bytes_before": int(disk_before.free), "disk_free_bytes_after": int(disk_after.free), "output_bytes": output_bytes},
    }
    atomic_json(OUT / "summary.json", summary)
    atomic_json(OUT / "progress.json", {"status": "complete", "completed_runs": 4, "elapsed_seconds": time.monotonic() - started})
    print(json.dumps({"status": "complete", "initial_parity": initial_parity["status"], "classifications": classifications, "output_bytes": output_bytes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
