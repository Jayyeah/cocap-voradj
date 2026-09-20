#!/usr/bin/env python3
"""AC-5B2: Full-Natural MC action-support ablation for LQ.

Only two new LQ MC-control fits are run.  The full AC-2B train split is used
with its existing episode split and natural row sampler.  AC-5B1 argmax-only
curves are read as the paired control; no AC-5B1 fit, FQE fit, actor update,
or counterfactual-supervised fit is performed here.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path("/home/yjq/rl/CoCap1/cocap-voradj-critic-audit")
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from cocap_voradj.models.critic_identifiability import parameter_count
from tools.collect_ac2b_canonical_bc_bank_20260920 import ACTION_GRID, EVENT_TO_ID, PHASE_TO_ID
from tools.distill_forward_final_actor_20260908 import load_actor
from tools.run_ac3a_critic_sampler_audit_20260920 import AC3Rows, make_config, sha256_file
from tools.run_ac4b_formal_counterfactual_aw9_20260920 import aggregate_ranking, ranking_row
from tools.run_ac5b1_mc_vs_iterative_bellman_lq_20260920 import (
    AC4B_ANCHOR_MANIFEST,
    AC4B_RETURNS,
    AC4B_SUMMARY,
    ACTOR_LOADED,
    ACTOR_REQUESTED,
    ACTOR_SHA256,
    BANK_MANIFEST,
    BANK_PATH,
    BANK_SHA256,
    BATCH_SIZE,
    BLOCKS,
    CHECKPOINT_POINTS,
    EXPECTED_INITIAL_STATE_SHA256,
    INITIAL_LQ_PATH,
    INITIAL_LQ_SHA256,
    TRAIN_MEAN,
    TRAIN_STD,
    UPDATES_PER_BLOCK,
    atomic_json,
    build_anchor_batch,
    compare_initial_parity,
    initial_model,
    load_bank,
    load_initial_state,
    metric_with_loss,
    ranking_eval,
    tensor_state_sha256,
)
from tools.run_critic_identifiability_audit_20260917 import (
    ACTION_SCALE,
    calibration,
    forward,
    make_model,
    predict,
    to_tensor,
)


OUT = ROOT / "artifacts/2026-09-20_ac5b2"
AC5B1_CURVES = ROOT / "artifacts/2026-09-20_ac5b1/learning_curves.json"
AC5B1_SUMMARY = ROOT / "artifacts/2026-09-20_ac5b1/summary.json"
SEEDS = (2026091711, 2026091712)
GAMMA = 0.99
PHASE_SUPPORT_NAMES = ("pursuing", "pre_capture_cover", "post_capture", "pure_coverage")
RANKING_PHASE_NAMES = ("pursuing", "pre_capture_cover", "early_recovery", "recovery_pure")


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_rows(bank: Mapping[str, np.ndarray], device: torch.device, mode: int | None = None) -> AC3Rows:
    rows = AC3Rows(bank, device)
    if mode is not None:
        keep = bank["policy_mode_id"][rows.transition] == mode
        rows.transition = rows.transition[keep]
        rows.agent = rows.agent[keep]
    rows.split = bank["split_id"][rows.transition]
    rows.target = bank["mc_return"][rows.transition, rows.agent].astype(np.float32)
    rows.primary = bank["primary_event"][rows.transition]
    rows.semantic_class = bank["replay_semantic_class"][rows.transition, rows.agent].astype(np.int8)
    return rows


def all_episode_contract(bank: Mapping[str, np.ndarray]) -> dict[str, Any]:
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
    expected = {
        "train": {"episodes": 64, "mixed": 48, "pure_coverage": 16},
        "validation": {"episodes": 8, "mixed": 6, "pure_coverage": 2},
        "test": {"episodes": 8, "mixed": 6, "pure_coverage": 2},
    }
    actual = {}
    for split, wanted in expected.items():
        chosen = [row for row in records if row["split"] == split]
        actual[split] = {
            "episodes": len(chosen),
            "mixed": sum(row["scene"] == "mixed" for row in chosen),
            "pure_coverage": sum(row["scene"] == "pure_coverage" for row in chosen),
            "argmax": sum(row["policy_mode"] == "bc_argmax" for row in chosen),
            "sampled": sum(row["policy_mode"] == "bc_sample" for row in chosen),
            "episode_ids": [row["episode_id"] for row in chosen],
            "seeds": [row["seed"] for row in chosen],
        }
        if any(actual[split][key] != value for key, value in wanted.items()):
            raise RuntimeError(f"AC5B2_EPISODE_SPLIT_FAIL: {split}: {actual[split]}")
    return {"records": records, "by_split": actual}


def actor_identity() -> dict[str, Any]:
    if sha256_file(ACTOR_LOADED) != ACTOR_SHA256:
        raise RuntimeError("AC5B2_ACTOR_IDENTITY_FAIL")
    actor, _ = load_actor(ACTOR_LOADED, "cpu")
    actor.requires_grad_(False)
    before = state_hash(actor.state_dict())
    grid = actor.action_grid.detach().cpu().numpy().astype(np.float32)
    if not np.array_equal(grid, ACTION_GRID.astype(np.float32)):
        raise RuntimeError("AC5B2_ACTION_GRID_FAIL")
    after = state_hash(actor.state_dict())
    if before != after:
        raise RuntimeError("AC5B2_ACTOR_STATE_CHANGED_DURING_LOAD")
    return {
        "checkpoint_requested": str(ACTOR_REQUESTED),
        "checkpoint_loaded": str(ACTOR_LOADED),
        "checkpoint_sha256": ACTOR_SHA256,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "actor_optimizer_steps": 0,
        "action_grid": grid.tolist(),
    }


def row_phase_labels(rows: AC3Rows, indices: np.ndarray) -> np.ndarray:
    bank = rows.bank
    transition = rows.transition[indices]
    agent = rows.agent[indices]
    phase = np.empty(len(indices), dtype=object)
    phase_id = bank["phase_id"][transition]
    phase[phase_id == PHASE_TO_ID["pure_coverage"]] = "pure_coverage"
    phase[phase_id == PHASE_TO_ID["post_capture"]] = "post_capture"
    pre = phase_id == PHASE_TO_ID["pre_capture"]
    pursuing = pre & (bank["local_self"][transition, agent, 8] >= 0.5)
    phase[pursuing] = "pursuing"
    phase[pre & ~pursuing] = "pre_capture_cover"
    if np.any(pd := np.asarray([value is None for value in phase], dtype=bool)):
        raise RuntimeError(f"AC5B2_PHASE_LABEL_FAIL: {int(pd.sum())}")
    return phase.astype(str)


def action_distribution(actions: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(actions.astype(np.int64), minlength=9).astype(np.int64)
    total = int(counts.sum())
    fractions = counts.astype(np.float64) / max(total, 1)
    entropy = float(-np.sum(fractions[fractions > 0] * np.log(fractions[fractions > 0])))
    return {
        "count": [int(value) for value in counts],
        "fraction": [float(value) for value in fractions],
        "total_rows": total,
        "entropy_nats": entropy,
        "effective_action_count": float(np.exp(entropy)),
    }


def support_report(rows: AC3Rows, indices: np.ndarray, bc_actions: np.ndarray) -> dict[str, Any]:
    actions = rows.bank["action_index"][rows.transition[indices], rows.agent[indices]].astype(np.int64)
    phase = row_phase_labels(rows, indices)
    result = {
        "total_rows": int(len(indices)),
        "unique_episode_count": int(len(np.unique(rows.bank["episode_id"][rows.transition[indices]]))),
        "overall": action_distribution(actions),
        "by_phase": {name: action_distribution(actions[phase == name]) for name in PHASE_SUPPORT_NAMES},
    }
    global_fraction = np.asarray(result["overall"]["fraction"], dtype=np.float64)
    bc_frequency = global_fraction[bc_actions.astype(np.int64)]
    alt_frequency = np.stack([
        np.delete(global_fraction, int(action)) for action in bc_actions.astype(np.int64)
    ])
    result["ac4b_global_frequency"] = {
        "bc_selected_action_mean_fraction": float(np.mean(bc_frequency)),
        "bc_selected_action_median_fraction": float(np.median(bc_frequency)),
        "bc_selected_action_per_anchor_fraction": [float(value) for value in bc_frequency],
        "alternative_actions_mean_fraction": float(np.mean(alt_frequency)),
        "alternative_actions_median_fraction": float(np.median(alt_frequency)),
        "alternative_actions_per_anchor_mean_fraction": [float(value) for value in np.mean(alt_frequency, axis=1)],
        "global_action_fraction": [float(value) for value in global_fraction],
    }
    return result


def strip_ranking(ranking: dict[str, Any]) -> dict[str, Any]:
    ranking.pop("predictions", None)
    return ranking


def counterfactual_drift(q0: np.ndarray, qk: np.ndarray, bc_actions: np.ndarray) -> dict[str, Any]:
    q0_common = q0.mean(axis=1)
    qk_common = qk.mean(axis=1)
    q0_relative = q0 - q0_common[:, None]
    qk_relative = qk - qk_common[:, None]
    common_shift = qk_common - q0_common
    relative_shift = qk_relative - q0_relative
    delta = np.abs(qk - q0)
    row = np.arange(len(bc_actions))
    bc_shift = delta[row, bc_actions.astype(np.int64)]
    alternative_mask = np.ones_like(delta, dtype=bool)
    alternative_mask[row, bc_actions.astype(np.int64)] = False
    alternative_shift = (delta * alternative_mask).sum(axis=1) / 8.0
    return {
        "common_mode_shift_rmse": float(np.sqrt(np.mean(common_shift ** 2))),
        "common_mode_shift_mean_abs": float(np.mean(np.abs(common_shift))),
        "relative_action_shift_rmse": float(np.sqrt(np.mean(relative_shift ** 2))),
        "relative_action_shift_mean_abs": float(np.mean(np.abs(relative_shift))),
        "bc_selected_action_abs_shift_mean": float(np.mean(bc_shift)),
        "alternative_action_abs_shift_mean": float(np.mean(alternative_shift)),
        "alternative_to_bc_shift_ratio": float(np.mean(alternative_shift) / max(float(np.mean(bc_shift)), 1e-12)),
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    full_rows: AC3Rows,
    natural_test: np.ndarray,
    argmax_rows: AC3Rows,
    argmax_test: np.ndarray,
    ranking_inputs: tuple[Any, ...],
    q0: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    natural_prediction = predict("LQ", model, full_rows, natural_test, BATCH_SIZE) * TRAIN_STD + TRAIN_MEAN
    argmax_prediction = predict("LQ", model, argmax_rows, argmax_test, BATCH_SIZE) * TRAIN_STD + TRAIN_MEAN
    batch, returns, bc_actions, phases, _ = ranking_inputs
    ranking = ranking_eval(model, batch, returns, bc_actions, phases, device)
    qk = ranking["predictions"]
    ranking = strip_ranking(ranking)
    return {
        "natural_test": metric_with_loss(full_rows.target[natural_test], natural_prediction),
        "argmax_test": metric_with_loss(argmax_rows.target[argmax_test], argmax_prediction),
        "ranking": ranking,
        "counterfactual_drift": counterfactual_drift(q0, qk, bc_actions),
    }


def aggregate_values(runs: list[dict[str, Any]], update: int, section: str, key: str) -> dict[str, Any]:
    values = [float(next(point for point in run["points"] if point["update"] == update)[section][key]) for run in runs]
    return {"mean": float(np.mean(values)), "seed_values": values}


def aggregate_curves(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for update in CHECKPOINT_POINTS:
        point = {
            "natural_test": {key: aggregate_values(runs, update, "natural_test", key) for key in ("rmse", "mae", "explained_variance", "pearson", "spearman", "decile_calibration_error")},
            "argmax_test": {key: aggregate_values(runs, update, "argmax_test", key) for key in ("rmse", "mae", "explained_variance", "pearson", "spearman", "decile_calibration_error")},
            "ranking": {},
        }
        ranking_keys = ("top1_agreement", "top3_overlap", "spearman", "regret", "sign_accuracy_vs_bc", "margin_filtered_sign_accuracy")
        point["ranking"] = {
            key: {
                "mean": float(np.mean([next(value for value in run["points"] if value["update"] == update)["ranking"]["overall"][key]["mean"] for run in runs])),
                "seed_values": [next(value for value in run["points"] if value["update"] == update)["ranking"]["overall"][key]["mean"] for run in runs],
            }
            for key in ranking_keys
        }
        point["phase_ranking"] = {}
        for phase in RANKING_PHASE_NAMES:
            point["phase_ranking"][phase] = {
                key: {
                    "mean": float(np.mean([next(value for value in run["points"] if value["update"] == update)["ranking"]["by_phase"][phase][key]["mean"] for run in runs])),
                    "seed_values": [next(value for value in run["points"] if value["update"] == update)["ranking"]["by_phase"][phase][key]["mean"] for run in runs],
                }
                for key in ("spearman", "sign_accuracy_vs_bc")
            }
        point["counterfactual_drift"] = {key: aggregate_values(runs, update, "counterfactual_drift", key) for key in (
            "common_mode_shift_rmse", "common_mode_shift_mean_abs", "relative_action_shift_rmse", "relative_action_shift_mean_abs",
            "bc_selected_action_abs_shift_mean", "alternative_action_abs_shift_mean", "alternative_to_bc_shift_ratio",
        )}
        result[str(update)] = point
    return result


def read_ac5b1_control() -> dict[str, Any]:
    curves = json.loads(AC5B1_CURVES.read_text())
    summary = json.loads(AC5B1_SUMMARY.read_text())
    if curves["schema"] != "ac5b1-learning-curves-v1" or summary["iteration_0_parity"]["status"] != "PASS":
        raise RuntimeError("AC5B1_CONTROL_ARTIFACT_FAIL")
    return {"curves": curves, "summary": summary}


def ac5b1_phase_control(control: Mapping[str, Any], update: int, phase: str, key: str) -> float:
    values = []
    for run in control["curves"]["runs"]:
        point = next(value for value in run["points"] if value["update"] == update)
        values.append(float(point["ranking"]["by_phase"][phase][key]["mean"]))
    return float(np.mean(values))


def paired_ranking(full: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for update in CHECKPOINT_POINTS:
        row = {}
        for key in ("spearman", "sign_accuracy_vs_bc", "regret"):
            full_value = float(full[str(update)]["ranking"][key]["mean"])
            control_value = float(control["curves"]["aggregate"]["mc_control"][str(update)]["ranking"][key]["mean"])
            row[key] = {"full_natural": full_value, "argmax_only": control_value, "delta_full_minus_argmax": full_value - control_value}
        result[str(update)] = row
    return result


def final_argmax_drift(control: Mapping[str, Any], state: Mapping[str, torch.Tensor], config: Mapping[str, Any], ranking_inputs: tuple[Any, ...], device: torch.device) -> dict[str, Any]:
    batch, returns, bc_actions, phases, _ = ranking_inputs
    q0_model = initial_model(state, config, SEEDS[0], device)
    q0 = ranking_eval(q0_model, batch, returns, bc_actions, phases, device)["predictions"]
    del q0_model
    values = []
    for run in control["curves"]["runs"]:
        checkpoint = ROOT / str(run["checkpoint"])
        if not checkpoint.exists():
            return {"status": "NOT_AVAILABLE", "reason": str(checkpoint)}
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model = make_model("LQ", config, device)
        model.load_state_dict(payload["state_dict"], strict=True)
        ranking = ranking_eval(model, batch, returns, bc_actions, phases, device)
        values.append(counterfactual_drift(q0, ranking["predictions"], bc_actions))
        del model
    keys = values[0].keys()
    return {"status": "PASS", "update": 1200, **{key: float(np.mean([value[key] for value in values])) for key in keys}}


def classify(full: Mapping[str, Any], paired: Mapping[str, Any], final_argmax: Mapping[str, Any]) -> dict[str, Any]:
    updates = [300, 600, 900, 1200]
    rank_delta = [float(paired[str(update)]["spearman"]["delta_full_minus_argmax"]) for update in updates]
    sign_delta = [float(paired[str(update)]["sign_accuracy_vs_bc"]["delta_full_minus_argmax"]) for update in updates]
    final_full = full["1200"]["counterfactual_drift"]
    final_relative_argmax = None if final_argmax.get("status") != "PASS" else float(final_argmax["relative_action_shift_rmse"])
    full_relative = float(final_full["relative_action_shift_rmse"]["mean"])
    protects = bool(np.mean(rank_delta) >= 0.05 and rank_delta[-1] >= 0.03 and np.mean(sign_delta) >= -0.01 and (final_relative_argmax is None or full_relative <= final_relative_argmax))
    both_collapse = bool(float(full["1200"]["ranking"]["spearman"]["mean"]) <= 0.0 and float(paired["1200"]["spearman"]["argmax_only"]) <= 0.0 and float(full["1200"]["ranking"]["sign_accuracy_vs_bc"]["mean"]) < 0.5)
    if protects:
        label = "ACTION_SUPPORT_PROTECTS_RANKING"
    elif both_collapse:
        label = "BEHAVIOR_MC_REGRESSION_RANKING_UNDERIDENTIFIED"
    else:
        label = "ACTION_SUPPORT_EFFECT_MIXED"
    phase_delta = {}
    for phase in RANKING_PHASE_NAMES:
        phase_delta[phase] = {
            "spearman_delta_at_1200": float(full["1200"]["phase_ranking"][phase]["spearman"]["mean"] - final_argmax.get("phase_control_spearman", {}).get(phase, float("nan"))),
            "sign_delta_at_1200": float(full["1200"]["phase_ranking"][phase]["sign_accuracy_vs_bc"]["mean"] - final_argmax.get("phase_control_sign", {}).get(phase, float("nan"))),
        }
    return {
        "classification": label,
        "ranking_delta_mean_300_to_1200": float(np.mean(rank_delta)),
        "ranking_delta_at_1200": rank_delta[-1],
        "sign_delta_mean_300_to_1200": float(np.mean(sign_delta)),
        "full_natural_relative_action_shift_rmse_at_1200": full_relative,
        "argmax_only_relative_action_shift_rmse_at_1200": final_relative_argmax,
        "phase_deltas": phase_delta,
        "rule": "protects requires mean ranking Spearman gain >=0.05, final gain >=0.03, mean sign delta >=-0.01, and no larger endpoint relative-action drift when the AC-5B1 final checkpoint is available",
    }


def train_full_natural(
    seed: int,
    state: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    full_rows: AC3Rows,
    argmax_rows: AC3Rows,
    train_indices: np.ndarray,
    natural_test: np.ndarray,
    argmax_test: np.ndarray,
    ranking_inputs: tuple[Any, ...],
    q0: np.ndarray,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    started = time.monotonic()
    model = initial_model(state, config, seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    rng = np.random.default_rng(seed)
    points = []
    initial = evaluate_model(model, full_rows, natural_test, argmax_rows, argmax_test, ranking_inputs, q0, device)
    points.append({"update": 0, **initial})
    updates = 0
    for _block in range(BLOCKS):
        for _ in range(UPDATES_PER_BLOCK):
            chosen = rng.choice(train_indices, size=BATCH_SIZE, replace=True)
            batch = full_rows.batch(chosen)
            target = to_tensor((full_rows.target[chosen] - TRAIN_MEAN) / TRAIN_STD, device)
            model.train()
            loss = torch.mean((forward("LQ", model, batch) - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5))
            optimizer.step()
            updates += 1
        model.eval()
        evaluation = evaluate_model(model, full_rows, natural_test, argmax_rows, argmax_test, ranking_inputs, q0, device)
        points.append({"update": updates, **evaluation, "last_train_loss_normalized": float(loss.detach().cpu()), "last_gradient_norm": gradient})
        atomic_json(output / "progress.json", {"status": "fitting", "condition": "full_natural_mc", "seed": seed, "completed_updates": updates, "total_updates": BLOCKS * UPDATES_PER_BLOCK, "elapsed_seconds": time.monotonic() - started})
    if updates != BLOCKS * UPDATES_PER_BLOCK:
        raise RuntimeError("AC5B2_UPDATE_COUNT_FAIL")
    checkpoint = output / "checkpoints" / f"lq_full_natural_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "cocap-ac5b2-full-natural-mc-lq-v1",
        "condition": "full_natural_mc",
        "seed": seed,
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "initial_checkpoint_sha256": INITIAL_LQ_SHA256,
        "normalization": {"train_mean": TRAIN_MEAN, "train_std": TRAIN_STD},
        "optimizer": {"name": "Adam", "learning_rate": 1e-4, "weight_decay": 1e-6, "batch_size": BATCH_SIZE, "grad_clip": 0.5},
        "updates": updates,
        "checkpoint_kind": "final",
    }, checkpoint)
    result = {
        "condition": "full_natural_mc",
        "seed": seed,
        "parameter_count": int(parameter_count(model)),
        "updates": updates,
        "actor_updates": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "final_state_sha256": tensor_state_sha256(model.state_dict()),
        "points": points,
        "elapsed_seconds": time.monotonic() - started,
    }
    del optimizer, model
    gc.collect()
    return result


def main() -> None:
    import argparse
    import shutil

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.device != "cpu":
        raise RuntimeError("AC5B2_CPU_ONLY")
    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device("cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    disk_before = shutil.disk_usage(ROOT)
    bank, bank_manifest = load_bank()
    config = make_config()
    config["training"] = {**config["training"], "batch_size": BATCH_SIZE, "updates": BLOCKS * UPDATES_PER_BLOCK, "learning_rate": 1e-4, "weight_decay": 1e-6, "grad_clip": 0.5}
    full_rows = load_rows(bank, device)
    argmax_rows = load_rows(bank, device, mode=0)
    episodes = all_episode_contract(bank)
    full_train, full_test = full_rows.indices(0), full_rows.indices(2)
    argmax_train, argmax_test = argmax_rows.indices(0), argmax_rows.indices(2)
    if (len(full_train), len(full_test), len(argmax_train), len(argmax_test)) != (54784, 6620, 28172, 3564):
        raise RuntimeError(f"AC5B2_ROW_SPLIT_FAIL: full train/test={len(full_train)}/{len(full_test)}; argmax={len(argmax_train)}/{len(argmax_test)}")
    state, state_sha = load_initial_state(config, device)
    if state_sha != EXPECTED_INITIAL_STATE_SHA256:
        raise RuntimeError("AC5B2_INITIAL_STATE_NOT_AC5B1")
    actor_meta = actor_identity()
    control = read_ac5b1_control()
    ranking_inputs = build_anchor_batch()
    parity = compare_initial_parity(state, config, argmax_rows, argmax_test, ranking_inputs, device)
    if parity["status"] != "PASS":
        raise RuntimeError("AC5B2_ITERATION0_PARITY_FAIL")
    baseline = initial_model(state, config, SEEDS[0], device)
    q0 = ranking_eval(baseline, ranking_inputs[0], ranking_inputs[1], ranking_inputs[2], ranking_inputs[3], device)["predictions"]
    del baseline
    support = {
        "schema": "ac5b2-action-support-v1",
        "contract": {"sampler": "uniform natural active-agent rows with replacement", "target": "exact individual full-episode MC return", "gamma": GAMMA},
        "argmax_only": support_report(argmax_rows, argmax_train, ranking_inputs[2]),
        "full_natural": support_report(full_rows, full_train, ranking_inputs[2]),
    }
    atomic_json(OUT / "action_support.json", support)
    runs = []
    started = time.monotonic()
    for seed in SEEDS:
        run = train_full_natural(seed, state, config, full_rows, argmax_rows, full_train, full_test, argmax_test, ranking_inputs, q0, OUT, device)
        runs.append(run)
        atomic_json(OUT / "progress.json", {"status": "run_complete", "completed_runs": len(runs), "total_runs": 2, "last": {"condition": "full_natural_mc", "seed": seed}, "elapsed_seconds": time.monotonic() - started})
    curves = {"schema": "ac5b2-learning-curves-v1", "points": list(CHECKPOINT_POINTS), "runs": runs, "aggregate": aggregate_curves(runs)}
    paired = paired_ranking(curves["aggregate"], control)
    final_argmax = final_argmax_drift(control, state, config, ranking_inputs, device)
    final_argmax["phase_control_spearman"] = {phase: ac5b1_phase_control(control, 1200, phase, "spearman") for phase in RANKING_PHASE_NAMES}
    final_argmax["phase_control_sign"] = {phase: ac5b1_phase_control(control, 1200, phase, "sign_accuracy_vs_bc") for phase in RANKING_PHASE_NAMES}
    classification = classify(curves["aggregate"], paired, final_argmax)
    actor_after = actor_meta["state_sha256_after"]
    disk_after = shutil.disk_usage(ROOT)
    output_bytes = int(sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()))
    atomic_json(OUT / "learning_curves.json", {**curves, "paired_ranking_deltas_vs_ac5b1": paired, "ac5b1_control": {"source": str(AC5B1_CURVES), "checkpoint_source": "AC-5B1 final checkpoints only for endpoint drift"}})
    summary = {
        "schema": "ac5b2-action-support-ablation-v1",
        "status": "complete",
        "scope": "new Full-Natural MC LQ x2 only; no AC5B1 rerun, no FQE, no NQ/CQ/V/Actor/PPO",
        "bank": {"path": str(BANK_PATH), "sha256": sha256_file(BANK_PATH), "manifest_sha256": bank_manifest["bank"]["sha256"], "full_train_rows": len(full_train), "full_test_rows": len(full_test), "argmax_train_rows": len(argmax_train), "argmax_test_rows": len(argmax_test)},
        "initial_lq": {"path": str(INITIAL_LQ_PATH), "checkpoint_sha256": INITIAL_LQ_SHA256, "state_sha256": state_sha, "ac5b1_state_sha256": EXPECTED_INITIAL_STATE_SHA256, "bit_exact_initial": parity["initial_state_sha256"] == EXPECTED_INITIAL_STATE_SHA256},
        "actor": {**actor_meta, "state_sha256_after_all_runs": actor_after},
        "episode_split": episodes["by_split"],
        "normalization": {"mean": TRAIN_MEAN, "std": TRAIN_STD, "frozen": True},
        "contract": {"architecture": "historical A1 LocalQ, LegacyVorAdjFeatureBackbone", "optimizer": "Adam", "learning_rate": 1e-4, "weight_decay": 1e-6, "batch_size": BATCH_SIZE, "grad_clip": 0.5, "outer_blocks": BLOCKS, "updates_per_block": UPDATES_PER_BLOCK, "total_updates_per_seed": BLOCKS * UPDATES_PER_BLOCK, "sampler": "uniform all full-natural train active-agent rows with replacement", "target": "exact individual-agent full MC return", "gamma": GAMMA, "counterfactual_training": False},
        "iteration_0_parity": parity,
        "action_support_artifact": str(OUT / "action_support.json"),
        "runs": runs,
        "aggregate_learning_curves": curves["aggregate"],
        "paired_ranking_deltas_vs_ac5b1": paired,
        "argmax_endpoint_drift": final_argmax,
        "classification": classification,
        "forbidden_executed": {"fqe": 0, "bellman_updates": 0, "nq_updates": 0, "cq_updates": 0, "v_updates": 0, "actor_updates": 0, "ppo_updates": 0, "gae": 0, "new_trajectory_bank": False},
        "storage": {"disk_free_bytes_before": int(disk_before.free), "disk_free_bytes_after": int(disk_after.free), "output_bytes": output_bytes},
    }
    atomic_json(OUT / "summary.json", summary)
    final_output_bytes = int(sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()))
    if final_output_bytes != summary["storage"]["output_bytes"]:
        summary["storage"]["output_bytes"] = final_output_bytes
        atomic_json(OUT / "summary.json", summary)
    atomic_json(OUT / "progress.json", {"status": "complete", "completed_runs": 2, "elapsed_seconds": time.monotonic() - started})
    print(json.dumps({"status": "complete", "parity": parity["status"], "classification": classification["classification"], "output_bytes": output_bytes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
