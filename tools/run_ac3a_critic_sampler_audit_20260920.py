#!/usr/bin/env python3
"""AC-3A: natural versus IQN-reference phase-balanced MC critic sampling.

This runner fits only LQ and NQ on the frozen AC-2B canonical-BC bank.  It
uses the historical A1 critic architecture and optimizer contract, changes
only the training row sampler, and evaluates both conditions on the same
natural held-out episode split.  There is no bootstrap, target network, GAE,
PPO, actor update, CQ, V, or counterfactual action-ranking experiment.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
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
from tools.run_critic_identifiability_audit_20260917 import (
    EVENT_TO_ID,
    BankRows,
    calibration,
    forward,
    make_model,
    predict,
    to_tensor,
)


OUT = ROOT / "artifacts/2026-09-20_ac3a"
BANK_PATH = ROOT / "artifacts/2026-09-20_ac2b/canonical_bc_critic_bank_v2.npz"
BANK_MANIFEST = ROOT / "artifacts/2026-09-20_ac2b/bank_manifest.json"
BANK_SHA256 = "07e08d17ae6283d785e0d2e00307f330b13a88dcd0f8ce83f92f0d451ad7808a"
SEEDS = (2026091711, 2026091712)
CRITICS = ("LQ", "NQ")
CONDITIONS = ("natural", "iqn_balanced")
CLASS_NAMES = ("pursuing", "pre_capture_cover", "post_capture_real", "recovery_pure")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
BALANCED_QUOTA = np.asarray([64, 16, 32, 16], dtype=np.int64)
SPLITS = ("train", "validation", "test")
EVENT_NAMES = (
    "early_recovery", "late_recovery", "pure_coverage", "ring2", "ring3",
    "capture_transition",
)
EVENT_MASKS = {
    "early_recovery": ("early_recovery",),
    "late_recovery": ("late_recovery",),
    "pure_coverage": ("pure_coverage",),
    "ring2": ("ring2",),
    "ring3": ("ring3",),
    "capture_transition": ("normal_capture", "stationary_capture"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_bank() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    actual = sha256_file(BANK_PATH)
    if actual != BANK_SHA256:
        raise RuntimeError(f"AC3A_BANK_IDENTITY_FAIL: {actual} != {BANK_SHA256}")
    manifest = json.loads(BANK_MANIFEST.read_text())
    if manifest["bank"]["sha256"] != actual or manifest["status"] != "formal_bank_complete":
        raise RuntimeError("AC3A_BANK_MANIFEST_MISMATCH")
    with np.load(BANK_PATH, allow_pickle=False) as source:
        bank = {name: source[name] for name in source.files}
    if sorted({int(value) for value in bank["split_id"]}) != [0, 1, 2]:
        raise RuntimeError("AC3A_SPLIT_CONTRACT_FAIL")
    if not {"neighbor_action_index", "neighbor_action_aw"}.issubset(bank):
        raise RuntimeError("AC3A_NQ_SCHEMA_MISSING_NEIGHBOR_ACTIONS")
    return bank, manifest


class AC3Rows(BankRows):
    """Historical A1 row adapter plus AC-2B semantic-class labels."""

    def __init__(self, bank: Mapping[str, np.ndarray], device: torch.device):
        super().__init__(bank, device)
        self.semantic_class = bank["replay_semantic_class"][self.transition, self.agent].astype(np.int8)


def make_config() -> dict[str, Any]:
    return {
        "training": {
            "hidden_dim": 256,
            "num_heads": 8,
            "num_layers": 4,
            "dropout": 0.0,
            "batch_size": 256,
            "updates": 1200,
            "learning_rate": 1e-4,
            "weight_decay": 1e-6,
            "grad_clip": 0.5,
            "validation_every": 100,
        },
        "samplers": {
            "natural": "uniform row sampling from all train active-agent rows",
            "iqn_balanced": "per-128-row quota block semantic class quota [64,16,32,16]; A1 batch 256 uses two blocks, with replacement within class",
        },
    }


def class_indices(rows: AC3Rows, indices: np.ndarray) -> list[np.ndarray]:
    return [indices[rows.semantic_class[indices] == class_id] for class_id in range(4)]


def sampled_batch(
    condition: str,
    rows: AC3Rows,
    train_indices: np.ndarray,
    by_class: list[np.ndarray],
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if condition == "natural":
        chosen = rng.choice(train_indices, size=batch_size, replace=True)
    else:
        quota_block = int(BALANCED_QUOTA.sum())
        if batch_size % quota_block != 0:
            raise ValueError("AC3A batch size must contain an integer number of 128-row quota blocks")
        block_count = batch_size // quota_block
        pieces = []
        for _ in range(block_count):
            pieces.extend(rng.choice(pool, size=int(quota), replace=True) for pool, quota in zip(by_class, BALANCED_QUOTA))
        chosen = np.concatenate(pieces)
        rng.shuffle(chosen)
    counts = np.bincount(rows.semantic_class[chosen], minlength=4).astype(np.int64)
    return chosen, counts


def metric_with_loss(target: np.ndarray, prediction: np.ndarray, mean: float, scale: float) -> dict[str, Any]:
    metric = calibration(target, prediction)
    normalized_target = (target - mean) / scale
    normalized_prediction = (prediction - mean) / scale
    metric["mse_normalized"] = float(np.mean((normalized_prediction - normalized_target) ** 2)) if len(target) else None
    return metric


def class_metrics(rows: AC3Rows, indices: np.ndarray, prediction: np.ndarray, mean: float, scale: float) -> dict[str, Any]:
    result = {}
    labels = rows.semantic_class[indices]
    for class_id, name in enumerate(CLASS_NAMES):
        take = labels == class_id
        result[name] = metric_with_loss(rows.target[indices][take], prediction[take], mean, scale)
    return result


def event_metrics(rows: AC3Rows, indices: np.ndarray, prediction: np.ndarray, mean: float, scale: float) -> dict[str, Any]:
    result = {}
    transition = rows.transition[indices]
    for name, events in EVENT_MASKS.items():
        take = np.zeros(len(indices), dtype=bool)
        for event in events:
            if event == "capture_transition":
                event_id = EVENT_TO_ID[event] if event in EVENT_TO_ID else None
            else:
                event_id = EVENT_TO_ID[event]
            if event == "capture_transition":
                take |= rows.bank["transition_event_flags"][transition, EVENT_TO_ID["normal_capture"]]
                take |= rows.bank["transition_event_flags"][transition, EVENT_TO_ID["stationary_capture"]]
            else:
                take |= rows.bank["transition_event_flags"][transition, event_id]
        result[name] = metric_with_loss(rows.target[indices][take], prediction[take], mean, scale)
    return result


def evaluate_kind(kind: str, rows: AC3Rows, model: torch.nn.Module, indices: np.ndarray, batch_size: int, mean: float, scale: float) -> tuple[np.ndarray, dict[str, Any]]:
    normalized = predict(kind, model, rows, indices, batch_size)
    prediction = normalized * scale + mean
    return prediction, {
        "overall": metric_with_loss(rows.target[indices], prediction, mean, scale),
        "classes": class_metrics(rows, indices, prediction, mean, scale),
        "events": event_metrics(rows, indices, prediction, mean, scale),
    }


def fit_one(
    kind: str,
    condition: str,
    seed: int,
    config: Mapping[str, Any],
    rows: AC3Rows,
    output: Path,
    device: torch.device,
    mean: float,
    scale: float,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_cfg = config["training"]
    model = make_model(kind, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    train_indices = rows.indices(0)
    validation_indices = rows.indices(1)
    test_indices = rows.indices(2)
    by_class = class_indices(rows, train_indices)
    rng = np.random.default_rng(seed)
    draw_counts = np.zeros(4, dtype=np.int64)
    seen_rows = [set() for _ in range(4)]
    history = []
    best = None
    started = time.monotonic()
    for update in range(1, int(train_cfg["updates"]) + 1):
        chosen, counts = sampled_batch(condition, rows, train_indices, by_class, rng, int(train_cfg["batch_size"]))
        draw_counts += counts
        for class_id in range(4):
            seen_rows[class_id].update(chosen[rows.semantic_class[chosen] == class_id].tolist())
        batch = rows.batch(chosen)
        target = to_tensor((rows.target[chosen] - mean) / scale, device)
        model.train()
        loss = torch.mean((forward(kind, model, batch) - target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"])))
        optimizer.step()
        if update % int(train_cfg["validation_every"]) == 0 or update == int(train_cfg["updates"]):
            prediction, validation = evaluate_kind(kind, rows, model, validation_indices, int(train_cfg["batch_size"]), mean, scale)
            del prediction
            score = float(validation["overall"]["rmse"])
            history.append({"update": update, "train_loss_normalized": float(loss.detach().cpu()), "gradient_norm": gradient, "validation_overall": validation["overall"]})
            if best is None or score < best["validation_rmse"]:
                best = {
                    "validation_rmse": score,
                    "update": update,
                    "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                }
    if best is None:
        raise RuntimeError("AC3A no validation checkpoint")
    model.load_state_dict(best.pop("state"))
    train_prediction, train_score = evaluate_kind(kind, rows, model, train_indices, int(train_cfg["batch_size"]), mean, scale)
    val_prediction, val_score = evaluate_kind(kind, rows, model, validation_indices, int(train_cfg["batch_size"]), mean, scale)
    test_prediction, test_score = evaluate_kind(kind, rows, model, test_indices, int(train_cfg["batch_size"]), mean, scale)
    del train_prediction, val_prediction, test_prediction
    checkpoint = output / "checkpoints" / f"{kind.lower()}_{condition}_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "cocap-ac3a-critic-sampler-v1",
        "kind": kind,
        "condition": condition,
        "seed": seed,
        "state_dict": model.state_dict(),
        "normalization": {"train_mean": mean, "train_std": scale},
        "best_validation": best,
        "architecture_contract": config["training"],
    }, checkpoint)
    result = {
        "kind": kind,
        "condition": condition,
        "seed": seed,
        "parameter_count": parameter_count(model),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_validation": best,
        "history": history,
        "train": train_score,
        "validation": val_score,
        "test": test_score,
        "sampling": {
            "draw_count_total": int(draw_counts.sum()),
            "draw_count_by_class": {name: int(draw_counts[index]) for index, name in enumerate(CLASS_NAMES)},
            "draw_ratio_by_class": {name: float(draw_counts[index] / max(draw_counts.sum(), 1)) for index, name in enumerate(CLASS_NAMES)},
            "unique_count_by_class": {name: int(len(seen_rows[index])) for index, name in enumerate(CLASS_NAMES)},
            "reuse_ratio_draws_per_unique_by_class": {name: float(draw_counts[index] / max(len(seen_rows[index]), 1)) for index, name in enumerate(CLASS_NAMES)},
            "train_available_rows_by_class": {name: int(len(by_class[index])) for index, name in enumerate(CLASS_NAMES)},
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def replay_sampling_stats(condition: str, seed: int, rows: AC3Rows, updates: int, batch_size: int) -> dict[str, Any]:
    train_indices = rows.indices(0)
    by_class = class_indices(rows, train_indices)
    rng = np.random.default_rng(seed)
    draw_counts = np.zeros(4, dtype=np.int64)
    seen_rows = [set() for _ in range(4)]
    for _ in range(updates):
        chosen, counts = sampled_batch(condition, rows, train_indices, by_class, rng, batch_size)
        draw_counts += counts
        for class_id in range(4):
            seen_rows[class_id].update(chosen[rows.semantic_class[chosen] == class_id].tolist())
    return {
        "draw_count_total": int(draw_counts.sum()),
        "draw_count_by_class": {name: int(draw_counts[index]) for index, name in enumerate(CLASS_NAMES)},
        "draw_ratio_by_class": {name: float(draw_counts[index] / max(draw_counts.sum(), 1)) for index, name in enumerate(CLASS_NAMES)},
        "unique_count_by_class": {name: int(len(seen_rows[index])) for index, name in enumerate(CLASS_NAMES)},
        "reuse_ratio_draws_per_unique_by_class": {name: float(draw_counts[index] / max(len(seen_rows[index]), 1)) for index, name in enumerate(CLASS_NAMES)},
        "train_available_rows_by_class": {name: int(len(by_class[index])) for index, name in enumerate(CLASS_NAMES)},
    }


def rescore_one(
    kind: str,
    condition: str,
    seed: int,
    config: Mapping[str, Any],
    rows: AC3Rows,
    output: Path,
    device: torch.device,
    mean: float,
    scale: float,
) -> dict[str, Any]:
    """Rescore completed checkpoints without optimizer or parameter updates."""
    checkpoint = output / "checkpoints" / f"{kind.lower()}_{condition}_seed{seed}.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != "cocap-ac3a-critic-sampler-v1" or payload.get("kind") != kind or payload.get("condition") != condition or int(payload.get("seed")) != seed:
        raise RuntimeError(f"AC3A_CHECKPOINT_CONTRACT_FAIL: {checkpoint}")
    normalization = payload["normalization"]
    if abs(float(normalization["train_mean"]) - mean) > 1e-6 or abs(float(normalization["train_std"]) - scale) > 1e-6:
        raise RuntimeError(f"AC3A_CHECKPOINT_NORMALIZATION_FAIL: {checkpoint}")
    model = make_model(kind, config, device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    train_score = evaluate_kind(kind, rows, model, rows.indices(0), int(config["training"]["batch_size"]), mean, scale)[1]
    val_score = evaluate_kind(kind, rows, model, rows.indices(1), int(config["training"]["batch_size"]), mean, scale)[1]
    test_score = evaluate_kind(kind, rows, model, rows.indices(2), int(config["training"]["batch_size"]), mean, scale)[1]
    sampling = replay_sampling_stats(condition, seed, rows, int(config["training"]["updates"]), int(config["training"]["batch_size"]))
    result = {
        "kind": kind,
        "condition": condition,
        "seed": seed,
        "parameter_count": parameter_count(model),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_validation": payload["best_validation"],
        "history": [],
        "train": train_score,
        "validation": val_score,
        "test": test_score,
        "sampling": sampling,
        "rescore_only": True,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def mean_spread(values: list[float | None]) -> dict[str, Any]:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return {"mean": None, "seed_spread_sd": None, "n": 0}
    array = np.asarray(usable, dtype=np.float64)
    return {"mean": float(array.mean()), "seed_spread_sd": float(array.std(ddof=0)), "n": len(usable)}


def aggregate_metric(runs: list[dict[str, Any]], split: str, section: str, key: str) -> dict[str, Any]:
    return mean_spread([run[split][section].get(key) for run in runs])


def aggregate_condition(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"seeds": [run["seed"] for run in runs]}
    for split in ("train", "validation", "test"):
        result[split] = {}
        for section in ("overall",):
            result[split][section] = {
                key: aggregate_metric(runs, split, section, key)
                for key in ("rmse", "mae", "explained_variance", "pearson", "spearman", "calibration_slope_target_on_prediction", "calibration_intercept", "decile_calibration_error", "mse_normalized")
            }
        result[split]["classes"] = {
            class_name: {
                key: mean_spread([run[split]["classes"][class_name].get(key) for run in runs])
                for key in ("rows", "rmse", "mae", "explained_variance", "pearson", "spearman", "calibration_slope_target_on_prediction", "calibration_intercept", "decile_calibration_error", "mse_normalized")
            }
            for class_name in CLASS_NAMES
        }
        result[split]["events"] = {
            event_name: {
                key: mean_spread([run[split]["events"][event_name].get(key) for run in runs])
                for key in ("rows", "rmse", "mae", "explained_variance", "pearson", "spearman", "calibration_slope_target_on_prediction", "calibration_intercept", "decile_calibration_error", "mse_normalized")
            }
            for event_name in EVENT_NAMES
        }
    result["sampling"] = {
        "draw_count_total": mean_spread([run["sampling"]["draw_count_total"] for run in runs]),
        "draw_count_by_class": {name: mean_spread([run["sampling"]["draw_count_by_class"][name] for run in runs]) for name in CLASS_NAMES},
        "draw_ratio_by_class": {name: mean_spread([run["sampling"]["draw_ratio_by_class"][name] for run in runs]) for name in CLASS_NAMES},
        "unique_count_by_class": {name: mean_spread([run["sampling"]["unique_count_by_class"][name] for run in runs]) for name in CLASS_NAMES},
        "reuse_ratio_draws_per_unique_by_class": {name: mean_spread([run["sampling"]["reuse_ratio_draws_per_unique_by_class"][name] for run in runs]) for name in CLASS_NAMES},
        "per_seed": [run["sampling"] for run in runs],
    }
    return result


def compare_sampler(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    natural = aggregate["natural"]
    balanced = aggregate["iqn_balanced"]
    n_overall = natural["test"]["overall"]
    b_overall = balanced["test"]["overall"]
    n_early = natural["test"]["events"]["early_recovery"]
    b_early = balanced["test"]["events"]["early_recovery"]
    n_post = natural["test"]["classes"]["post_capture_real"]
    b_post = balanced["test"]["classes"]["post_capture_real"]
    n_pursuit_train = natural["train"]["classes"]["pursuing"]["rmse"]["mean"]
    b_pursuit_train = balanced["train"]["classes"]["pursuing"]["rmse"]["mean"]
    n_pursuit_test = natural["test"]["classes"]["pursuing"]["rmse"]["mean"]
    b_pursuit_test = balanced["test"]["classes"]["pursuing"]["rmse"]["mean"]
    n_ece = n_overall["decile_calibration_error"]["mean"]
    b_ece = b_overall["decile_calibration_error"]["mean"]
    n_overall_rmse = n_overall["rmse"]["mean"]
    b_overall_rmse = b_overall["rmse"]["mean"]
    overfit = bool(
        b_pursuit_train <= 0.90 * n_pursuit_train
        and (b_pursuit_test >= 1.10 * n_pursuit_test or b_overall_rmse >= 1.05 * n_overall_rmse)
    )
    b_early_rmse = b_early["rmse"]["mean"]
    n_early_rmse = n_early["rmse"]["mean"]
    b_post_rmse = b_post["rmse"]["mean"]
    n_post_rmse = n_post["rmse"]["mean"]
    balanced_early_better = b_early_rmse <= 0.90 * n_early_rmse
    balanced_post_better = b_post_rmse <= 0.90 * n_post_rmse
    overall_not_worse = b_overall_rmse <= 1.05 * n_overall_rmse
    calibration_not_worse = b_ece <= 1.10 * n_ece
    if overfit:
        classification = "BALANCED_OVERFIT"
    elif balanced_early_better and balanced_post_better and overall_not_worse and calibration_not_worse:
        classification = "BALANCED_PREFERRED"
    elif abs(b_overall_rmse - n_overall_rmse) <= 0.05 * n_overall_rmse and abs(b_early_rmse - n_early_rmse) <= 0.05 * n_early_rmse:
        classification = "SAMPLER_NEUTRAL"
    else:
        classification = "NATURAL_PREFERRED"
    return {
        "classification": classification,
        "balanced_overfit_check": overfit,
        "balanced_pursuing_train_rmse": b_pursuit_train,
        "natural_pursuing_train_rmse": n_pursuit_train,
        "balanced_pursuing_test_rmse": b_pursuit_test,
        "natural_pursuing_test_rmse": n_pursuit_test,
        "balanced_early_recovery_rmse": b_early_rmse,
        "natural_early_recovery_rmse": n_early_rmse,
        "balanced_post_capture_rmse": b_post_rmse,
        "natural_post_capture_rmse": n_post_rmse,
        "balanced_overall_rmse": b_overall_rmse,
        "natural_overall_rmse": n_overall_rmse,
        "balanced_overall_ece": b_ece,
        "natural_overall_ece": n_ece,
        "balanced_early_better": balanced_early_better,
        "balanced_post_better": balanced_post_better,
        "overall_not_worse": overall_not_worse,
        "calibration_not_worse": calibration_not_worse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--rescore", action="store_true", help="rescore completed checkpoints without fitting")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = make_config()
    before_disk = shutil.disk_usage(ROOT)
    bank, manifest = load_bank()
    device = torch.device(args.device)
    rows = AC3Rows(bank, device)
    train_indices = rows.indices(0)
    mean = float(rows.target[train_indices].mean())
    scale = max(float(rows.target[train_indices].std()), 1e-6)
    class_counts_train = {name: int((rows.semantic_class[train_indices] == class_id).sum()) for class_id, name in enumerate(CLASS_NAMES)}
    conditions = {}
    all_runs = []
    started = time.monotonic()
    for kind in CRITICS:
        for condition in CONDITIONS:
            key = f"{kind}_{condition}"
            runs = []
            for seed in SEEDS:
                run = rescore_one(kind, condition, seed, config, rows, output, device, mean, scale) if args.rescore else fit_one(kind, condition, seed, config, rows, output, device, mean, scale)
                runs.append(run)
                all_runs.append(run)
                atomic_json(output / "progress.json", {"status": "fitting", "completed": len(all_runs), "total": 8, "current": key, "seed": seed, "elapsed_seconds": time.monotonic() - started})
            conditions[key] = {"kind": kind, "condition": condition, "runs": runs, "aggregate": aggregate_condition(runs)}
    aggregate_by_kind = {
        kind: {
            condition: conditions[f"{kind}_{condition}"]["aggregate"]
            for condition in CONDITIONS
        }
        for kind in CRITICS
    }
    classifications = {kind: compare_sampler(aggregate_by_kind[kind]) for kind in CRITICS}
    if classifications["LQ"]["classification"] == classifications["NQ"]["classification"]:
        if classifications["LQ"]["classification"] == "NATURAL_PREFERRED":
            recommendation = "USE_NATURAL"
        elif classifications["LQ"]["classification"] == "BALANCED_PREFERRED":
            recommendation = "USE_IQN_BALANCED"
        else:
            recommendation = "NO_CLEAR_WINNER"
    else:
        recommendation = "NO_CLEAR_WINNER"
    after_disk = shutil.disk_usage(ROOT)
    manifest_out = {
        "schema": "ac3a-critic-sampler-audit-v1",
        "status": "complete",
        "scope": "LQ_NQ_only",
        "bank": {
            "path": str(BANK_PATH),
            "sha256": sha256_file(BANK_PATH),
            "manifest_sha256": manifest["bank"]["sha256"],
            "transitions": int(len(bank["episode_id"])),
            "active_agent_rows": int(bank["active_mask"].sum()),
            "split_episodes": {name: int(len(np.unique(bank["episode_id"][bank["split_id"] == split_id]))) for split_id, name in enumerate(SPLITS)},
        },
        "contract": {
            "architecture": "historical A1 LocalQ/NeighborQ, LegacyVorAdjFeatureBackbone",
            "hidden_dim": 256,
            "num_layers": 4,
            "num_heads": 8,
            "dropout": 0.0,
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "weight_decay": 1e-6,
            "grad_clip": 0.5,
            "batch_size": 256,
            "updates": 1200,
            "validation_every": 100,
            "model_selection": "natural validation split MC-return RMSE, same criterion for all conditions",
            "normalization": {"train_mean": mean, "train_std": scale, "train_only": True, "frozen": True},
            "target": "individual realized full-episode MC return, gamma=0.99, no bootstrap",
            "seeds": list(SEEDS),
        },
        "sampler_contract": {
            "natural": "uniform with replacement over all train active-agent rows",
            "iqn_balanced": {"class_order": list(CLASS_NAMES), "quota_per_128_row_block": BALANCED_QUOTA.tolist(), "blocks_per_a1_batch": 2, "effective_quota_per_batch": (BALANCED_QUOTA * 2).tolist(), "batch_fraction": (BALANCED_QUOTA / BALANCED_QUOTA.sum()).tolist(), "with_replacement": True, "ring3_or_capture_quota": False},
            "train_semantic_class_rows": class_counts_train,
        },
        "conditions": conditions,
        "classifications": classifications,
        "unified_recommendation": recommendation,
        "forbidden_executed": {"CQ": False, "V": False, "bootstrap": False, "GAE": False, "PPO": False, "actor_updates": 0, "action_ranking_main_experiment": False},
        "storage": {"disk_free_bytes_before": int(before_disk.free), "disk_free_bytes_after": int(after_disk.free), "output_bytes": int(sum(path.stat().st_size for path in output.rglob("*") if path.is_file()))},
    }
    atomic_json(output / "summary.json", manifest_out)
    atomic_json(output / "progress.json", {"status": "complete", "conditions": list(conditions), "unified_recommendation": recommendation, "elapsed_seconds": time.monotonic() - started})
    print(json.dumps({"status": "complete", "bank_sha256": manifest_out["bank"]["sha256"], "conditions": list(conditions), "classifications": classifications, "unified_recommendation": recommendation, "output_bytes": manifest_out["storage"]["output_bytes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
