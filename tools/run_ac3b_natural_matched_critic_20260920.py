#!/usr/bin/env python3
"""AC-3B: matched Natural-sampler CQ/V MC critic comparison.

LQ and NQ are intentionally not fitted here: their Natural results are read
from the completed AC-3A artifact.  This runner fits only the historical A1
central Q and action-free centralized V on the same canonical-BC bank, with
the same seeds, optimizer, updates, validation selector, and individual
full-episode MC target.  It contains no bootstrap, counterfactual action
ranking, actor update, GAE, or PPO path.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path("/home/yjq/rl/CoCap1/cocap-voradj-critic-audit")
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from cocap_voradj.models.critic_identifiability import parameter_count
from tools.run_ac3a_critic_sampler_audit_20260920 import (
    AC3Rows,
    CLASS_NAMES,
    EVENT_NAMES,
    evaluate_kind,
    make_config,
    mean_spread,
    sha256_file,
)
from tools.run_critic_identifiability_audit_20260917 import (
    forward,
    make_model,
    to_tensor,
)


OUT = ROOT / "artifacts/2026-09-20_ac3b"
BANK_PATH = ROOT / "artifacts/2026-09-20_ac2b/canonical_bc_critic_bank_v2.npz"
BANK_MANIFEST = ROOT / "artifacts/2026-09-20_ac2b/bank_manifest.json"
AC3A_SUMMARY = ROOT / "artifacts/2026-09-20_ac3a/summary.json"
BANK_SHA256 = "07e08d17ae6283d785e0d2e00307f330b13a88dcd0f8ce83f92f0d451ad7808a"
SEEDS = (2026091711, 2026091712)
CRITICS = ("CQ", "V")
SPLITS = ("train", "validation", "test")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def load_bank() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    actual = sha256_file(BANK_PATH)
    if actual != BANK_SHA256:
        raise RuntimeError(f"AC3B_BANK_IDENTITY_FAIL: {actual} != {BANK_SHA256}")
    manifest = json.loads(BANK_MANIFEST.read_text())
    if manifest.get("status") != "formal_bank_complete" or manifest["bank"]["sha256"] != actual:
        raise RuntimeError("AC3B_BANK_MANIFEST_MISMATCH")
    with np.load(BANK_PATH, allow_pickle=False) as source:
        bank = {name: source[name] for name in source.files}
    if sorted({int(value) for value in bank["split_id"]}) != [0, 1, 2]:
        raise RuntimeError("AC3B_SPLIT_CONTRACT_FAIL")
    required = {"neighbor_action_index", "neighbor_action_aw", "transition_event_flags", "replay_semantic_class"}
    missing = sorted(required.difference(bank))
    if missing:
        raise RuntimeError(f"AC3B_SCHEMA_MISSING: {missing}")
    return bank, manifest


def natural_sampling_stats(rows: AC3Rows, seed: int, updates: int, batch_size: int) -> dict[str, Any]:
    train_indices = rows.indices(0)
    rng = np.random.default_rng(seed)
    draw_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    seen_rows = [set() for _ in CLASS_NAMES]
    for _ in range(updates):
        chosen = rng.choice(train_indices, size=batch_size, replace=True)
        labels = rows.semantic_class[chosen]
        draw_counts += np.bincount(labels, minlength=len(CLASS_NAMES))
        for class_id in range(len(CLASS_NAMES)):
            seen_rows[class_id].update(chosen[labels == class_id].tolist())
    available = {
        name: int((rows.semantic_class[train_indices] == class_id).sum())
        for class_id, name in enumerate(CLASS_NAMES)
    }
    return {
        "draw_count_total": int(draw_counts.sum()),
        "draw_count_by_class": {name: int(draw_counts[i]) for i, name in enumerate(CLASS_NAMES)},
        "draw_ratio_by_class": {name: float(draw_counts[i] / draw_counts.sum()) for i, name in enumerate(CLASS_NAMES)},
        "unique_count_by_class": {name: int(len(seen_rows[i])) for i, name in enumerate(CLASS_NAMES)},
        "reuse_ratio_draws_per_unique_by_class": {
            name: float(draw_counts[i] / max(len(seen_rows[i]), 1)) for i, name in enumerate(CLASS_NAMES)
        },
        "train_available_rows_by_class": available,
    }


def fit_one(
    kind: str,
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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    train_indices = rows.indices(0)
    validation_indices = rows.indices(1)
    test_indices = rows.indices(2)
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    started = time.monotonic()
    for update in range(1, int(train_cfg["updates"]) + 1):
        chosen = rng.choice(train_indices, size=int(train_cfg["batch_size"]), replace=True)
        batch = rows.batch(chosen)
        target = to_tensor((rows.target[chosen] - mean) / scale, device)
        model.train()
        loss = torch.mean((forward(kind, model, batch) - target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"])))
        optimizer.step()
        if update % int(train_cfg["validation_every"]) == 0 or update == int(train_cfg["updates"]):
            _, validation = evaluate_kind(
                kind, rows, model, validation_indices, int(train_cfg["batch_size"]), mean, scale
            )
            score = float(validation["overall"]["rmse"])
            history.append({
                "update": update,
                "train_loss_normalized": float(loss.detach().cpu()),
                "gradient_norm": gradient,
                "validation_overall": validation["overall"],
            })
            if best is None or score < float(best["validation_rmse"]):
                best = {
                    "validation_rmse": score,
                    "update": update,
                    "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                }
    if best is None:
        raise RuntimeError(f"AC3B_NO_VALIDATION_CHECKPOINT_{kind}_{seed}")
    best_state = best.pop("state")
    model.load_state_dict(best_state)
    _, train_score = evaluate_kind(kind, rows, model, train_indices, int(train_cfg["batch_size"]), mean, scale)
    _, validation_score = evaluate_kind(kind, rows, model, validation_indices, int(train_cfg["batch_size"]), mean, scale)
    _, test_score = evaluate_kind(kind, rows, model, test_indices, int(train_cfg["batch_size"]), mean, scale)
    checkpoint = output / "checkpoints" / f"{kind.lower()}_natural_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "cocap-ac3b-natural-matched-v1",
        "kind": kind,
        "condition": "natural",
        "seed": seed,
        "state_dict": model.state_dict(),
        "normalization": {"train_mean": mean, "train_std": scale},
        "best_validation": best,
        "architecture_contract": config["training"],
        "target_contract": "individual full-episode MC return, gamma=0.99, no bootstrap",
    }, checkpoint)
    result = {
        "kind": kind,
        "condition": "natural",
        "seed": seed,
        "parameter_count": parameter_count(model),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_validation": best,
        "history": history,
        "train": train_score,
        "validation": validation_score,
        "test": test_score,
        "sampling": natural_sampling_stats(rows, seed, int(train_cfg["updates"]), int(train_cfg["batch_size"])),
        "optimizer_steps": int(train_cfg["updates"]),
        "elapsed_seconds": time.monotonic() - started,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def compact_metrics(aggregate: Mapping[str, Any], split: str = "test") -> dict[str, Any]:
    sections = [
        "rmse", "mae", "explained_variance", "pearson", "spearman",
        "calibration_slope_target_on_prediction", "calibration_intercept",
        "decile_calibration_error",
    ]
    result: dict[str, Any] = {
        "overall": {key: aggregate[split]["overall"][key] for key in sections},
        "classes": {
            name: {key: aggregate[split]["classes"][name][key] for key in sections + ["rows"]}
            for name in CLASS_NAMES
        },
        "events": {
            name: {key: aggregate[split]["events"][name][key] for key in sections + ["rows"]}
            for name in EVENT_NAMES
        },
    }
    return result


def compact_ac3a_condition(summary: Mapping[str, Any], key: str) -> dict[str, Any]:
    condition = summary["conditions"][key]
    aggregate = condition["aggregate"]
    run0 = condition["runs"][0]
    return {
        "kind": condition["kind"],
        "condition": condition["condition"],
        "parameter_count": int(run0["parameter_count"]),
        "source": str(AC3A_SUMMARY),
        "source_sha256": sha256_file(AC3A_SUMMARY),
        "test": compact_metrics(aggregate),
        "sampling": aggregate["sampling"],
        "seeds": aggregate["seeds"],
    }


def metric_mean(metrics: Mapping[str, Any], name: str) -> float | None:
    value = metrics["overall"][name]["mean"]
    return None if value is None else float(value)


def relative_deltas(consolidated: Mapping[str, Any]) -> dict[str, Any]:
    base = consolidated["LQ"]["test"]
    result = {}
    for kind, value in consolidated.items():
        deltas: dict[str, Any] = {}
        for key in ("rmse", "ev", "spearman", "ece"):
            if key == "ev":
                candidate = metric_mean(value["test"], "explained_variance")
                reference = metric_mean(base, "explained_variance")
                deltas["delta_ev"] = None if candidate is None or reference is None else candidate - reference
            elif key == "spearman":
                candidate = metric_mean(value["test"], "spearman")
                reference = metric_mean(base, "spearman")
                deltas["delta_spearman"] = None if candidate is None or reference is None else candidate - reference
            elif key == "ece":
                candidate = metric_mean(value["test"], "decile_calibration_error")
                reference = metric_mean(base, "decile_calibration_error")
                deltas["delta_ece"] = None if candidate is None or reference is None else candidate - reference
            else:
                candidate = metric_mean(value["test"], "rmse")
                reference = metric_mean(base, "rmse")
                deltas["delta_rmse_percent"] = None if candidate is None or reference is None else 100.0 * (candidate - reference) / reference
        result[kind] = deltas
    return result


def descriptive_classification(consolidated: Mapping[str, Any]) -> list[str]:
    base = consolidated["LQ"]["test"]
    base_rmse = metric_mean(base, "rmse")
    base_ev = metric_mean(base, "explained_variance")
    labels: list[str] = []
    for kind in ("NQ", "CQ", "V"):
        value = consolidated[kind]["test"]
        rmse = metric_mean(value, "rmse")
        ev = metric_mean(value, "explained_variance")
        if rmse is not None and ev is not None and base_rmse is not None and base_ev is not None:
            if rmse < base_rmse and ev > base_ev:
                labels.append("CENTRAL_CONTEXT_HELPFUL" if kind in {"CQ", "V"} else "NEIGHBOR_CONTEXT_HELPFUL")
    if not labels:
        labels.append("LOCAL_STILL_COMPETITIVE")
    phase_best = []
    for phase in ("early_recovery", "late_recovery", "pure_coverage", "ring2", "ring3", "capture_transition"):
        values = {
            kind: consolidated[kind]["test"]["events"][phase]["rmse"]["mean"]
            for kind in ("LQ", "NQ", "CQ", "V")
        }
        usable = {kind: value for kind, value in values.items() if value is not None}
        if usable:
            phase_best.append(min(usable, key=usable.get))
    if len(set(phase_best)) > 1:
        labels.append("MIXED_PHASE_DEPENDENT")
    return labels


def smoke(kind: str, rows: AC3Rows, config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    model = make_model(kind, config, device)
    indices = rows.indices(2)[: min(4, len(rows.indices(2)))]
    with torch.no_grad():
        output = forward(kind, model, rows.batch(indices))
    finite = bool(torch.isfinite(output).all().item())
    shape = list(output.shape)
    parameters = int(parameter_count(model))
    del model, output
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    if not finite:
        raise RuntimeError(f"AC3B_FORWARD_NONFINITE_{kind}")
    return {"kind": kind, "finite": finite, "output_shape": shape, "parameter_count": parameters, "optimizer_steps": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = make_config()
    before_disk = shutil.disk_usage(ROOT)
    bank, bank_manifest = load_bank()
    device = torch.device(args.device)
    rows = AC3Rows(bank, device)
    train_indices = rows.indices(0)
    mean = float(rows.target[train_indices].mean())
    scale = max(float(rows.target[train_indices].std()), 1e-6)
    smoke_results = {kind: smoke(kind, rows, config, device) for kind in CRITICS}
    if args.smoke_only:
        print(json.dumps({"status": "smoke_pass", "bank_sha256": BANK_SHA256, "smoke": smoke_results}, ensure_ascii=False))
        return
    if not AC3A_SUMMARY.exists():
        raise RuntimeError("AC3B_AC3A_SUMMARY_MISSING")
    ac3a = json.loads(AC3A_SUMMARY.read_text())
    if ac3a.get("scope") != "LQ_NQ_only":
        raise RuntimeError("AC3B_AC3A_SCOPE_MISMATCH")
    if list(ac3a["contract"]["seeds"]) != list(SEEDS):
        raise RuntimeError("AC3B_SEED_CONTRACT_MISMATCH")
    started = time.monotonic()
    new_conditions: dict[str, Any] = {}
    for kind in CRITICS:
        runs = []
        for seed in SEEDS:
            run = fit_one(kind, int(seed), config, rows, output, device, mean, scale)
            runs.append(run)
            atomic_json(output / "progress.json", {
                "status": "fitting",
                "completed": sum(len(value.get("runs", [])) for value in new_conditions.values()) + len(runs),
                "total": len(CRITICS) * len(SEEDS),
                "current": kind,
                "seed": int(seed),
                "elapsed_seconds": time.monotonic() - started,
            })
        from tools.run_ac3a_critic_sampler_audit_20260920 import aggregate_condition
        new_conditions[f"{kind}_natural"] = {
            "kind": kind,
            "condition": "natural",
            "runs": runs,
            "aggregate": aggregate_condition(runs),
        }
    compact: dict[str, Any] = {
        "LQ": compact_ac3a_condition(ac3a, "LQ_natural"),
        "NQ": compact_ac3a_condition(ac3a, "NQ_natural"),
        "CQ": {
            "kind": "CQ", "condition": "natural",
            "parameter_count": int(new_conditions["CQ_natural"]["runs"][0]["parameter_count"]),
            "test": compact_metrics(new_conditions["CQ_natural"]["aggregate"]),
            "seeds": list(SEEDS),
        },
        "V": {
            "kind": "V", "condition": "natural",
            "parameter_count": int(new_conditions["V_natural"]["runs"][0]["parameter_count"]),
            "test": compact_metrics(new_conditions["V_natural"]["aggregate"]),
            "seeds": list(SEEDS),
        },
    }
    deltas = relative_deltas(compact)
    classification = descriptive_classification(compact)
    after_disk = shutil.disk_usage(ROOT)
    summary = {
        "schema": "ac3b-strong-bc-critic-architecture-comparison-v1",
        "status": "complete",
        "scope": "CQ_V_natural_only_with_AC3A_LQ_NQ_reference",
        "git_head_at_run": git_head(),
        "bank": {
            "path": str(BANK_PATH),
            "sha256": sha256_file(BANK_PATH),
            "manifest_status": bank_manifest["status"],
            "transitions": int(len(bank["episode_id"])),
            "active_agent_rows": int(bank["active_mask"].sum()),
            "split_episodes": {
                name: int(len(np.unique(bank["episode_id"][bank["split_id"] == split_id])))
                for split_id, name in enumerate(SPLITS)
            },
        },
        "contract": {
            "architecture_source": "historical A1 CentralAttentionCritic (CQ) and CentralValueNetwork (V)",
            "cq_input": "central global entity state + full joint AW9 continuous actions + focal agent index selection",
            "v_input": "central global entity state; no actions",
            "output_semantics": "one value per active pursuer; focal active-agent row is selected by agent index",
            "parameter_counts": {kind: int(smoke_results[kind]["parameter_count"]) for kind in CRITICS},
            "hidden_dim": int(config["training"]["hidden_dim"]),
            "num_heads": int(config["training"]["num_heads"]),
            "num_layers": int(config["training"]["num_layers"]),
            "dropout": float(config["training"]["dropout"]),
            "optimizer": "Adam",
            "learning_rate": float(config["training"]["learning_rate"]),
            "weight_decay": float(config["training"]["weight_decay"]),
            "grad_clip": float(config["training"]["grad_clip"]),
            "batch_size": int(config["training"]["batch_size"]),
            "updates": int(config["training"]["updates"]),
            "validation_every": int(config["training"]["validation_every"]),
            "model_selection": "validation MC-return RMSE only",
            "normalization": {"train_mean": mean, "train_std": scale, "train_only": True, "frozen": True},
            "target": "individual full-episode MC return, gamma=0.99, no bootstrap",
            "seeds": list(SEEDS),
            "sampler": "uniform with replacement over all train active-agent rows",
        },
        "smoke": smoke_results,
        "new_conditions": new_conditions,
        "ac3a_reference": {"path": str(AC3A_SUMMARY), "sha256": sha256_file(AC3A_SUMMARY), "scope": ac3a["scope"]},
        "consolidated_test": compact,
        "relative_to_lq": deltas,
        "classification": classification,
        "forbidden_executed": {
            "LQ_retrained": False, "NQ_retrained": False, "bootstrap": False,
            "GAE": False, "PPO": False, "counterfactual_action_ranking": False,
            "actor_updates": 0,
        },
        "storage": {
            "disk_free_bytes_before": int(before_disk.free),
            "disk_free_bytes_after": int(after_disk.free),
            "estimated_checkpoint_bytes": int(sum(smoke_results[k]["parameter_count"] for k in CRITICS) * 4 * len(SEEDS)),
            "output_bytes_after_summary": int(sum(path.stat().st_size for path in output.rglob("*") if path.is_file())),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "progress.json", {
        "status": "complete",
        "conditions": list(new_conditions),
        "classification": classification,
        "elapsed_seconds": summary["elapsed_seconds"],
    })
    print(json.dumps({
        "status": "complete",
        "bank_sha256": BANK_SHA256,
        "classification": classification,
        "disk_free_before": before_disk.free,
        "disk_free_after": after_disk.free,
        "output_bytes": summary["storage"]["output_bytes_after_summary"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
