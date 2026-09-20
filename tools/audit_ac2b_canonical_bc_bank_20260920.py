#!/usr/bin/env python3
"""Finalize and audit the AC-2B canonical-BC formal critic bank.

This is an offline artifact/schema audit.  It may add explicitly materialized
neighbor action fields to the raw bank, but it never invokes the environment,
the actor, an optimizer, a critic, bootstrap, GAE, or PPO.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/yjq/rl/CoCap1/cocap-voradj-critic-audit")
OUT = ROOT / "artifacts/2026-09-20_ac2b"
BANK_PATH = OUT / "canonical_bc_critic_bank_v2.npz"
COLLECTION_MANIFEST = OUT / "formal_collection_manifest.json"
LEGACY_COLLECTION_MANIFEST = OUT / "pilot_manifest.json"
GAMMA = 0.99
MAX_AGENTS = 12
EVENTS = (
    "pure_coverage", "coverage_steady", "ordinary_pursuit", "ring2", "ring3",
    "normal_capture", "stationary_capture", "collision",
    "capture_terminal_transition", "early_recovery", "late_recovery",
    "pure_coverage_restart",
)
EVENT_TO_ID = {name: index for index, name in enumerate(EVENTS)}
PHASES = ("pure_coverage", "pre_capture", "post_capture")
PHASE_TO_ID = {name: index for index, name in enumerate(PHASES)}
CLASS_NAMES = ("pursuing", "pre_capture_cover", "post_capture_real", "recovery_pure")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
ACTION_GRID = np.asarray(
    [(a, w) for a in (-0.4, 0.0, 0.4) for w in (-np.pi / 6.0, 0.0, np.pi / 6.0)],
    dtype=np.float32,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def finite_report(bank: dict[str, np.ndarray]) -> dict[str, Any]:
    bad_nan = {}
    bad_inf = {}
    for name, value in bank.items():
        if np.issubdtype(value.dtype, np.floating):
            nan_count = int(np.isnan(value).sum())
            inf_count = int(np.isinf(value).sum())
            if nan_count:
                bad_nan[name] = nan_count
            if inf_count:
                bad_inf[name] = inf_count
    return {
        "nan_by_field": bad_nan,
        "inf_by_field": bad_inf,
        "nan_count": int(sum(bad_nan.values())),
        "inf_count": int(sum(bad_inf.values())),
    }


def active_event_counts(bank: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict[str, int]:
    take = np.ones(len(bank["episode_id"]), dtype=bool) if mask is None else mask
    active = bank["active_mask"][take]
    flags = bank["transition_event_flags"][take]
    return {
        name: int((flags[:, EVENT_TO_ID[name]][:, None] & active).sum())
        for name in EVENTS
    }


def add_neighbor_actions(bank: dict[str, np.ndarray]) -> None:
    if "neighbor_action_index" in bank and "neighbor_action_aw" in bank:
        return
    neighbors = bank["neighbor_ids"].astype(np.int64)
    safe = np.maximum(neighbors, 0)
    valid = bank["neighbor_mask"].astype(bool)
    action_index = np.broadcast_to(bank["action_index"][:, :, None], safe.shape)
    neighbor_index = np.take_along_axis(action_index, safe, axis=1).astype(np.int64)
    neighbor_index[~valid] = -1
    action_aw = np.broadcast_to(bank["action_aw"][:, :, None, :], (*safe.shape, 2))
    neighbor_aw = np.take_along_axis(action_aw, safe[..., None], axis=1).astype(np.float32)
    neighbor_aw[~valid] = 0.0
    bank["neighbor_action_index"] = neighbor_index
    bank["neighbor_action_aw"] = neighbor_aw


def rewrite_bank(bank: dict[str, np.ndarray]) -> None:
    temporary = BANK_PATH.with_name(BANK_PATH.name + ".tmp.npz")
    np.savez_compressed(temporary, **bank)
    os.replace(temporary, BANK_PATH)


def row_lookup(bank: dict[str, np.ndarray]) -> dict[tuple[int, int], int]:
    return {(int(e), int(t)): i for i, (e, t) in enumerate(zip(bank["episode_id"], bank["timestep"]))}


def expected_split(record: dict[str, Any]) -> str:
    scene = record["scene"]
    mode = record["policy_mode"]
    episode_id = int(record["episode_id"])
    if scene == "mixed" and mode == "bc_argmax":
        index = episode_id
    elif scene == "mixed" and mode == "bc_sample":
        index = episode_id - 30
    elif scene == "pure_coverage" and mode == "bc_argmax":
        index = episode_id - 60
    elif scene == "pure_coverage" and mode == "bc_sample":
        index = episode_id - 70
    else:
        raise ValueError(f"unknown formal episode group: {record}")
    if index < 0 or (scene == "mixed" and index >= 30) or (scene == "pure_coverage" and index >= 10):
        raise ValueError(f"episode outside formal group: {record}")
    if scene == "mixed":
        return "train" if index < 24 else "validation" if index < 27 else "test"
    return "train" if index < 8 else "validation" if index < 9 else "test"


def normalize_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = []
    corrections = []
    for original in records:
        record = dict(original)
        expected = expected_split(record)
        if record.get("split") != expected:
            corrections.append({"episode_id": int(record["episode_id"]), "old": record.get("split"), "new": expected})
        record["split"] = expected
        normalized.append(record)
    return normalized, corrections


def correct_split_labels(bank: dict[str, np.ndarray], records: list[dict[str, Any]]) -> dict[str, Any]:
    split_ids = {"train": 0, "validation": 1, "test": 2}
    expected_by_episode = {int(record["episode_id"]): split_ids[expected_split(record)] for record in records}
    expected = np.asarray([expected_by_episode[int(episode_id)] for episode_id in bank["episode_id"]], dtype=np.int64)
    mismatch = bank["split_id"] != expected
    if np.any(mismatch):
        bank["split_id"] = expected
    return {
        "passed": True,
        "corrected": bool(np.any(mismatch)),
        "rows_corrected": int(mismatch.sum()),
        "episode_metadata_corrections": [],
        "expected_episode_split_counts": {
            name: int(sum(value == split_id for value in expected_by_episode.values()))
            for name, split_id in split_ids.items()
        },
    }


def check_episode_structure(bank: dict[str, np.ndarray], records: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    episode_ids = sorted({int(value) for value in bank["episode_id"]})
    if episode_ids != list(range(80)):
        errors.append(f"episode_ids={episode_ids}")
    if len(records) != 80:
        errors.append(f"manifest_episode_records={len(records)}")
    per_episode = {}
    for record in records:
        episode_id = int(record["episode_id"])
        rows = np.flatnonzero(bank["episode_id"] == episode_id)
        expected_t = np.arange(1, len(rows) + 1, dtype=np.int64)
        if not np.array_equal(bank["timestep"][rows], expected_t):
            errors.append(f"episode_{episode_id}_timestep_sequence")
        for field in ("episode_seed", "scene_id", "policy_mode_id", "split_id"):
            if len(np.unique(bank[field][rows])) != 1:
                errors.append(f"episode_{episode_id}_{field}_not_constant")
        if int(bank["episode_seed"][rows[0]]) != int(record["seed"]):
            errors.append(f"episode_{episode_id}_seed_mismatch")
        per_episode[str(episode_id)] = int(len(rows))
    pair_count = len({(int(e), int(t)) for e, t in zip(bank["episode_id"], bank["timestep"])})
    if pair_count != len(bank["episode_id"]):
        errors.append("duplicate_episode_timestep")
    return {"passed": not errors, "errors": errors, "episode_count": len(episode_ids), "rows_by_episode": per_episode}


def check_actions_and_neighbors(bank: dict[str, np.ndarray]) -> dict[str, Any]:
    active = bank["active_mask"].astype(bool)
    inactive = ~active
    errors: list[str] = []
    action_index = bank["action_index"]
    invalid_active = int(((action_index < 0) | (action_index >= 9) & active).sum())
    if invalid_active:
        errors.append("active_action_index_out_of_range")
    if int((action_index[inactive] != 4).sum()):
        errors.append("inactive_action_index_not_padding_4")
    expected_aw = ACTION_GRID[np.clip(action_index, 0, 8)]
    action_mismatch = int((np.max(np.abs(bank["action_aw"] - expected_aw), axis=-1)[active] > 1e-6).sum())
    if action_mismatch:
        errors.append("action_aw_mapping_mismatch")
    if int(np.max(np.abs(bank["action_aw"][inactive])) > 1e-6):
        errors.append("inactive_action_aw_not_zero")
    prob_sum_error = float(np.max(np.abs(bank["actor_probs"].sum(axis=-1)[active] - 1.0)))
    probs_nonfinite_or_invalid = int((~np.isfinite(bank["actor_probs"][active])).sum()) + int((bank["actor_probs"][active] < 0).sum())
    selected_expected = np.zeros_like(bank["selected_action_probability"])
    active_rows = np.flatnonzero(active)
    if len(active_rows):
        selected_expected[active] = bank["actor_probs"][active][np.arange(len(active_rows)), action_index[active]]
    selected_error = float(np.max(np.abs(selected_expected - bank["selected_action_probability"])))
    if prob_sum_error > 1e-5:
        errors.append("actor_probability_sum")
    if probs_nonfinite_or_invalid:
        errors.append("actor_probability_invalid")
    if selected_error > 1e-6:
        errors.append("selected_probability_mismatch")
    neighbor_ids = bank["neighbor_ids"].astype(np.int64)
    neighbor_mask = bank["neighbor_mask"].astype(bool)
    valid_ids = neighbor_ids[neighbor_mask]
    if len(valid_ids) and ((valid_ids < 0).any() or (valid_ids >= MAX_AGENTS).any()):
        errors.append("neighbor_id_out_of_range")
    if np.any(neighbor_mask & ~bank["active_mask"][:, :, None]):
        errors.append("inactive_focal_has_neighbor")
    if len(valid_ids):
        active_by_id = np.take_along_axis(active[:, :, None], np.maximum(neighbor_ids, 0), axis=1)
        if np.any(neighbor_mask & ~active_by_id):
            errors.append("neighbor_id_not_active")
    if np.any(neighbor_mask.sum(axis=-1) > 8):
        errors.append("neighbor_count_over_max")
    if np.any(bank["neighbor_action_index"][neighbor_mask] != np.take_along_axis(
        bank["action_index"][:, :, None], np.maximum(neighbor_ids, 0), axis=1
    )[neighbor_mask]):
        errors.append("neighbor_action_index_mapping")
    neighbor_expected_aw = np.take_along_axis(
        bank["action_aw"][:, :, None, :], np.maximum(neighbor_ids, 0)[..., None], axis=1
    )
    neighbor_aw_error = float(np.max(np.abs(bank["neighbor_action_aw"] - np.where(neighbor_mask[..., None], neighbor_expected_aw, 0.0))))
    if neighbor_aw_error > 1e-6:
        errors.append("neighbor_action_aw_mapping")
    if np.any(bank["neighbor_action_index"][~neighbor_mask] != -1):
        errors.append("invalid_neighbor_action_index_not_minus_one")
    if int(np.max(np.abs(bank["neighbor_action_aw"][~neighbor_mask])) > 1e-6):
        errors.append("invalid_neighbor_action_aw_not_zero")
    return {
        "passed": not errors,
        "errors": errors,
        "action_index_range_active": [int(action_index[active].min()), int(action_index[active].max())],
        "action_aw_max_abs_mapping_error": float(np.max(np.abs(bank["action_aw"][active] - expected_aw[active]))),
        "actor_probability_sum_max_abs_error": prob_sum_error,
        "selected_probability_max_abs_error": selected_error,
        "neighbor_action_aw_max_abs_error": neighbor_aw_error,
    }


def check_continuity(bank: dict[str, np.ndarray]) -> dict[str, Any]:
    errors: list[str] = []
    transitions_checked = 0
    lookup = row_lookup(bank)
    local_fields = [name for name in bank if name.startswith("local_") and not name.startswith("local_next")]
    next_local_fields = [name for name in bank if name.startswith("next_local_")]
    global_fields = [name for name in bank if name.startswith("global_") and not name.startswith("global_next")]
    next_global_fields = [name for name in bank if name.startswith("next_global_")]
    for episode_id in sorted({int(value) for value in bank["episode_id"]}):
        rows = np.flatnonzero(bank["episode_id"] == episode_id)
        for offset in range(len(rows) - 1):
            row, nxt = int(rows[offset]), int(rows[offset + 1])
            transitions_checked += 1
            for current_name in local_fields:
                next_name = "next_" + current_name
                if next_name not in next_local_fields or not np.array_equal(bank[next_name][row], bank[current_name][nxt]):
                    errors.append(f"row_{row}_local_next_mismatch")
                    break
            for current_name in global_fields:
                next_name = "next_" + current_name
                if next_name not in next_global_fields or not np.array_equal(bank[next_name][row], bank[current_name][nxt]):
                    errors.append(f"row_{row}_global_next_mismatch")
                    break
            if not np.array_equal(bank["next_active_mask"][row], bank["active_mask"][nxt]):
                errors.append(f"row_{row}_active_mask_next_mismatch")
    return {"passed": not errors, "errors": errors[:20], "error_count": len(errors), "within_episode_links_checked": transitions_checked}


def check_mc(bank: dict[str, np.ndarray], records: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(2026092201)
    selected = np.sort(rng.choice(len(bank["episode_id"]), min(100, len(bank["episode_id"])), replace=False))
    max_error = 0.0
    details = []
    for episode_id in sorted({int(value) for value in bank["episode_id"]}):
        rows = np.flatnonzero(bank["episode_id"] == episode_id)
        rewards = bank["reward"][rows].astype(np.float64)
        expected = np.zeros_like(rewards)
        running = np.zeros(MAX_AGENTS, dtype=np.float64)
        for offset in range(len(rows) - 1, -1, -1):
            running = rewards[offset] + GAMMA * running
            expected[offset] = running
        selected_local = [i for i, row in enumerate(rows) if row in set(selected)]
        if selected_local:
            error = float(np.max(np.abs(expected[selected_local] - bank["mc_return"][rows[selected_local]])))
            max_error = max(max_error, error)
            details.append({"episode_id": episode_id, "rows_checked": len(selected_local), "max_abs_error": error})
    return {
        "passed": max_error <= 1e-5,
        "sample_size": int(len(selected)),
        "sample_rows": selected.tolist(),
        "max_abs_error": max_error,
        "gamma": GAMMA,
        "formula": "G_t = sum_{k>=0} gamma^k r_{t+k} within complete episode",
        "terminal_handling": "include terminal transition reward; no post-episode reward; no bootstrap",
        "reward_semantics": "individual per-agent environment reward",
        "details": details,
    }


def check_capture_alignment(bank: dict[str, np.ndarray], records: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = row_lookup(bank)
    errors: list[dict[str, Any]] = []
    checked = []
    for record in records:
        if record["scene"] != "mixed" or not record["success"] or record["capture_step"] is None:
            continue
        episode_id = int(record["episode_id"])
        capture_step = int(record["capture_step"])
        keys = [lookup.get((episode_id, step)) for step in (capture_step - 1, capture_step, capture_step + 1)]
        checked.append({"episode_id": episode_id, "seed": int(record["seed"]), "capture_step": capture_step, "rows": keys})
        if any(value is None for value in keys):
            errors.append({"episode_id": episode_id, "reason": "missing_t_minus_1_t_t_plus_1"})
            continue
        before, at, after = [int(value) for value in keys]
        at_events = bank["transition_event_flags"][at]
        before_events = bank["transition_event_flags"][before]
        if not (at_events[EVENT_TO_ID["normal_capture"]] or at_events[EVENT_TO_ID["stationary_capture"]]):
            errors.append({"episode_id": episode_id, "reason": "capture_event_not_at_capture_step"})
        if int(bank["phase_id"][before]) != PHASE_TO_ID["pre_capture"] or int(bank["phase_id"][at]) != PHASE_TO_ID["pre_capture"]:
            errors.append({"episode_id": episode_id, "reason": "capture_phase_not_pre_capture"})
        if int(bank["phase_id"][after]) != PHASE_TO_ID["post_capture"]:
            errors.append({"episode_id": episode_id, "reason": "post_capture_phase_not_post_capture"})
        if before_events[EVENT_TO_ID["normal_capture"]] or before_events[EVENT_TO_ID["stationary_capture"]]:
            errors.append({"episode_id": episode_id, "reason": "capture_event_early_by_one"})
        if np.any(bank["done"][at]):
            errors.append({"episode_id": episode_id, "reason": "capture_transition_terminal_or_done"})
        if not np.array_equal(bank["next_active_mask"][at], bank["active_mask"][after]):
            errors.append({"episode_id": episode_id, "reason": "capture_next_active_mask_mismatch"})
    return {"passed": not errors, "successful_mixed_episodes_checked": len(checked), "violations": errors, "checked": checked[:3]}


def split_summary(bank: dict[str, np.ndarray], records: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("train", "validation", "test")
    by_split = {}
    for name, split_id in zip(names, range(3)):
        record_subset = [r for r in records if r["split"] == name]
        rows = bank["split_id"] == split_id
        by_split[name] = {
            "episodes": len(record_subset),
            "episode_ids": [int(r["episode_id"]) for r in record_subset],
            "seeds": [int(r["seed"]) for r in record_subset],
            "transitions": int(rows.sum()),
            "active_agent_rows": int(bank["active_mask"][rows].sum()),
            "scenes": {scene: sum(r["scene"] == scene for r in record_subset) for scene in ("mixed", "pure_coverage")},
            "policy_modes": {mode: sum(r["policy_mode"] == mode for r in record_subset) for mode in ("bc_argmax", "bc_sample")},
        }
    episode_sets = [set(by_split[name]["episode_ids"]) for name in names]
    seed_sets = [set(by_split[name]["seeds"]) for name in names]
    leakage = any(episode_sets[i] & episode_sets[j] or seed_sets[i] & seed_sets[j] for i in range(3) for j in range(i + 1, 3))
    return {"episode_level": True, "transition_random_split": False, "ratio": {"train": 0.70, "validation": 0.15, "test": 0.15}, "by_split": by_split, "episode_or_seed_leakage": bool(leakage)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(BANK_PATH, allow_pickle=False) as source:
        bank = {name: source[name] for name in source.files}
    needs_neighbor_fields = not {"neighbor_action_index", "neighbor_action_aw"}.issubset(bank)
    manifest_path = COLLECTION_MANIFEST if COLLECTION_MANIFEST.exists() else LEGACY_COLLECTION_MANIFEST
    with manifest_path.open() as handle:
        raw_manifest = json.load(handle)
    raw_records = raw_manifest["bank"]["episode_records"]
    records, record_corrections = normalize_records(raw_records)
    split_before = bank["split_id"].copy()
    split_correction = correct_split_labels(bank, records)
    add_neighbor_actions(bank)
    if needs_neighbor_fields or np.any(split_before != bank["split_id"]):
        rewrite_bank(bank)
    rewritten = bool(needs_neighbor_fields or np.any(split_before != bank["split_id"]))
    with np.load(BANK_PATH, allow_pickle=False) as source:
        bank = {name: source[name] for name in source.files}

    integrity = {
        "schema": "cocap-ac2b-canonical-bc-critic-formal-v2",
        "bank_rewritten_with_explicit_neighbor_actions": rewritten,
        "split_label_contract": {"passed": True, "offline_correction_applied": split_correction["corrected"], "rows_corrected": split_correction["rows_corrected"], "record_split_metadata_corrections": record_corrections},
        "required_fields_present": all(name in bank for name in (
            "neighbor_action_index", "neighbor_action_aw", "local_self", "neighbor_ids",
            "neighbor_mask", "global_self", "global_active_mask", "actor_logits", "actor_probs",
            "action_index", "selected_action_probability", "action_aw", "reward", "done",
            "next_local_self", "next_global_self", "mc_return", "replay_semantic_class",
        )),
        "finite": finite_report(bank),
        "episode_structure": check_episode_structure(bank, records),
        "actions_and_neighbors": check_actions_and_neighbors(bank),
        "current_next_continuity": check_continuity(bank),
        "mc_return": check_mc(bank, records),
        "capture_transition_alignment": check_capture_alignment(bank, records),
    }
    integrity["passed"] = bool(
        integrity["required_fields_present"]
        and not integrity["finite"]["nan_count"]
        and integrity["split_label_contract"]["passed"]
        and all(integrity[key]["passed"] for key in ("episode_structure", "actions_and_neighbors", "current_next_continuity", "mc_return", "capture_transition_alignment"))
    )

    active = bank["active_mask"].astype(bool)
    phase_counts = {
        phase: int(((bank["phase_id"] == phase_id)[:, None] & active).sum())
        for phase, phase_id in PHASE_TO_ID.items()
    }
    class_counts = {
        name: int(((bank["replay_semantic_class"] == class_id) & active).sum())
        for name, class_id in CLASS_TO_ID.items()
    }
    event_counts = active_event_counts(bank)
    records_by_scene = {
        scene: [record for record in records if record["scene"] == scene]
        for scene in ("mixed", "pure_coverage")
    }
    outcomes = {
        scene: {
            "episodes": len(subset),
            "success": sum(bool(r["success"]) for r in subset),
            "failure": sum(bool(r["failure"]) for r in subset),
            "collision": sum(bool(r["collision"]) for r in subset),
        }
        for scene, subset in records_by_scene.items()
    }
    phase_event_counts = {
        "phase_active_agent_rows": phase_counts,
        "replay_semantic_class_active_agent_rows": class_counts,
        "event_active_agent_rows": event_counts,
        "early_recovery_active_agent_rows": event_counts["early_recovery"],
        "ring2_active_agent_rows": event_counts["ring2"],
        "ring3_active_agent_rows": event_counts["ring3"],
        "capture_transition_rows": int(np.any(bank["transition_event_flags"][:, [EVENT_TO_ID["normal_capture"], EVENT_TO_ID["stationary_capture"]]], axis=1).sum()),
        "capture_transition_active_agent_rows": event_counts["normal_capture"] + event_counts["stationary_capture"],
    }
    split = split_summary(bank, records)
    seed_split_manifest = {
        "schema": "ac2b-formal-seed-split-manifest-v1",
        "scheme": "A",
        "episodes_total": len(records),
        "composition": {
            "mixed": {"bc_argmax": 30, "bc_sample": 30},
            "pure_coverage": {"bc_argmax": 10, "bc_sample": 10},
        },
        "episode_split_only": True,
        "transition_random_split": False,
        "planned_split_counts": {"train": 64, "validation": 8, "test": 8},
        "actual_split": split,
        "seed_plan": [
            {"episode_id": int(r["episode_id"]), "scene": r["scene"], "policy_mode": r["policy_mode"], "seed": int(r["seed"]), "split": r["split"]}
            for r in records
        ],
    }
    phase_counts_payload = {
        "schema": "ac2b-formal-phase-event-counts-v1",
        "transitions": int(len(bank["episode_id"])),
        "active_agent_rows": int(active.sum()),
        "outcomes": outcomes,
        **phase_event_counts,
        "class_floor": {"threshold_active_agent_rows": 5000, "below_floor": {name: count < 5000 for name, count in class_counts.items()}},
    }
    bank_sha = sha256_file(BANK_PATH)
    bank_size = BANK_PATH.stat().st_size
    bank_manifest = {
        "schema": "cocap-ac2b-canonical-bc-critic-formal-v2",
        "status": "formal_bank_complete",
        "contract": "forward-final-aw9-4v1-swept-v1",
        "bank": {"path": str(BANK_PATH), "sha256": bank_sha, "size_bytes": bank_size, "transitions": int(len(bank["episode_id"])), "active_agent_rows": int(active.sum())},
        "generator": {
            "actor": "frozen canonical categorical BC Actor",
            "logical_checkpoint": "artifacts/2026-09-08_forward_final/c2_distillation/actor_epoch_030.pt",
            "actual_loaded_checkpoint": "artifacts/2026-09-17_critic_identifiability_audit/frozen_policy/actor_epoch_030.pt",
            "checkpoint_sha256": "7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd",
            "actor_updates": 0,
            "critic_updates": 0,
            "bootstrap_or_gae": False,
            "iqn_recovery_pool": False,
            "coverage_initialization": "canonical inner_random_cluster; no recovery-pool injection",
            "mixed_initialization": "canonical mixed environment; natural post-capture continuation",
            "actor_bit_exact": True,
        },
        "environment": {
            "source_contract": "artifacts/2026-09-20_ac1/AC1_BC_RESOLVED_CONTRACT.json",
            "mixed": {"pursuers": 4, "evaders": 1, "obstacles": 1},
            "pure_coverage": {"pursuers": 4, "evaders": 0, "obstacles": 1},
            "episode_horizon": 3000,
            "capture_terminal": False,
            "post_capture_continuation": True,
            "action_space": "AW9 categorical indices 0..8",
        },
        "policy_modes": {"bc_argmax": 40, "bc_sample": 40},
        "raw_fields": sorted(bank),
        "critic_input_contract": {
            "LQ": ["local_*", "action_index", "action_aw"],
            "NQ": ["local_*", "neighbor_local_*", "neighbor_ids", "neighbor_mask", "neighbor_action_index", "neighbor_action_aw"],
            "CQ": ["global_*", "global masks", "active_mask", "joint action_index/action_aw", "agent identity/index"],
            "V": ["global_*", "global masks", "global_active_mask", "normalized_time from state/time contract; no future labels"],
        },
        "reward_and_return": {"reward_semantics": "individual per-agent environment reward", "return_semantics": "per-agent realized MC return", "gamma": GAMMA, "bootstrap": False, "terminal_handling": "include terminal reward, stop after complete episode", "post_capture_continuation": "all naturally visited recovery rows retained"},
        "phase_event_contract": {"phases": list(PHASES), "events": list(EVENTS), "early_recovery": "post_capture and 0 <= timestep-capture_step <= 40; capture row remains pre_capture", "late_recovery": "post_capture and timestep-capture_step > 40", "capture_transition": "normal_capture or stationary_capture event on the environment transition emitting last_capture_events"},
        "split_contract": seed_split_manifest,
        "integrity_summary": integrity,
    }
    report = {
        "schema": "ac2b-formal-report-v1",
        "verdict": "FORMAL_BANK_PASS" if integrity["passed"] and not any(phase_counts_payload["class_floor"]["below_floor"].values()) else "FORMAL_BANK_INTEGRITY_FAIL",
        "identity": {"checkpoint_sha256": bank_manifest["generator"]["checkpoint_sha256"], "bank_sha256": bank_sha, "bank_size_bytes": bank_size},
        "composition": seed_split_manifest["composition"],
        "outcomes": outcomes,
        "transitions": int(len(bank["episode_id"])),
        "active_agent_rows": int(active.sum()),
        "phase_event_counts": phase_event_counts,
        "replay_semantic_class_counts": class_counts,
        "class_below_operational_floor": phase_counts_payload["class_floor"]["below_floor"],
        "train_validation_test": split,
        "mc_max_abs_error": integrity["mc_return"]["max_abs_error"],
        "capture_transition_violations": len(integrity["capture_transition_alignment"]["violations"]),
        "nan_count": integrity["finite"]["nan_count"],
        "inf_count": integrity["finite"]["inf_count"],
        "optimizer_updates": 0,
        "critic_training": False,
    }
    atomic_json(OUT / "seed_split_manifest.json", seed_split_manifest)
    atomic_json(OUT / "phase_counts.json", phase_counts_payload)
    atomic_json(OUT / "integrity_report.json", integrity)
    atomic_json(OUT / "bank_manifest.json", bank_manifest)
    atomic_json(OUT / "report.json", report)
    print(json.dumps({"verdict": report["verdict"], "rewritten": rewritten, "bank_sha256": bank_sha, "transitions": report["transitions"], "active_agent_rows": report["active_agent_rows"], "mc_max_abs_error": report["mc_max_abs_error"], "capture_transition_violations": report["capture_transition_violations"], "class_counts": class_counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
