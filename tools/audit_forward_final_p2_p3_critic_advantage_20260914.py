#!/usr/bin/env python3
"""Frozen-BC P2/P3 critic and advantage diagnostic.

This is deliberately an evaluation-only entry point.  It reuses the current
Forward Final transition collector, loads an existing fixed-MC critic, and
never calls a critic or actor optimizer.  The report separates realized MC
targets from the production GAE/lambda target and keeps the six requested
phase/event buckets disjoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

from cocap_voradj.training.small_step_ac import compute_gae, tensor_tree
from tools.run_forward_final_bridge_20260908 import atomic_json
from tools.train_forward_final_ppo_20260909 import (
    BC_SHA,
    FinalMissionStream,
    collect_transition,
    empty_rollout,
    make_trainer,
    tensor_hash,
)


GAMMA = 0.99
GAE_LAMBDA = 0.95
ROLLOUT_LENGTH = 256
BUCKETS = (
    "pre_capture",
    "capture_near",
    "capture_transition",
    "early_recovery",
    "late_recovery",
    "pure_coverage_early",
    "pure_coverage_steady",
)
PHASES = ("pre_capture", "post_capture", "pure_coverage")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks without requiring scipy."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def calibration(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(prediction)
    target, prediction = target[valid], prediction[valid]
    if not len(target):
        return {"rows": 0}
    error = prediction - target
    variance = float(np.var(target))
    pred_variance = float(np.var(prediction))
    if pred_variance > 1e-12:
        slope, intercept = np.polyfit(prediction, target, 1)
        slope, intercept = float(slope), float(intercept)
    else:
        slope = intercept = None
    return {
        "rows": int(len(target)),
        "episodes": None,
        "target": finite_stats(target),
        "prediction": finite_stats(prediction),
        "residual": finite_stats(error),
        "bias_prediction_minus_target": float(np.mean(error)),
        "median_prediction_minus_target": float(np.median(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "explained_variance": None if variance <= 1e-12 else 1.0 - float(np.var(error)) / variance,
        "calibration_slope_target_on_prediction": slope,
        "calibration_intercept_target_on_prediction": intercept,
        "rank_correlation": spearman(prediction, target),
    }


def quantile_calibration(target: np.ndarray, prediction: np.ndarray, bins: int = 5) -> list[dict[str, Any]]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(prediction)
    target, prediction = target[valid], prediction[valid]
    if not len(target):
        return []
    order = np.argsort(prediction, kind="mergesort")
    chunks = np.array_split(order, min(bins, len(order)))
    result = []
    for index, ids in enumerate(chunks):
        pred = prediction[ids]
        gold = target[ids]
        result.append({
            "bin": int(index),
            "rows": int(len(ids)),
            "prediction_mean": float(np.mean(pred)),
            "target_mean": float(np.mean(gold)),
            "bias_prediction_minus_target": float(np.mean(pred - gold)),
            "prediction_p10": float(np.quantile(pred, 0.10)),
            "prediction_p90": float(np.quantile(pred, 0.90)),
        })
    return result


def full_returns(rewards: np.ndarray, terminated: np.ndarray) -> np.ndarray:
    result = np.zeros_like(rewards, dtype=np.float64)
    future = np.zeros(rewards.shape[1], dtype=np.float64)
    for index in range(len(rewards) - 1, -1, -1):
        future = np.asarray(rewards[index], dtype=np.float64) + GAMMA * future * (
            ~np.asarray(terminated[index], dtype=bool)
        )
        result[index] = future
    return result


def phase_for_stream(stream: FinalMissionStream) -> str:
    if stream.task == "voradj_coverage":
        return "pure_coverage"
    return "pre_capture" if any(not evader.deactivated for evader in stream.env.evaders) else "post_capture"


def load_geometry_checkpoint(trainer: Any, checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    if "trainer" not in payload:
        raise ValueError(f"unsupported fixed-MC checkpoint: {checkpoint}")
    trainer.load_state_dict(payload["trainer"])
    trainer.actor.eval()
    trainer.value.eval()
    for parameter in trainer.actor.parameters():
        parameter.requires_grad_(False)
    for parameter in trainer.value.parameters():
        parameter.requires_grad_(False)
    return payload


def collect_heldout(trainer: Any, seed: int, episodes: int, out: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    stream = FinalMissionStream(seed, out / "stream")
    captured_events: list[list[dict[str, Any]]] = []
    original_step = stream.step

    def tapped(indices: np.ndarray):
        outcome = original_step(indices)
        # StepResult intentionally carries observations/rewards/dones/infos;
        # capture events remain on the live env until collect_transition calls
        # finish/reset.  Read that audited runtime field before reset.
        raw_events = getattr(stream.env, "last_capture_events", []) or []
        captured_events.append(json.loads(json.dumps(raw_events)))
        return outcome

    stream.step = tapped
    rows: list[dict[str, Any]] = []
    episode_reports: list[dict[str, Any]] = []
    episode_steps: defaultdict[int, int] = defaultdict(int)
    episode_start = 0
    for _ in range(episodes):
        while True:
            episode_id = int(stream.episode)
            phase = phase_for_stream(stream)
            timestep = episode_steps[episode_id]
            before_event_count = len(captured_events)
            row, completed = collect_transition(trainer, stream)
            if len(captured_events) != before_event_count + 1:
                raise AssertionError("collector did not produce exactly one step event")
            events = captured_events[-1]
            stored_v = torch.as_tensor(row["values"], device=trainer.device)
            stored_next_v = torch.as_tensor(row["next_values"], device=trainer.device)
            with torch.no_grad():
                raw_v = trainer._denormalize_values(stored_v).detach().cpu().numpy()
                raw_next_v = trainer._denormalize_values(stored_next_v).detach().cpu().numpy()
            rows.append({
                "episode": episode_id,
                "timestep": timestep,
                "phase": phase,
                "capture_events": events,
                "global_obs": row["global_obs"],
                "rewards": np.asarray(row["rewards"], dtype=np.float32),
                "values": np.asarray(raw_v, dtype=np.float32),
                "next_values": np.asarray(raw_next_v, dtype=np.float32),
                "active": np.asarray(row["active_mask"], dtype=bool),
                "terminated": np.asarray(row["terminated"], dtype=bool),
                "truncated": np.asarray(row["truncated"], dtype=bool),
                "episode_end": np.asarray(row["episode_end"], dtype=bool),
                "actions": np.asarray(row["latent"], dtype=np.int64),
            })
            episode_steps[episode_id] += 1
            if completed is not None:
                episode_reports.append({"episode": episode_id, "length": len(rows) - episode_start, **completed})
                episode_start = len(rows)
                break
    if len(episode_reports) != episodes:
        raise AssertionError(f"expected {episodes} completed episodes, got {len(episode_reports)}")
    keys = ("rewards", "values", "next_values", "active", "terminated", "truncated", "episode_end", "actions")
    arrays = {key: np.stack([row[key] for row in rows]) for key in keys}
    arrays["episode"] = np.asarray([row["episode"] for row in rows], dtype=np.int64)
    arrays["timestep"] = np.asarray([row["timestep"] for row in rows], dtype=np.int64)
    arrays["phase"] = np.asarray([row["phase"] for row in rows], dtype=object)
    arrays["capture_events_json"] = np.asarray(
        [json.dumps(row["capture_events"], sort_keys=True) for row in rows], dtype=object
    )
    return arrays, episode_reports


def assign_buckets(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[int, int | None]]:
    episodes = arrays["episode"]
    timesteps = arrays["timestep"]
    phases = arrays["phase"]
    capture_t: dict[int, int | None] = {}
    for episode in np.unique(episodes):
        mask = episodes == episode
        times = timesteps[mask]
        events = [json.loads(value) for value in arrays["capture_events_json"][mask]]
        event_times = [int(time) for time, event in zip(times, events) if event]
        capture_t[int(episode)] = min(event_times) if event_times else None
    buckets = np.empty(len(episodes), dtype=object)
    for index, (episode, timestep, phase) in enumerate(zip(episodes, timesteps, phases)):
        event_t = capture_t[int(episode)]
        if phase == "pure_coverage":
            buckets[index] = "pure_coverage_early" if timestep < 20 else "pure_coverage_steady"
        elif event_t is None:
            buckets[index] = "pre_capture"
        elif timestep == event_t:
            buckets[index] = "capture_transition"
        elif timestep < event_t:
            buckets[index] = "capture_near" if timestep >= event_t - 10 else "pre_capture"
        else:
            recovery_offset = timestep - event_t - 1
            buckets[index] = "early_recovery" if recovery_offset < 20 else "late_recovery"
    return buckets, capture_t


def calculate_gae(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    advantages = np.zeros_like(arrays["rewards"], dtype=np.float32)
    returns = np.zeros_like(arrays["rewards"], dtype=np.float32)
    normalized = np.zeros_like(arrays["rewards"], dtype=np.float32)
    for start in range(0, len(arrays["rewards"]), ROLLOUT_LENGTH):
        stop = min(start + ROLLOUT_LENGTH, len(arrays["rewards"]))
        adv, ret = compute_gae(
            torch.as_tensor(arrays["rewards"][start:stop]),
            torch.as_tensor(arrays["values"][start:stop]),
            torch.as_tensor(arrays["next_values"][start:stop]),
            torch.as_tensor(arrays["terminated"][start:stop]),
            torch.as_tensor(arrays["active"][start:stop]),
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            truncated=torch.as_tensor(arrays["truncated"][start:stop]),
            episode_end=torch.as_tensor(arrays["episode_end"][start:stop]),
        )
        adv = adv.numpy()
        ret = ret.numpy()
        active = arrays["active"][start:stop]
        valid = adv[active]
        mean = float(valid.mean()) if len(valid) else 0.0
        std = float(valid.std()) if len(valid) else 1.0
        std = max(std, 1e-6)
        advantages[start:stop] = adv
        returns[start:stop] = ret
        normalized[start:stop] = (adv - mean) / std
    return advantages, returns, normalized


def episode_mc(arrays: dict[str, np.ndarray], reports: list[dict[str, Any]]) -> np.ndarray:
    result = np.full_like(arrays["rewards"], np.nan, dtype=np.float64)
    for report in reports:
        episode = int(report["episode"])
        mask = arrays["episode"] == episode
        if np.any(arrays["truncated"][mask]):
            continue
        result[mask] = full_returns(arrays["rewards"][mask], arrays["terminated"][mask])
    return result.astype(np.float32)


def rows_for(bucket: str, buckets: np.ndarray, active: np.ndarray) -> np.ndarray:
    return active & (buckets == bucket)[:, None]


def p2_metrics(
    bucket: str,
    buckets: np.ndarray,
    arrays: dict[str, np.ndarray],
    mc: np.ndarray,
    gae_returns: np.ndarray,
) -> dict[str, Any]:
    active = arrays["active"]
    mask = rows_for(bucket, buckets, active)
    episode_count = len(np.unique(arrays["episode"][np.any(mask, axis=1)]))
    result = {"bucket": bucket, "active_rows": int(mask.sum()), "episodes": int(episode_count)}
    for target_name, target in (("mc_return", mc), ("gae_lambda_return", gae_returns)):
        target_flat = target[mask]
        value_flat = arrays["values"][mask]
        stats = calibration(target_flat, value_flat)
        stats["episodes"] = int(episode_count)
        stats["prediction_quantile_calibration"] = quantile_calibration(target_flat, value_flat)
        result[target_name] = stats
    result["reward_sum"] = float(arrays["rewards"][mask].sum())
    result["reward_per_active_row"] = float(arrays["rewards"][mask].mean()) if mask.any() else None
    return result


def p3_metrics(
    bucket: str,
    buckets: np.ndarray,
    arrays: dict[str, np.ndarray],
    mc: np.ndarray,
    advantages: np.ndarray,
    normalized_advantages: np.ndarray,
) -> dict[str, Any]:
    mask = rows_for(bucket, buckets, arrays["active"])
    mc_values = mc[mask]
    valid = np.isfinite(mc_values)
    advantage = advantages[mask][valid]
    normalized = normalized_advantages[mask][valid]
    value = arrays["values"][mask][valid]
    mc_values = mc_values[valid]
    residual = mc_values - value
    if not len(advantage):
        return {"bucket": bucket, "active_rows": int(mask.sum()), "mc_rows": 0}
    positive = advantage > 0
    negative = advantage < 0
    normalized_positive = normalized > 0
    return {
        "bucket": bucket,
        "active_rows": int(mask.sum()),
        "mc_rows": int(len(advantage)),
        "advantage": finite_stats(advantage),
        "normalized_advantage": finite_stats(normalized),
        "absolute_advantage_p90": float(np.quantile(np.abs(advantage), 0.90)),
        "absolute_advantage_sum": float(np.abs(advantage).sum()),
        "positive_fraction_raw": float(np.mean(positive)),
        "positive_fraction_normalized": float(np.mean(normalized_positive)),
        "raw_to_normalized_sign_flip_fraction": float(np.mean(np.sign(advantage) != np.sign(normalized))),
        "sign_agreement_with_mc_minus_v": float(np.mean(np.sign(advantage) == np.sign(residual))),
        "positive_advantage_but_mc_minus_v_negative": float(np.mean(positive & (residual < 0))),
        "negative_advantage_but_mc_minus_v_positive": float(np.mean(negative & (residual > 0))),
        "mean_mc_minus_v_when_advantage_positive": float(np.mean(residual[positive])) if positive.any() else None,
        "mean_mc_minus_v_when_advantage_negative": float(np.mean(residual[negative])) if negative.any() else None,
        "advantage_vs_mc_minus_v_spearman": spearman(advantage, residual),
        "normalized_vs_raw_advantage_spearman": spearman(normalized, advantage),
    }


def episode_extremes(
    arrays: dict[str, np.ndarray],
    reports: list[dict[str, Any]],
    mc: np.ndarray,
    advantages: np.ndarray,
    normalized_advantages: np.ndarray,
) -> dict[str, Any]:
    complete = [item for item in reports if not item.get("truncated", False)]
    if not complete:
        return {"available": False, "reason": "no complete episodes"}
    lengths = np.asarray([item["length"] for item in complete], dtype=np.float64)
    low = float(np.quantile(lengths, 0.25))
    high = float(np.quantile(lengths, 0.75))
    groups = {
        "fast_q25": [item for item in complete if item["length"] <= low],
        "slow_q75": [item for item in complete if item["length"] >= high],
    }
    result: dict[str, Any] = {"available": True, "lengths": finite_stats(lengths), "groups": {}}
    for label, items in groups.items():
        mask = np.isin(arrays["episode"], [item["episode"] for item in items])[:, None] & arrays["active"]
        a = advantages[mask]
        na = normalized_advantages[mask]
        m = mc[mask]
        v = arrays["values"][mask]
        valid = np.isfinite(m)
        result["groups"][label] = {
            "episodes": [int(item["episode"]) for item in items],
            "mean_length": float(np.mean([item["length"] for item in items])),
            "rows": int(mask.sum()),
            "mean_raw_advantage": float(np.mean(a)),
            "mean_normalized_advantage": float(np.mean(na)),
            "positive_raw_fraction": float(np.mean(a > 0)),
            "mean_mc_minus_v": float(np.mean(m[valid] - v[valid])) if valid.any() else None,
            "mean_mc_return": float(np.mean(m[valid])) if valid.any() else None,
        }
    return result


def teacher_q_ranking(trainer: Any, dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    supported = manifest.get("contract") == "forward-final-aw9-4v1-swept-v1"
    if not supported:
        return {
            "contract_supported": False,
            "reason": "teacher dataset contract does not match current Forward Final contract",
        }
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    total_rows = 0
    action_mismatch = 0
    teacher_q_gaps = []
    for shard in manifest["shards"]:
        path = dataset_root / shard["path"]
        data = np.load(path, allow_pickle=False)
        local = {
            key[len("local_obs."):]: data[key]
            for key in data.files
            if key.startswith("local_obs.")
        }
        with torch.no_grad():
            distribution = trainer.actor.distribution(
                tensor_tree({key: torch.as_tensor(value, device=trainer.device) for key, value in local.items()}, trainer.device)
            )
            probabilities = distribution.probs.detach().cpu().numpy()
        bc_action = probabilities.argmax(axis=1)
        teacher_q = np.asarray(data["teacher_q"], dtype=np.float64)
        teacher_action = np.asarray(data["greedy_action"], dtype=np.int64)
        if not np.array_equal(teacher_action, teacher_q.argmax(axis=1)):
            raise AssertionError(f"teacher Q/greedy mismatch in {path}")
        action_mismatch += int(np.sum(bc_action != teacher_action))
        total_rows += len(bc_action)
        q_best = teacher_q.max(axis=1)
        q_chosen = teacher_q[np.arange(len(teacher_q)), bc_action]
        gap = q_best - q_chosen
        teacher_q_gaps.extend(gap.tolist())
        phase_names = {0: "pre_capture", 1: "post_capture", 2: "pure_coverage"}
        role_names = {0: "direct", 1: "pursuing_memory", 2: "support", 3: "coverage"}
        group_labels = [
            ("overall", np.ones(len(bc_action), dtype=bool)),
            ("near_capture", np.asarray(data["near_capture"], dtype=bool)),
            ("capture_transition", np.asarray(data["capture_transition"], dtype=bool)),
        ]
        group_labels.extend(
            ("phase/" + phase_names[int(value)], data["phase_id"] == value)
            for value in sorted(set(data["phase_id"].tolist()))
        )
        group_labels.extend(
            ("role/" + role_names[int(value)], data["role_id"] == value)
            for value in sorted(set(data["role_id"].tolist()))
        )
        for label, mask in group_labels:
            if not np.any(mask):
                continue
            selected_gap = gap[mask]
            selected_match = bc_action[mask] == teacher_action[mask]
            payload = {
                "rows": int(mask.sum()),
                "bc_action_matches_teacher_greedy_fraction": float(np.mean(selected_match)),
                "teacher_q_gap_best_minus_bc_action": finite_stats(selected_gap),
                "bc_action_is_teacher_top1": bool(np.all(selected_match)) if len(selected_match) == 1 else None,
                "bc_action_teacher_top1_mismatch_fraction": float(np.mean(~selected_match)),
                "bc_action_teacher_rank_mean": float(np.mean(1 + np.sum(teacher_q[mask] > q_chosen[mask, None], axis=1))),
                "bc_action_teacher_rank_p90": float(np.quantile(1 + np.sum(teacher_q[mask] > q_chosen[mask, None], axis=1), 0.90)),
                "bc_action_probability": finite_stats(probabilities[mask, bc_action[mask]]),
            }
            groups[label].append(payload)
    summary = {}
    for label, values in groups.items():
        rows = sum(item["rows"] for item in values)
        summary[label] = {
            "rows": rows,
            "bc_action_matches_teacher_greedy_fraction": float(
                sum(item["rows"] * item["bc_action_matches_teacher_greedy_fraction"] for item in values) / max(rows, 1)
            ),
            "teacher_q_gap_best_minus_bc_action": finite_stats(
                np.repeat(
                    [item["teacher_q_gap_best_minus_bc_action"]["mean"] for item in values],
                    [item["rows"] for item in values],
                )
            ),
            "subgroups": values,
        }
    return {
        "contract_supported": True,
        "contract": manifest.get("contract"),
        "teacher_sha256": manifest.get("teacher_sha256"),
        "rows": total_rows,
        "bc_action_teacher_greedy_mismatch_fraction": float(action_mismatch / max(total_rows, 1)),
        "overall_teacher_q_gap_best_minus_bc_action": finite_stats(np.asarray(teacher_q_gaps)),
        "groups": summary,
        "empirical_action_ranking_supported": False,
        "empirical_action_ranking_reason": (
            "C1 stores one realized reward/trajectory per state-action and has no matched same-state "
            "counterfactual BC/PPO actions; observed MC return cannot rank alternative actions."
        ),
    }


def build_report(
    arrays: dict[str, np.ndarray],
    reports: list[dict[str, Any]],
    buckets: np.ndarray,
    capture_t: dict[int, int | None],
    mc: np.ndarray,
    advantages: np.ndarray,
    gae_returns: np.ndarray,
    normalized_advantages: np.ndarray,
    actor_hash_before: str,
    actor_hash_after: str,
    teacher_report: dict[str, Any],
) -> dict[str, Any]:
    p2 = {bucket: p2_metrics(bucket, buckets, arrays, mc, gae_returns) for bucket in BUCKETS}
    p3 = {bucket: p3_metrics(bucket, buckets, arrays, mc, advantages, normalized_advantages) for bucket in BUCKETS}
    p2["overall"] = p2_metrics("overall", np.asarray(["overall"] * len(buckets), dtype=object), arrays, mc, gae_returns)
    p3["overall"] = p3_metrics("overall", np.asarray(["overall"] * len(buckets), dtype=object), arrays, mc, advantages, normalized_advantages)
    non_early = [
        p2[bucket]["mc_return"]["rmse"]
        for bucket in ("pre_capture", "capture_near", "capture_transition", "late_recovery", "pure_coverage_early", "pure_coverage_steady")
        if p2[bucket]["mc_return"].get("rows", 0)
    ]
    early_rmse = p2["early_recovery"]["mc_return"].get("rmse")
    early_target_std = p2["early_recovery"]["mc_return"].get("target", {}).get("std")
    comparisons = {
        "early_recovery_mc_rmse_over_median_other_bucket_rmse": (
            float(early_rmse / np.median(non_early)) if early_rmse is not None and non_early else None
        ),
        "early_recovery_target_std_over_median_other_bucket_target_std": (
            float(
                early_target_std
                / np.median([
                    p2[bucket]["mc_return"]["target"]["std"]
                    for bucket in BUCKETS
                    if bucket != "early_recovery" and p2[bucket]["mc_return"].get("rows", 0)
                ])
            )
            if early_target_std is not None
            else None
        ),
        "early_recovery_hotspot_rule": (
            "hotspot if both RMSE and target std exceed the median of the other available buckets"
        ),
    }
    abs_mass = sum(item.get("absolute_advantage_sum", 0.0) for item in p3.values() if "absolute_advantage_sum" in item)
    for bucket in BUCKETS:
        if "absolute_advantage_sum" in p3[bucket]:
            p3[bucket]["absolute_advantage_mass_share"] = float(p3[bucket]["absolute_advantage_sum"] / max(abs_mass, 1e-12))
    episode_lengths = {str(item["episode"]): int(item["length"]) for item in reports}
    return {
        "schema": "forward-final-p2-p3-critic-advantage-diagnostic-v1",
        "contract": "forward-final-aw9-4v1-swept-v1",
        "transition_semantics": "terminal-priority-truncation-bootstrap-weighted-ce-v2",
        "actor_updates": 0,
        "critic_updates": 0,
        "normalizer_updates": 0,
        "actor_bit_exact": actor_hash_before == actor_hash_after,
        "actor_hash_before": actor_hash_before,
        "actor_hash_after": actor_hash_after,
        "episodes": reports,
        "episode_lengths": episode_lengths,
        "capture_t_by_episode": {str(key): value for key, value in capture_t.items()},
        "sample_counts": {
            bucket: {
                "active_rows": int(rows_for(bucket, buckets, arrays["active"]).sum()),
                "episodes": int(len(np.unique(arrays["episode"][np.any(rows_for(bucket, buckets, arrays["active"]), axis=1)]))),
            }
            for bucket in BUCKETS
        },
        "p2_value_calibration": p2,
        "p2_early_recovery_comparison": comparisons,
        "p3_advantage_semantics": p3,
        "p3_episode_extremes": episode_extremes(arrays, reports, mc, advantages, normalized_advantages),
        "p3_normalization": {
            "rollout_length": ROLLOUT_LENGTH,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "ordering_preserved_in_theory": True,
            "overall_raw_vs_normalized_spearman": p3["overall"].get("normalized_vs_raw_advantage_spearman"),
            "overall_raw_to_normalized_sign_flip_fraction": p3["overall"].get("raw_to_normalized_sign_flip_fraction"),
        },
        "teacher_q_ranking": teacher_report,
        "limitations": [
            "MC is a single realized return under the frozen BC rollout; rows within an episode and agents are correlated.",
            "The held-out episodes are safe in the current seed stream unless episode reports state otherwise; failure calibration is unavailable when no failures occur.",
            "Teacher-Q ranking is legal only for the current Forward Final C1 contract; it is not a causal empirical action ranking.",
            "No same-state counterfactual BC/PPO actions are stored, so empirical short-horizon/MC ranking of alternative actions is unsupported.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/2026-09-09_forward_final_d/fixed_mc_critic/critic_mc_100updates.pt")
    parser.add_argument("--teacher-dataset", type=Path, default=ROOT / "artifacts/2026-09-08_forward_final/c1_dataset")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2027099201)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (args.out / "report.json").exists():
        raise ValueError("fresh output required")
    args.out.mkdir(parents=True, exist_ok=True)
    trainer = make_trainer(args.device, 2026097101)
    checkpoint_payload = load_geometry_checkpoint(trainer, args.checkpoint)
    actor_before = tensor_hash(trainer.actor.state_dict())
    if checkpoint_payload.get("actor_sha256") != actor_before:
        raise AssertionError("fixed-MC checkpoint actor tensor hash does not match loaded actor")
    actor_before = tensor_hash(trainer.actor.state_dict())
    arrays, reports = collect_heldout(trainer, args.seed, args.episodes, args.out)
    buckets, capture_t = assign_buckets(arrays)
    advantages, gae_returns, normalized_advantages = calculate_gae(arrays)
    mc = episode_mc(arrays, reports)
    teacher = teacher_q_ranking(trainer, args.teacher_dataset)
    actor_after = tensor_hash(trainer.actor.state_dict())
    np.savez_compressed(
        args.out / "rollout_diagnostics.npz",
        rewards=arrays["rewards"],
        values=arrays["values"],
        next_values=arrays["next_values"],
        active=arrays["active"],
        terminated=arrays["terminated"],
        truncated=arrays["truncated"],
        episode=arrays["episode"],
        timestep=arrays["timestep"],
        bucket=np.asarray(buckets, dtype="U32"),
        mc_return=mc,
        gae_return=gae_returns,
        advantage=advantages,
        normalized_advantage=normalized_advantages,
        actions=arrays["actions"],
    )
    report = build_report(
        arrays,
        reports,
        buckets,
        capture_t,
        mc,
        advantages,
        gae_returns,
        normalized_advantages,
        actor_before,
        actor_after,
        teacher,
    )
    report["launch"] = {
        "seed": args.seed,
        "episodes": args.episodes,
        "device": str(args.device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "teacher_dataset": str(args.teacher_dataset),
        "teacher_manifest_sha256": sha256_file(args.teacher_dataset / "manifest.json"),
        "bc_parent_sha256": BC_SHA,
        "source_sha256": sha256_file(Path(__file__)),
        "ppo_rollout_length": ROLLOUT_LENGTH,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "optimizer_steps": 0,
    }
    atomic_json(args.out / "report.json", report)
    atomic_json(args.out / "progress.json", {"status": "complete", "episodes": args.episodes, "optimizer_steps": 0})
    print(json.dumps({
        "status": "complete",
        "episodes": args.episodes,
        "actor_bit_exact": report["actor_bit_exact"],
        "p2_early_recovery": report["p2_early_recovery_comparison"],
        "p3_overall": report["p3_advantage_semantics"]["overall"],
        "teacher_q": {key: value for key, value in teacher.items() if key not in ("groups",)},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
