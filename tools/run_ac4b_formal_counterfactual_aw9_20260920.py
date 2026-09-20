#!/usr/bin/env python3
"""AC-4B: formal frozen-BC AW9 counterfactual ranking audit.

This runner intentionally contains no optimizer, learner, bootstrap, PPO, or
trajectory-bank writer.  It registers a fresh seed manifest, collects only
compact in-memory anchor snapshots from natural canonical-BC argmax rollouts,
branches the focal AW9 action, and writes small matrices/metadata only.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
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

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.evaluation.mission_events import snapshot as mission_snapshot
from cocap_voradj.training.continuous.central_schema import build_central_global_obs
from tools.collect_ac2b_canonical_bc_bank_20260920 import ACTION_GRID, event_flags
from tools.distill_forward_final_actor_20260908 import load_actor
from tools.run_ac3a_critic_sampler_audit_20260920 import sha256_file
from tools.run_ac4a_counterfactual_aw9_pilot_20260920 import (
    ACTOR_LOADED,
    ACTOR_REQUESTED,
    ACTOR_SHA256,
    AC3A_SUMMARY,
    AC3B_SUMMARY,
    ACTION_SCALE,
    GAMMA,
    MAX_AGENTS,
    MAX_NEIGHBORS,
    atomic_json,
    canonical_actions,
    canonical_path_from_state,
    compare_observations,
    critic_predictions,
    current_phase_class,
    json_safe,
    load_critics,
    pad_observations,
    save_env_state,
    step_once,
    tensor_state_sha256,
)
from tools.run_continuous_ctde_training import _split_termination_flags
from tools.run_forward_final_bridge_20260908 import act_evaders, make_env


OUT = ROOT / "artifacts/2026-09-20_ac4b"
AC4A_SUMMARY = ROOT / "artifacts/2026-09-20_ac4a/summary.json"
ANCHOR_TARGETS = {
    "pursuing": 16,
    "pre_capture_cover": 16,
    "early_recovery": 16,
    "recovery_pure": 16,
}
ANCHOR_ORDER = tuple(ANCHOR_TARGETS)
MIXED_SEEDS = tuple(2026092501 + index for index in range(64))
PURE_SEEDS = tuple(2026092601 + index for index in range(64))
MAX_ANCHORS_PER_CLASS_PER_EPISODE = 2
MIN_SAME_CLASS_STEP_GAP = 10
GAMMA = 0.99
RETURN_TOLERANCE = 1e-3
NOISE_CHECK_ANCHOR_OFFSETS = (0, 16, 32, 48)
NOISE_CHECK_ACTIONS = (0, 4, 8)
EXTREME_DELAY_FRACTION = 0.90
SNAPSHOT_STATES = 5
SNAPSHOT_CONTINUATION_STEPS = 5
EXPECTED_CRITIC_SHA = {
    "LQ": "e8b52a32040937c2a4060e359546d1a427aee6c955dcaec501827038fac27552",
    "NQ": "53a194b17d9e963c5a63fce137d44725df19dc59420b624a5657c87d290a5052",
    "CQ": "361ace40555e478ec5ad9eb6e5cf5f8b7daafca1fb11daed49ec67e5b63db4e0",
}
EXPECTED_CRITIC_SEED = {"LQ": 2026091711, "NQ": 2026091711, "CQ": 2026091711}


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def seed_manifest() -> dict[str, Any]:
    return {
        "schema": "ac4b-seed-manifest-v1",
        "registered_before_rollout": True,
        "seed_policy": "fresh fixed natural canonical-BC argmax rollout; no noise/exploration",
        "excluded_ac4a_seed_base": 2026092401,
        "mixed": [int(value) for value in MIXED_SEEDS],
        "pure_coverage": [int(value) for value in PURE_SEEDS],
        "anchor_targets": ANCHOR_TARGETS,
        "max_anchors_per_episode_per_class": MAX_ANCHORS_PER_CLASS_PER_EPISODE,
        "minimum_same_class_step_gap": MIN_SAME_CLASS_STEP_GAP,
    }


def make_anchor(
    env: Any,
    observations: list[Any],
    state: Mapping[str, Any],
    actions: np.ndarray,
    scene: str,
    seed: int,
    episode_index: int,
    focal: int,
    phase: str,
    semantic_class: str,
    recovery_age: int,
    capture_step: int | None,
    within_episode_index: int,
) -> dict[str, Any]:
    local_full, _ = pad_observations(observations)
    neighbor_ids = np.full((MAX_AGENTS, MAX_NEIGHBORS), -1, dtype=np.int16)
    neighbor_mask = np.zeros((MAX_AGENTS, MAX_NEIGHBORS), dtype=bool)
    ids = sorted(state["friends"].get(int(focal), set()))[:MAX_NEIGHBORS]
    neighbor_ids[int(focal), : len(ids)] = ids
    neighbor_mask[int(focal), : len(ids)] = True
    central = build_central_global_obs(
        env,
        max_agents=MAX_AGENTS,
        max_evaders=8,
        max_obstacles=5,
        self_feature_dim=9,
    )
    joint_aw = np.zeros((MAX_AGENTS, 2), dtype=np.float32)
    joint_aw[: len(actions)] = ACTION_GRID[np.clip(actions, 0, 8)]
    return {
        "scene": scene,
        "seed": int(seed),
        "episode_id": f"{scene}:{seed}",
        "episode_index": int(episode_index),
        "episode_step": int(env.episode_step),
        "transition_step": int(env.episode_step + 1),
        "episode_horizon": int(env.episode_max_length),
        "focal": int(focal),
        "semantic_class": semantic_class,
        "phase": phase,
        "recovery_age": int(recovery_age),
        "capture_step": capture_step,
        "within_episode_index": int(within_episode_index),
        "bundle": save_env_state(env, observations, capture_step),
        "local_full": local_full,
        "neighbor_ids": neighbor_ids,
        "neighbor_mask": neighbor_mask,
        "central": central,
        "joint_action_indices": actions.copy(),
        "joint_aw": joint_aw,
        "original_bc_action": int(actions[int(focal)]),
        "event_labels_at_anchor": {
            "phase": phase,
            "semantic_class": semantic_class,
            "recovery_age": int(recovery_age),
            "ring_participant_count": int(
                max((len(value["participants"]) for value in state["targets"].values()), default=0)
            ),
        },
    }


def collect_formal_anchors(
    actor: torch.nn.Module,
    registered_seeds: Mapping[str, list[int]],
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected: dict[str, list[dict[str, Any]]] = {name: [] for name in ANCHOR_TARGETS}
    deterministic_states: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    episode_counts = {"mixed": 0, "pure_coverage": 0}
    start = time.monotonic()
    for scene in ("mixed", "pure_coverage"):
        target_classes = ("pursuing", "pre_capture_cover", "early_recovery") if scene == "mixed" else ("recovery_pure",)
        for episode_index, seed in enumerate(registered_seeds[scene]):
            if all(len(selected[name]) >= ANCHOR_TARGETS[name] for name in target_classes):
                break
            env_scene = "coverage" if scene == "pure_coverage" else "mixed"
            env, observations = make_env(env_scene, int(seed))
            apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
            capture_step = None
            local_counts = {name: 0 for name in target_classes}
            last_steps: dict[str, int] = {}
            episode_anchor_ids: list[int] = []
            episode_counts[scene] += 1
            for _ in range(env.episode_max_length):
                if not any(value is not None for value in observations):
                    break
                actions, policy_meta, _ = canonical_actions(actor, observations)
                state = mission_snapshot(env, observations)
                transition_step = int(env.episode_step + 1)
                if len(deterministic_states) < SNAPSHOT_STATES:
                    active = policy_meta["active"].tolist()
                    deterministic_states.append(
                        {
                            "bundle": save_env_state(env, observations, capture_step),
                            "scene": scene,
                            "seed": int(seed),
                            "episode_step": int(env.episode_step),
                            "focal": int(active[0]),
                        }
                    )
                for focal in policy_meta["active"].tolist():
                    phase, semantic_class, age = current_phase_class(
                        scene, observations, int(focal), capture_step, transition_step
                    )
                    if semantic_class not in target_classes:
                        continue
                    if len(selected[semantic_class]) >= ANCHOR_TARGETS[semantic_class]:
                        continue
                    if local_counts[semantic_class] >= MAX_ANCHORS_PER_CLASS_PER_EPISODE:
                        continue
                    previous_step = last_steps.get(semantic_class)
                    if previous_step is not None and transition_step - previous_step < MIN_SAME_CLASS_STEP_GAP:
                        continue
                    anchor = make_anchor(
                        env,
                        observations,
                        state,
                        actions,
                        scene,
                        int(seed),
                        int(episode_index),
                        int(focal),
                        phase,
                        semantic_class,
                        int(age),
                        capture_step,
                        len(episode_anchor_ids),
                    )
                    selected[semantic_class].append(anchor)
                    local_counts[semantic_class] += 1
                    last_steps[semantic_class] = transition_step
                    episode_anchor_ids.append(len(episode_anchor_ids))
                step = step_once(env, observations, actions, apf_agents, scene, capture_step)
                observations = step["observations"]
                capture_step = step["capture_step"]
                if bool(step["done"].all()):
                    break
            episode_stats.append(
                {
                    "scene": scene,
                    "seed": int(seed),
                    "episode_id": f"{scene}:{seed}",
                    "anchors_collected": int(len(episode_anchor_ids)),
                    "class_counts": {key: int(value) for key, value in local_counts.items()},
                    "class_cap": MAX_ANCHORS_PER_CLASS_PER_EPISODE,
                    "minimum_same_class_step_gap": MIN_SAME_CLASS_STEP_GAP,
                }
            )
            atomic_json(
                output / "progress.json",
                {
                    "status": "fresh_formal_anchor_rollout",
                    "scene": scene,
                    "episodes_completed": episode_counts,
                    "anchor_counts": {name: len(selected[name]) for name in ANCHOR_TARGETS},
                    "determinism_states": len(deterministic_states),
                    "elapsed_seconds": time.monotonic() - start,
                },
            )
        if not all(len(selected[name]) >= ANCHOR_TARGETS[name] for name in target_classes):
            missing = {name: ANCHOR_TARGETS[name] - len(selected[name]) for name in target_classes}
            raise RuntimeError(f"ANCHOR_COMPOSITION_FAIL: {missing}")
    anchors: list[dict[str, Any]] = []
    for name in ANCHOR_ORDER:
        anchors.extend(selected[name][: ANCHOR_TARGETS[name]])
    for anchor_id, anchor in enumerate(anchors):
        anchor["anchor_id"] = int(anchor_id)
    return anchors, deterministic_states, {
        "episodes": episode_counts,
        "episode_stats": episode_stats,
        "anchor_composition": {name: int(sum(row["semantic_class"] == name for row in anchors)) for name in ANCHOR_TARGETS},
        "elapsed_seconds": time.monotonic() - start,
    }


def snapshot_gate(states: list[dict[str, Any]], actor: torch.nn.Module) -> dict[str, Any]:
    if len(states) < SNAPSHOT_STATES:
        raise RuntimeError("SNAPSHOT_RESTORE_FAIL: fewer than five fresh states")
    rows = []
    max_obs_error = 0.0
    max_reward_error = 0.0
    for index, state in enumerate(states[:SNAPSHOT_STATES]):
        first = canonical_path_from_state(
            state["bundle"], actor, state["scene"], state["focal"], max_steps=1 + SNAPSHOT_CONTINUATION_STEPS
        )
        second = canonical_path_from_state(
            state["bundle"], actor, state["scene"], state["focal"], max_steps=1 + SNAPSHOT_CONTINUATION_STEPS
        )
        if len(first["trace"]) != len(second["trace"]) or len(first["trace"]) < 1 + SNAPSHOT_CONTINUATION_STEPS:
            raise RuntimeError(f"SNAPSHOT_RESTORE_FAIL: state {index} ended too early")
        obs_error = 0.0
        reward_error = 0.0
        event_phase_done_match = True
        for left, right in zip(first["trace"], second["trace"]):
            obs_error = max(obs_error, compare_observations(left["observations"], right["observations"]))
            reward_error = max(reward_error, float(np.max(np.abs(left["reward"] - right["reward"]), initial=0.0)))
            event_phase_done_match = event_phase_done_match and left["event_signature"] == right["event_signature"]
            event_phase_done_match = event_phase_done_match and left["phase"] == right["phase"]
            event_phase_done_match = event_phase_done_match and np.array_equal(left["done"], right["done"])
            event_phase_done_match = event_phase_done_match and np.array_equal(left["truncated"], right["truncated"])
        max_obs_error = max(max_obs_error, obs_error)
        max_reward_error = max(max_reward_error, reward_error)
        if obs_error > 1e-7 or reward_error > 1e-9 or not event_phase_done_match:
            raise RuntimeError(f"SNAPSHOT_RESTORE_FAIL: state {index} mismatch")
        rows.append(
            {
                "state_index": index,
                "scene": state["scene"],
                "seed": state["seed"],
                "episode_step": state["episode_step"],
                "focal_agent": state["focal"],
                "steps_compared": len(first["trace"]),
                "max_observation_abs_error": obs_error,
                "max_reward_abs_error": reward_error,
                "event_phase_done_truncation_match": True,
            }
        )
        del first, second
    return {
        "status": "PASS",
        "states_checked": len(rows),
        "continuation_steps_after_first": SNAPSHOT_CONTINUATION_STEPS,
        "max_observation_abs_error": max_obs_error,
        "max_reward_abs_error": max_reward_error,
        "max_error": max(max_obs_error, max_reward_error),
        "states": rows,
        "snapshot_contents": [
            "deepcopied env object",
            "observations",
            "env-owned RNG",
            "python RNG",
            "numpy RNG",
            "torch RNG",
        ],
    }


def compact_branch(branch: Mapping[str, Any], anchor: Mapping[str, Any], action: int) -> dict[str, Any]:
    return {
        "candidate_action": int(action),
        "empirical_return": float(branch["return"]),
        "collision": bool(branch["collision"]),
        "captured": bool(branch["captured"]),
        "normal_capture": bool(branch["normal_capture"]),
        "stationary_capture": bool(branch["stationary_capture"]),
        "ce_success": bool(branch["ce_success"]),
        "safe_complete": bool(branch["safe_complete"]),
        "length": int(branch["length"]),
        "terminal_reason": list(branch["terminal_reason"]),
        "extreme_delay_threshold": int(math.ceil(EXTREME_DELAY_FRACTION * anchor["episode_horizon"])),
    }


def run_empirical_branches(
    anchors: list[dict[str, Any]], actor: torch.nn.Module, output: Path
) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
    returns = np.zeros((len(anchors), 9), dtype=np.float64)
    outcomes: list[list[dict[str, Any]]] = []
    for index, anchor in enumerate(anchors):
        anchor_outcomes = []
        for action in range(9):
            branch = canonical_path_from_state(
                anchor["bundle"], actor, anchor["scene"], int(anchor["focal"]), forced_action=action
            )
            returns[index, action] = float(branch["return"])
            anchor_outcomes.append(compact_branch(branch, anchor, action))
            del branch
        outcomes.append(anchor_outcomes)
        atomic_json(
            output / "progress.json",
            {"status": "empirical_counterfactual_branching", "anchors_completed": index + 1, "anchors_total": len(anchors)},
        )
    return returns, outcomes


def check_return_noise(anchors: list[dict[str, Any]], actor: torch.nn.Module) -> dict[str, Any]:
    comparisons = []
    for offset in NOISE_CHECK_ANCHOR_OFFSETS:
        anchor = anchors[offset]
        for action in NOISE_CHECK_ACTIONS:
            first = canonical_path_from_state(
                anchor["bundle"], actor, anchor["scene"], int(anchor["focal"]), forced_action=action
            )
            second = canonical_path_from_state(
                anchor["bundle"], actor, anchor["scene"], int(anchor["focal"]), forced_action=action
            )
            difference = abs(float(first["return"]) - float(second["return"]))
            comparisons.append(
                {
                    "anchor_id": int(anchor["anchor_id"]),
                    "action": int(action),
                    "abs_return_difference": float(difference),
                }
            )
            del first, second
    maximum = max((row["abs_return_difference"] for row in comparisons), default=0.0)
    return {
        "method": "repeat same frozen snapshot/action twice before critic inference",
        "comparisons": len(comparisons),
        "max_abs_return_difference": float(maximum),
        "mean_abs_return_difference": float(np.mean([row["abs_return_difference"] for row in comparisons])) if comparisons else 0.0,
        "fixed_tolerance": RETURN_TOLERANCE,
        "tolerance_registered_before_critic_inference": True,
        "within_tolerance": bool(maximum <= RETURN_TOLERANCE),
    }


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


def empirical_spread(returns: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for index, values in enumerate(returns):
        order = np.argsort(-values, kind="mergesort")
        rows.append(
            {
                "anchor_id": int(index),
                "best_action": int(order[0]),
                "second_action": int(order[1]),
                "best_minus_second": float(values[order[0]] - values[order[1]]),
                "best_minus_worst": float(values[order[0]] - values[order[-1]]),
                "return_std_across_actions": float(np.std(values)),
                "near_tie_best_second": bool(values[order[0]] - values[order[1]] <= RETURN_TOLERANCE),
            }
        )
    return rows


def ranking_row(prediction: np.ndarray, empirical: np.ndarray, bc_action: int) -> dict[str, Any]:
    from tools.run_ac4a_counterfactual_aw9_pilot_20260920 import ranking_metrics

    row = ranking_metrics(prediction, empirical, bc_action)
    alternatives = np.asarray([index for index in range(9) if index != int(bc_action)], dtype=np.int64)
    empirical_delta = empirical[alternatives] - empirical[int(bc_action)]
    predicted_delta = prediction[alternatives] - prediction[int(bc_action)]
    keep = np.abs(empirical_delta) > RETURN_TOLERANCE
    row["margin_filtered_sign_accuracy"] = (
        float(np.mean(np.sign(empirical_delta[keep]) == np.sign(predicted_delta[keep]))) if np.any(keep) else None
    )
    row["margin_filtered_pair_count"] = int(np.sum(keep))
    row["margin_filter_tolerance"] = RETURN_TOLERANCE
    return row


def aggregate_ranking(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "top1_agreement",
        "top3_overlap",
        "spearman",
        "regret",
        "bc_action_empirical_rank",
        "sign_accuracy_vs_bc",
        "margin_filtered_sign_accuracy",
    )
    result: dict[str, Any] = {}
    for key in keys:
        result[key] = describe([float(row[key]) for row in rows if row[key] is not None])
    result["top1_count"] = int(sum(bool(row["top1_agreement"]) for row in rows))
    result["count"] = len(rows)
    result["margin_filtered_pair_count"] = int(sum(row["margin_filtered_pair_count"] for row in rows))
    return result


def independence_stats(anchors: list[dict[str, Any]], collection: Mapping[str, Any]) -> dict[str, Any]:
    by_class: dict[str, list[dict[str, Any]]] = {name: [] for name in ANCHOR_ORDER}
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for anchor in anchors:
        by_class[anchor["semantic_class"]].append(anchor)
        by_episode.setdefault(anchor["episode_id"], []).append(anchor)
    class_stats = {}
    for name, rows in by_class.items():
        episode_groups: dict[str, list[int]] = {}
        for row in rows:
            episode_groups.setdefault(row["episode_id"], []).append(int(row["transition_step"]))
        gaps = []
        max_per_episode = 0
        for steps in episode_groups.values():
            steps.sort()
            max_per_episode = max(max_per_episode, len(steps))
            gaps.extend([right - left for left, right in zip(steps, steps[1:])])
        class_stats[name] = {
            "anchors": len(rows),
            "unique_episodes": len(episode_groups),
            "max_same_class_per_episode": int(max_per_episode),
            "minimum_same_class_step_gap_observed": int(min(gaps)) if gaps else None,
            "gap_values": [int(value) for value in gaps],
        }
    counts = {}
    for episode_id, rows in by_episode.items():
        counts[episode_id] = len(rows)
    return {
        "unique_anchor_episodes": len(by_episode),
        "anchors_per_episode": describe(list(counts.values())),
        "max_anchors_per_episode": int(max(counts.values(), default=0)),
        "class_stats": class_stats,
        "same_class_cap": MAX_ANCHORS_PER_CLASS_PER_EPISODE,
        "same_class_minimum_gap_registered": MIN_SAME_CLASS_STEP_GAP,
        "episode_collection_stats": collection["episode_stats"],
    }


def outcome_counts(anchors: list[dict[str, Any]], branch_outcomes: list[list[dict[str, Any]]]) -> dict[str, Any]:
    rows = []
    for anchor, outcomes in zip(anchors, branch_outcomes):
        for outcome in outcomes:
            row = dict(outcome)
            row["scene"] = anchor["scene"]
            row["semantic_class"] = anchor["semantic_class"]
            row["anchor_id"] = anchor["anchor_id"]
            row["capture_failure"] = bool(anchor["scene"] == "mixed" and not row["captured"])
            row["ce_failure"] = bool(
                (anchor["scene"] == "mixed" and row["captured"] and not row["ce_success"])
                or (anchor["scene"] == "pure_coverage" and not row["ce_success"])
            )
            row["unsafe_incomplete"] = bool(not row["safe_complete"])
            row["extreme_delay"] = bool(row["length"] >= row["extreme_delay_threshold"])
            row["catastrophic"] = bool(
                row["collision"] or row["unsafe_incomplete"] or row["capture_failure"] or row["ce_failure"] or row["extreme_delay"]
            )
            rows.append(row)
    def count(key: str, subset: list[dict[str, Any]] | None = None) -> int:
        return int(sum(bool(row[key]) for row in (rows if subset is None else subset)))
    mixed = [row for row in rows if row["scene"] == "mixed"]
    pure = [row for row in rows if row["scene"] == "pure_coverage"]
    return {
        "total_branches": len(rows),
        "collision": {"count": count("collision"), "denominator": len(rows)},
        "unsafe_incomplete": {"count": count("unsafe_incomplete"), "denominator": len(rows)},
        "capture_failure": {"count": count("capture_failure"), "denominator": len(mixed), "scope": "mixed only"},
        "ce_failure": {"count": count("ce_failure"), "denominator": len(rows), "scope": "mixed captured or pure coverage"},
        "extreme_delay": {"count": count("extreme_delay"), "denominator": len(rows), "threshold": f">={EXTREME_DELAY_FRACTION:.0%} of episode horizon"},
        "catastrophic_unique": {"count": count("catastrophic"), "denominator": len(rows)},
        "by_scene": {
            "mixed": {"branches": len(mixed), "catastrophic": count("catastrophic", mixed)},
            "pure_coverage": {"branches": len(pure), "catastrophic": count("catastrophic", pure)},
        },
        "rows": rows,
    }


def bc_quality(anchors: list[dict[str, Any]], returns: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    spread = empirical_spread(returns)
    for index, anchor in enumerate(anchors):
        values = returns[index]
        order = np.argsort(-values, kind="mergesort")
        bc_action = int(anchor["original_bc_action"])
        rows.append(
            {
                "anchor_id": int(anchor["anchor_id"]),
                "semantic_class": anchor["semantic_class"],
                "empirical_rank": int(np.flatnonzero(order == bc_action)[0]) + 1,
                "top1": bool(order[0] == bc_action),
                "top3": bool(bc_action in order[:3]),
                "regret": float(values[order[0]] - values[bc_action]),
            }
        )
    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "top1": describe([float(row["top1"]) for row in subset]),
            "top3": describe([float(row["top3"]) for row in subset]),
            "empirical_rank": describe([float(row["empirical_rank"]) for row in subset]),
            "regret": describe([float(row["regret"]) for row in subset]),
            "count": len(subset),
        }
    phase = {name: summarize([row for row in rows if row["semantic_class"] == name]) for name in ANCHOR_ORDER}
    spread_summary = {
        "best_minus_second": describe([row["best_minus_second"] for row in spread]),
        "best_minus_worst": describe([row["best_minus_worst"] for row in spread]),
        "return_std_across_actions": describe([row["return_std_across_actions"] for row in spread]),
        "near_tie_best_second_count": int(sum(row["near_tie_best_second"] for row in spread)),
        "near_tie_best_second_fraction": float(np.mean([row["near_tie_best_second"] for row in spread])),
        "per_anchor": spread,
    }
    return {"overall": summarize(rows), "by_phase": phase, "per_anchor": rows}, spread_summary


def run_critic_rankings(
    anchors: list[dict[str, Any]],
    returns: np.ndarray,
    models: Mapping[str, torch.nn.Module],
    mean: float,
    scale: float,
    device: torch.device,
    output: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    predictions = {kind: np.zeros((len(anchors), 9), dtype=np.float64) for kind in models}
    rankings: dict[str, Any] = {}
    for index, anchor in enumerate(anchors):
        predicted = critic_predictions(models, anchor, mean, scale, device)
        for kind in models:
            predictions[kind][index] = predicted[kind]
        atomic_json(output / "progress.json", {"status": "critic_counterfactual_inference", "anchors_completed": index + 1, "anchors_total": len(anchors)})
    for kind in models:
        rows = [
            ranking_row(predictions[kind][index], returns[index], int(anchor["original_bc_action"]))
            for index, anchor in enumerate(anchors)
        ]
        rankings[kind] = {
            "overall": aggregate_ranking(rows),
            "per_anchor": rows,
            "by_phase": {
                phase: aggregate_ranking([row for row, anchor in zip(rows, anchors) if anchor["semantic_class"] == phase])
                for phase in ANCHOR_ORDER
            },
        }
    return rankings, predictions


def replication_status(ranking: Mapping[str, Any]) -> dict[str, Any]:
    pilot = json.loads(AC4A_SUMMARY.read_text())["ranking"]
    rows = {}
    for kind in ("LQ", "NQ", "CQ"):
        pilot_overall = pilot[kind]["overall"]
        current = ranking[kind]["overall"]
        pilot_spearman = pilot_overall["spearman"]["mean"]
        current_spearman = current["spearman"]["mean"]
        pilot_sign = pilot_overall["sign_accuracy_vs_bc"]["mean"]
        current_sign = current["sign_accuracy_vs_bc"]["mean"]
        current_filtered = current["margin_filtered_sign_accuracy"]["mean"]
        positive_direction = bool(current_spearman > 0.0 and current_sign > 0.5 and (current_filtered is None or current_filtered > 0.5))
        rows[kind] = {
            "ac4a": {"spearman": pilot_spearman, "sign_accuracy_vs_bc": pilot_sign},
            "ac4b": {"spearman": current_spearman, "sign_accuracy_vs_bc": current_sign, "margin_filtered_sign_accuracy": current_filtered},
            "direction_status": "positive_direction_replicated" if positive_direction else "signal_not_replicated_or_weak",
        }
    strong_pilot_signal_preserved = all(rows[kind]["direction_status"] == "positive_direction_replicated" for kind in ("LQ", "NQ"))
    return {
        "overall_status": "AC4A_SIGNAL_DIRECTION_REPLICATED" if strong_pilot_signal_preserved else "AC4A_SMALL_SAMPLE_SIGNAL_NOT_REPLICATED",
        "per_critic": rows,
        "interpretation_rule": "positive direction requires overall Spearman > 0, raw BC sign accuracy > 0.5, and filtered sign accuracy > 0.5 when defined",
    }


def classification(ranking: Mapping[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for kind in ("LQ", "NQ", "CQ"):
        overall = ranking[kind]["overall"]
        spearman = overall["spearman"]["mean"]
        filtered = overall["margin_filtered_sign_accuracy"]["mean"]
        sign = overall["sign_accuracy_vs_bc"]["mean"]
        phase_spearman = [ranking[kind]["by_phase"][phase]["spearman"]["mean"] for phase in ANCHOR_ORDER]
        if spearman > 0 and sign > 0.5 and (filtered is None or filtered > 0.5):
            labels[kind] = "PHASE_DEPENDENT_RANKING" if any(value <= 0 for value in phase_spearman) else f"{kind}_RANKING_CONFIRMED"
        elif spearman > 0 or sign > 0.5:
            labels[kind] = "WEAK_RANKING_SIGNAL"
        else:
            labels[kind] = "NO_RANKING_SIGNAL"
    return labels


def manifest_anchor(anchor: Mapping[str, Any], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    focal = int(anchor["focal"])
    return {
        "anchor_id": int(anchor["anchor_id"]),
        "scene": anchor["scene"],
        "episode_id": anchor["episode_id"],
        "seed": int(anchor["seed"]),
        "episode_index": int(anchor["episode_index"]),
        "episode_step": int(anchor["episode_step"]),
        "transition_step": int(anchor["transition_step"]),
        "episode_horizon": int(anchor["episode_horizon"]),
        "focal_agent_id": focal,
        "semantic_class": anchor["semantic_class"],
        "phase": anchor["phase"],
        "recovery_age": int(anchor["recovery_age"]),
        "capture_step": anchor["capture_step"],
        "within_episode_index": int(anchor["within_episode_index"]),
        "event_labels_at_anchor": json_safe(anchor["event_labels_at_anchor"]),
        "original_bc_action": int(anchor["original_bc_action"]),
        "local_observation": json_safe({key: np.asarray(value)[focal] for key, value in anchor["local_full"].items()}),
        "neighbor_ids": np.asarray(anchor["neighbor_ids"])[focal].astype(int).tolist(),
        "neighbor_mask": np.asarray(anchor["neighbor_mask"])[focal].astype(bool).tolist(),
        "central_state_metadata": json_safe(anchor["central"]),
        "joint_bc_action_indices": np.asarray(anchor["joint_action_indices"]).astype(int).tolist(),
        "branch_outcomes": json_safe(outcomes),
        "snapshot_saved_to_disk": False,
        "full_trajectory_saved_to_disk": False,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    before_disk = shutil.disk_usage(ROOT)
    registered = seed_manifest()
    atomic_json(output / "seed_manifest.json", registered)
    atomic_json(output / "progress.json", {"status": "seed_registered", "seed_manifest": str(output / "seed_manifest.json")})

    actual_actor_sha = sha256_file(ACTOR_LOADED)
    if actual_actor_sha != ACTOR_SHA256:
        raise RuntimeError("BC_CHECKPOINT_IDENTITY_FAIL")
    actor, _ = load_actor(ACTOR_LOADED, "cpu")
    actor.eval()
    actor.requires_grad_(False)
    actor_hash_before = tensor_state_sha256(actor.state_dict())
    if not np.array_equal(np.asarray(actor.action_grid.cpu(), dtype=np.float32), ACTION_GRID):
        raise RuntimeError("ACTION_MAPPING_FAIL")
    action_mapping = {
        "status": "MATCH",
        "indices": list(range(9)),
        "grid": ACTION_GRID.astype(float).tolist(),
        "ordering": "[(a,w) for a in (-0.4,0,0.4) for w in (-pi/6,0,pi/6)]",
    }

    anchors, deterministic_states, collection = collect_formal_anchors(
        actor,
        {"mixed": list(MIXED_SEEDS), "pure_coverage": list(PURE_SEEDS)},
        output,
    )
    gate = snapshot_gate(deterministic_states, actor)
    returns, branch_outcomes = run_empirical_branches(anchors, actor, output)
    noise = check_return_noise(anchors, actor)
    if not noise["within_tolerance"]:
        raise RuntimeError("EMPIRICAL_RETURN_NUMERICAL_NOISE_FAIL")
    spread_rows = empirical_spread(returns)
    bc_report, spread_report = bc_quality(anchors, returns)

    device = torch.device(args.device)
    models, critic_provenance = load_critics(device)
    for kind, provenance in critic_provenance.items():
        if provenance["seed"] != EXPECTED_CRITIC_SEED[kind] or provenance["sha256"] != EXPECTED_CRITIC_SHA[kind]:
            raise RuntimeError(f"CRITIC_CHECKPOINT_IDENTITY_FAIL_{kind}")
    critic_hash_before = {kind: provenance["state_sha256_before"] for kind, provenance in critic_provenance.items()}
    ac3b = json.loads(AC3B_SUMMARY.read_text())
    mean = float(ac3b["contract"]["normalization"]["train_mean"])
    scale = float(ac3b["contract"]["normalization"]["train_std"])
    ranking, prediction_matrix = run_critic_rankings(anchors, returns, models, mean, scale, device, output)

    arrays = {
        "anchor_ids": np.asarray([anchor["anchor_id"] for anchor in anchors], dtype=np.int64),
        "returns": returns.astype(np.float64),
        "action_grid": ACTION_GRID.astype(np.float32),
        "bc_actions": np.asarray([anchor["original_bc_action"] for anchor in anchors], dtype=np.int64),
        "semantic_class": np.asarray([anchor["semantic_class"] for anchor in anchors]),
    }
    for kind, values in prediction_matrix.items():
        arrays[f"{kind.lower()}_predictions"] = values.astype(np.float64)
    npz_path = output / "counterfactual_returns.npz"
    np.savez_compressed(npz_path, **arrays)
    atomic_json(
        output / "anchor_manifest.json",
        {
            "schema": "ac4b-anchor-manifest-v1",
            "status": "complete",
            "seed_registration": registered,
            "anchor_composition": {name: int(sum(anchor["semantic_class"] == name for anchor in anchors)) for name in ANCHOR_ORDER},
            "anchors": [manifest_anchor(anchor, branch_outcomes[index]) for index, anchor in enumerate(anchors)],
        },
    )

    outcomes = outcome_counts(anchors, branch_outcomes)
    independence = independence_stats(anchors, collection)
    actor_hash_after = tensor_state_sha256(actor.state_dict())
    critic_hash_after = {kind: tensor_state_sha256(model.state_dict()) for kind, model in models.items()}
    replication = replication_status(ranking)
    labels = classification(ranking)
    after_disk = shutil.disk_usage(ROOT)
    summary: dict[str, Any] = {
        "schema": "ac4b-formal-counterfactual-aw9-ranking-v1",
        "status": "complete",
        "git_head_at_run": git_head(),
        "contract": {
            "environment_contract": "forward-final-aw9-4v1-swept-v1",
            "bc_checkpoint_requested": str(ACTOR_REQUESTED),
            "bc_checkpoint_loaded": str(ACTOR_LOADED),
            "bc_checkpoint_sha256": actual_actor_sha,
            "gamma": GAMMA,
            "return": "individual focal reward full continuation, no bootstrap/V/learned-Q tail",
            "branch": "teammates use anchor BC argmax for first step; only focal action 0..8 is forced; all agents use BC argmax from t+1",
            "target_normalization": {"train_mean": mean, "train_std": scale, "source": str(AC3B_SUMMARY)},
            "no_full_trajectory_saved": True,
        },
        "action_mapping": action_mapping,
        "snapshot_gate": gate,
        "seed_manifest": registered,
        "anchor_collection": collection,
        "anchor_composition": {name: int(sum(anchor["semantic_class"] == name for anchor in anchors)) for name in ANCHOR_ORDER},
        "anchor_independence": independence,
        "branches": {"anchors": len(anchors), "actions_per_anchor": 9, "total": len(anchors) * 9},
        "empirical_return_noise": noise,
        "empirical_action_spread": spread_report,
        "empirical_action_spread_per_anchor": spread_rows,
        "bc_action_quality": bc_report,
        "critic_provenance": critic_provenance,
        "ranking": ranking,
        "margin_filter": {
            "tolerance": RETURN_TOLERANCE,
            "selection": "absolute empirical alternative-vs-BC return difference > fixed tolerance",
            "registered_before_critic_inference": True,
        },
        "outcomes": outcomes,
        "ac4a_to_ac4b": replication,
        "classification": labels,
        "forbidden_executed": {
            "actor_optimizer_steps": 0,
            "critic_optimizer_steps": 0,
            "bootstrap": False,
            "GAE": False,
            "PPO": False,
            "TD3": False,
            "MADDPG": False,
            "SAC": False,
            "actor_training": False,
            "critic_training": False,
            "environment_contract_change": False,
            "reward_change": False,
        },
        "state_hashes": {
            "actor_before": actor_hash_before,
            "actor_after": actor_hash_after,
            "actor_bit_exact": actor_hash_before == actor_hash_after,
            "critics_before": critic_hash_before,
            "critics_after": critic_hash_after,
            "critics_bit_exact": critic_hash_before == critic_hash_after,
        },
        "artifacts": {
            "seed_manifest": str(output / "seed_manifest.json"),
            "anchor_manifest": str(output / "anchor_manifest.json"),
            "counterfactual_returns": str(npz_path),
            "counterfactual_returns_sha256": sha256_file(npz_path),
            "ac4a_summary_sha256": sha256_file(AC4A_SUMMARY),
            "ac3a_summary_sha256": sha256_file(AC3A_SUMMARY),
            "ac3b_summary_sha256": sha256_file(AC3B_SUMMARY),
        },
        "storage": {
            "disk_free_bytes_before": int(before_disk.free),
            "disk_free_bytes_after": int(after_disk.free),
            "output_bytes": 0,
        },
    }
    atomic_json(output / "summary.json", summary)
    summary["storage"]["output_bytes"] = int(sum(path.stat().st_size for path in output.rglob("*") if path.is_file()))
    summary["storage"]["disk_free_bytes_after"] = int(shutil.disk_usage(ROOT).free)
    atomic_json(output / "summary.json", summary)
    summary["storage"]["output_bytes"] = int(sum(path.stat().st_size for path in output.rglob("*") if path.is_file()))
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "progress.json", {"status": "complete", "anchors": len(anchors), "branches": len(anchors) * 9})
    for model in models.values():
        del model
    del actor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": "complete",
                "snapshot_gate": gate["status"],
                "anchors": len(anchors),
                "branches": len(anchors) * 9,
                "catastrophic_unique": outcomes["catastrophic_unique"]["count"],
                "output_bytes": summary["storage"]["output_bytes"],
                "replication": replication["overall_status"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
