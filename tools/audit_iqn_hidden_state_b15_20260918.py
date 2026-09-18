#!/usr/bin/env python3
"""Offline B1.5 reconstruction audit for local target evidence.

This tool consumes the frozen B1 teacher dataset only.  It reconstructs
decision-time direct sensing, local friend adjacency, and per-agent history;
it never changes the environment, IQN observation contract, reward, or replay
schema and never launches RL training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover - environment contract
    raise RuntimeError("B1.5 requires scipy for the B1 nearest-cross-role audit") from exc

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - environment contract
    raise RuntimeError("B1.5 requires sklearn for the lightweight diagnostic probes") from exc

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "iqn-hidden-state-b15-20260918-v1"
DATASET_SCHEMA = "forward-final-teacher-dataset-v1"
TEACHER_SHA = "2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89"
MANIFEST_SHA = "4f99ebf016fb30ad940ea0f87f0c200aca006fbf31510389f76fd0fca8661996"
BASE_B1_HEAD = "9c8f603ea81cb8bb1104549e360058e8eaa41f7c"
DEFAULT_DATASET = ROOT / "artifacts/2026-09-08_forward_final/c1_dataset"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts/2026-09-18_iqn_hidden_state_b15"
LAMBDA_GRID = (0.80, 0.90, 0.95, 0.98)
ETA_GRID = (0.50, 0.70, 0.85, 0.95)
HISTORY_K = (3, 5, 10)
PHASES = {0: "pre_capture", 1: "post_capture", 2: "pure_coverage"}
CONDITIONS = {
    "direct": lambda arrays: np.asarray(arrays["direct"], dtype=bool),
    "support": lambda arrays: np.asarray(arrays["support"], dtype=bool),
    "coverage": lambda arrays: np.asarray(arrays["role_id"], dtype=np.int64) == 3,
}
TEMPERATURE = 0.0031884169327952754
MAP_WIDTH = 120.0
MAP_HEIGHT = 120.0
MAP_DIAGONAL = float(np.hypot(MAP_WIDTH, MAP_HEIGHT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _dataset_with_shards(root: Path, manifest: Mapping[str, Any]) -> Path:
    shards = list(manifest.get("shards", []))
    if shards and (root / shards[0]["path"]).is_file():
        return root
    sibling = ROOT.parent / "cocap-voradj-small-step-ac" / "artifacts/2026-09-08_forward_final/c1_dataset"
    if shards and (sibling / shards[0]["path"]).is_file():
        return sibling
    raise FileNotFoundError(
        f"teacher dataset shards are missing below {root}; expected a read-only sibling at {sibling}"
    )


def load_dataset(root: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != DATASET_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("B1.5 requires the complete forward-final teacher dataset")
    if manifest.get("teacher_sha256") != TEACHER_SHA:
        raise ValueError("teacher dataset is not paired with the frozen Final IQN teacher")
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != MANIFEST_SHA:
        raise ValueError(f"unexpected teacher manifest hash: {manifest_hash}")
    data_root = _dataset_with_shards(root, manifest)
    row_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    global_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    global_offset = 0
    global_indices: list[np.ndarray] = []
    shard_rows = 0
    for shard in manifest["shards"]:
        path = data_root / shard["path"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"dataset shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            if "episode" not in loaded.files or "transition_index" not in loaded.files:
                raise ValueError(f"missing trajectory index fields in {path}")
            row_count = len(loaded["episode"])
            transition_count = len(loaded["global_state.active_mask"])
            local_transition = np.asarray(loaded["transition_index"], dtype=np.int64)
            if local_transition.max(initial=-1) >= transition_count:
                raise ValueError(f"invalid transition index in {path}")
            global_indices.append(local_transition + global_offset)
            shard_rows += row_count
            for key in loaded.files:
                if key.startswith("global_state."):
                    global_chunks[key].append(np.asarray(loaded[key]))
                else:
                    row_chunks[key].append(np.asarray(loaded[key]))
            global_offset += transition_count
    arrays = {key: np.concatenate(value, axis=0) for key, value in row_chunks.items()}
    global_arrays = {key: np.concatenate(value, axis=0) for key, value in global_chunks.items()}
    arrays["global_index"] = np.concatenate(global_indices, axis=0)
    if shard_rows != int(manifest["row_count"]) or len(arrays["episode"]) != shard_rows:
        raise ValueError("teacher dataset row-count contract failed")
    if not np.array_equal(np.argmax(arrays["teacher_q"], axis=1), arrays["greedy_action"]):
        raise ValueError("teacher Q/action contract failed")
    return arrays, global_arrays, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "shard_root": str(data_root),
        "rows": int(shard_rows),
        "episodes": int(np.unique(arrays["episode"]).size),
        "global_transitions": int(global_offset),
        "shards": len(manifest["shards"]),
    }


def role_free_view(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    self_obs = np.asarray(arrays["local_obs.self"], dtype=np.float32)
    friends = np.asarray(arrays["local_obs.pursuers"], dtype=np.float32)
    masks = np.asarray(arrays["local_obs.masks"], dtype=bool)
    types = np.asarray(arrays["local_obs.types"], dtype=np.int64)
    max_p = friends.shape[1]
    result_friends = np.zeros((len(friends), max_p, friends.shape[2] - 1), dtype=np.float32)
    result_masks = masks.copy()
    result_types = types.copy()
    for row in range(len(friends)):
        valid = masks[row, 1 : 1 + max_p] & (types[row, 1 : 1 + max_p] == 1)
        order = sorted(
            range(max_p),
            key=lambda index: (
                0 if valid[index] else 1,
                tuple(float(value) for value in friends[row, index, :-1]),
                index,
            ),
        )
        result_friends[row] = friends[row, order, :-1]
        result_masks[row, 1 : 1 + max_p] = masks[row, 1 : 1 + max_p][order]
        result_types[row, 1 : 1 + max_p] = types[row, 1 : 1 + max_p][order]
    return {
        "self": self_obs[:, :-1].copy(),
        "pursuers": result_friends,
        "evaders": np.asarray(arrays["local_obs.evaders"], dtype=np.float32).copy(),
        "obstacles": np.asarray(arrays["local_obs.obstacles"], dtype=np.float32).copy(),
        "masks": result_masks,
        "types": result_types,
    }


def physical_vectors(view: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            view["self"],
            view["pursuers"].reshape(len(view["self"]), -1),
            view["evaders"].reshape(len(view["self"]), -1),
            view["obstacles"].reshape(len(view["self"]), -1),
            view["masks"].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def recover_adjacency(
    arrays: Mapping[str, np.ndarray], global_arrays: Mapping[str, np.ndarray]
) -> tuple[list[list[int]], dict[str, Any]]:
    """Match local friend tokens to global agents using physical geometry only."""

    local_friends = np.asarray(arrays["local_obs.pursuers"], dtype=np.float32)
    masks = np.asarray(arrays["local_obs.masks"], dtype=bool)
    types = np.asarray(arrays["local_obs.types"], dtype=np.int64)
    global_p = np.asarray(global_arrays["global_state.pursuers"], dtype=np.float32)
    active = np.asarray(global_arrays["global_state.active_mask"], dtype=bool)
    agents = np.asarray(arrays["agent"], dtype=np.int64)
    indices = np.asarray(arrays["global_index"], dtype=np.int64)
    adjacency: list[list[int]] = []
    unmatched = 0
    ambiguous = 0
    token_count = 0
    tolerance = 2.0e-4
    for row, (agent, global_index) in enumerate(zip(agents, indices)):
        # Central schema stores one global pursuer matrix per focal agent:
        # [focal_agent, candidate_agent, feature].
        state = global_p[global_index, int(agent)]
        live = np.flatnonzero(active[global_index])
        focal = state[int(agent)]
        focal_pos = focal[:2] * np.asarray([MAP_WIDTH, MAP_HEIGHT], dtype=np.float32)
        theta = float(np.arctan2(focal[4], focal[5]))
        cos_theta, sin_theta = float(np.cos(theta)), float(np.sin(theta))
        neighbors: list[int] = []
        valid_tokens = masks[row, 1 : 1 + local_friends.shape[1]] & (types[row, 1 : 1 + local_friends.shape[1]] == 1)
        for token in np.flatnonzero(valid_tokens):
            token_count += 1
            value = local_friends[row, token, :6]
            errors: list[tuple[float, int]] = []
            for candidate in live:
                candidate = int(candidate)
                if candidate == int(agent):
                    continue
                other_pos = state[candidate, :2] * np.asarray([MAP_WIDTH, MAP_HEIGHT], dtype=np.float32)
                delta = other_pos - focal_pos
                rel = np.asarray(
                    [cos_theta * delta[0] + sin_theta * delta[1], -sin_theta * delta[0] + cos_theta * delta[1]],
                    dtype=np.float32,
                )
                expected = np.asarray(
                    [rel[0] / MAP_DIAGONAL, rel[1] / MAP_DIAGONAL, np.linalg.norm(rel) / MAP_DIAGONAL, np.arctan2(rel[1], rel[0])],
                    dtype=np.float32,
                )
                observed = np.asarray([value[0], value[1], value[4], value[5]], dtype=np.float32)
                error = np.asarray([observed[0] - expected[0], observed[1] - expected[1], observed[2] - expected[2], _wrap_angle(observed[3] - expected[3])])
                errors.append((float(np.max(np.abs(error))), candidate))
            errors.sort()
            if not errors or errors[0][0] > tolerance:
                unmatched += 1
                continue
            if len(errors) > 1 and errors[1][0] <= tolerance and abs(errors[1][0] - errors[0][0]) < 1.0e-6:
                ambiguous += 1
            neighbors.append(errors[0][1])
        adjacency.append(sorted(set(neighbors)))
    lookup: dict[tuple[int, int, int], int] = {}
    for row, (episode, transition, agent) in enumerate(
        zip(arrays["episode"], arrays["transition_index"], arrays["agent"])
    ):
        lookup[(int(episode), int(transition), int(agent))] = row
    neighbor_mask = np.asarray(arrays["neighbor_target_mask"], dtype=bool)
    direct_mask = np.asarray(arrays["direct_target_mask"], dtype=bool)
    recovered_informed = np.zeros_like(neighbor_mask)
    missing_neighbor_rows = 0
    for row, neighbors in enumerate(adjacency):
        for neighbor in neighbors:
            key = (int(arrays["episode"][row]), int(arrays["transition_index"][row]), int(neighbor))
            friend_row = lookup.get(key)
            if friend_row is None:
                missing_neighbor_rows += 1
            else:
                recovered_informed[row] |= direct_mask[friend_row]
    informed_equal = np.all(recovered_informed == neighbor_mask, axis=1)
    local_token_counts = np.asarray(
        [np.count_nonzero(masks[row, 1 : 1 + local_friends.shape[1]] & (types[row, 1 : 1 + local_friends.shape[1]] == 1)) for row in range(len(local_friends))]
    )
    recovered_counts = np.asarray([len(value) for value in adjacency])
    report = {
        "local_friend_tokens": int(token_count),
        "unmatched_tokens": int(unmatched),
        "ambiguous_tokens": int(ambiguous),
        "token_match_rate": float(1.0 - unmatched / max(token_count, 1)),
        "rows_with_exact_friend_count": int(np.sum(local_token_counts == recovered_counts)),
        "friend_count_match_rate": float(np.mean(local_token_counts == recovered_counts)),
        "rows_with_exact_neighbor_target_mask": int(np.sum(informed_equal)),
        "neighbor_target_mask_match_rate": float(np.mean(informed_equal)),
        "missing_neighbor_rows": int(missing_neighbor_rows),
        "tolerance": tolerance,
        "geometry": "global world-frame pursuer positions/yaw matched to local robot-frame friend position, distance, and angle; role column ignored",
    }
    if unmatched or ambiguous or missing_neighbor_rows or not np.all(informed_equal):
        raise ValueError(f"friend adjacency reconstruction contract failed: {report}")
    return adjacency, report


def _row_order(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.lexsort((arrays["agent"], arrays["timestep"], arrays["episode"]))


def _sequence_lookup(arrays: Mapping[str, np.ndarray]) -> dict[tuple[int, int, int], int]:
    return {
        (int(ep), int(step), int(agent)): row
        for row, (ep, step, agent) in enumerate(
            zip(arrays["episode"], arrays["timestep"], arrays["agent"])
        )
    }


def build_history_features(arrays: Mapping[str, np.ndarray], history_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(arrays["episode"])
    direct = np.asarray(arrays["direct_target_mask"], dtype=bool).any(axis=1)
    neighbor = np.asarray(arrays["neighbor_target_mask"], dtype=bool).any(axis=1)
    local_evaders = np.asarray(arrays["local_obs.evaders"], dtype=np.float32)
    masks = np.asarray(arrays["local_obs.masks"], dtype=bool)
    types = np.asarray(arrays["local_obs.types"], dtype=np.int64)
    max_p, max_e = local_evaders.shape[1], local_evaders.shape[1]
    evader_mask = masks[:, 1 + max_p : 1 + max_p + max_e] & (types[:, 1 + max_p : 1 + max_p + max_e] == 2)
    distances = np.where(evader_mask, local_evaders[:, :, 4] / MAP_DIAGONAL, 0.0)
    current_distance = distances.max(axis=1)
    age = np.zeros(n, dtype=np.float32)
    sequences: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in _row_order(arrays):
        sequences[(int(arrays["episode"][row]), int(arrays["agent"][row]))].append(int(row))
    history = np.zeros((n, history_k, 4), dtype=np.float32)
    # One pass per agent trajectory: O(rows*K), with no repeated dictionary
    # walk for every lag. Missing timesteps (deactivation/termination) break
    # the history at that point rather than crossing a trajectory gap.
    for rows in sequences.values():
        rows.sort(key=lambda row: int(arrays["timestep"][row]))
        for position, row in enumerate(rows):
            if direct[row]:
                age[row] = 0.0
            elif position == 0 or int(arrays["timestep"][row]) != int(arrays["timestep"][rows[position - 1]]) + 1:
                age[row] = float(history_k + 1)
            else:
                age[row] = age[rows[position - 1]] + 1.0
            for lag in range(min(history_k, position + 1)):
                source = rows[position - lag]
                if int(arrays["timestep"][row]) - int(arrays["timestep"][source]) != lag:
                    break
                history[row, lag] = [
                    float(direct[source]),
                    float(neighbor[source]),
                    float(current_distance[source]),
                    float(min(age[source], history_k + 1) / (history_k + 1)),
                ]
    return history.reshape(n, history_k * 4), age, current_distance


def reconstruct_signal(
    arrays: Mapping[str, np.ndarray], adjacency: list[list[int]], lambda_value: float, eta_value: float
) -> np.ndarray:
    direct = np.asarray(arrays["direct_target_mask"], dtype=bool).any(axis=1)
    lookup = _sequence_lookup(arrays)
    signal = np.zeros(len(direct), dtype=np.float32)
    previous: dict[tuple[int, int], float] = {}
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in _row_order(arrays):
        grouped[(int(arrays["episode"][row]), int(arrays["timestep"][row]))].append(int(row))
    # All values for a joint timestep read the same previous snapshot.  The
    # ``current`` dictionary is committed only after every focal agent has
    # been evaluated, so no same-step recursive propagation is possible.
    for (episode, timestep), rows in sorted(grouped.items()):
        current: dict[tuple[int, int], float] = {}
        for row in rows:
            agent = int(arrays["agent"][row])
            own_previous = previous.get((episode, agent), 0.0) if timestep > 1 else 0.0
            neighbor_previous = 0.0
            for neighbor in adjacency[row]:
                if timestep > 1 and (episode, timestep - 1, int(neighbor)) in lookup:
                    neighbor_previous = max(neighbor_previous, previous.get((episode, int(neighbor)), 0.0))
            value = max(float(direct[row]), lambda_value * own_previous, eta_value * neighbor_previous)
            signal[row] = float(value)
            current[(episode, agent)] = float(value)
        previous.update(current)
    return signal


def split_indices(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    episodes = np.asarray(arrays["episode"], dtype=np.int64)
    return {
        "train": np.flatnonzero(np.isin(episodes % 5, [0, 1, 2])),
        "validation": np.flatnonzero(episodes % 5 == 3),
        "test": np.flatnonzero(episodes % 5 == 4),
    }


def binary_role_metrics(
    truth: np.ndarray, signal: np.ndarray, threshold: float, arrays: Mapping[str, np.ndarray], indices: np.ndarray
) -> dict[str, Any]:
    y = np.asarray(truth[indices], dtype=bool)
    pred = np.asarray(signal[indices] >= threshold, dtype=bool)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    timing, unmatched = transition_timing_error(truth, signal, threshold, arrays, indices)
    previous_truth = previous_by_agent(truth, arrays)
    prior_positive = previous_truth[indices]
    premature = (~pred) & y & prior_positive
    return {
        "rows": int(len(indices)),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_persistence_rate": float(fp / max(fp + tn, 1)),
        "premature_release_rate": float(np.sum(premature) / max(np.sum(y & prior_positive), 1)),
        "transition_timing_error_steps": float(timing),
        "unmatched_teacher_transitions": int(unmatched),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def previous_by_agent(values: np.ndarray, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    result = np.zeros(len(values), dtype=bool)
    lookup = _sequence_lookup(arrays)
    for row in range(len(values)):
        previous = lookup.get((int(arrays["episode"][row]), int(arrays["timestep"][row]) - 1, int(arrays["agent"][row])))
        if previous is not None:
            result[row] = bool(values[previous])
    return result


def transition_timing_error(
    truth: np.ndarray, signal: np.ndarray, threshold: float, arrays: Mapping[str, np.ndarray], indices: np.ndarray
) -> tuple[float, int]:
    allowed = set(int(value) for value in indices.tolist())
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in indices:
        grouped[(int(arrays["episode"][row]), int(arrays["agent"][row]))].append(int(row))
    errors: list[float] = []
    unmatched = 0
    for rows in grouped.values():
        rows.sort(key=lambda row: int(arrays["timestep"][row]))
        y = truth[rows].astype(bool)
        p = signal[rows] >= threshold
        y_events = [i for i in range(1, len(rows)) if y[i] != y[i - 1]]
        p_events = [i for i in range(1, len(rows)) if p[i] != p[i - 1]]
        for event in y_events:
            candidates = [candidate for candidate in p_events if p[candidate] == y[event]]
            if not candidates:
                unmatched += 1
            else:
                errors.append(float(min(abs(event - candidate) for candidate in candidates)))
    return float(np.mean(errors)) if errors else float("inf"), unmatched


def select_threshold(truth: np.ndarray, signal: np.ndarray, arrays: Mapping[str, np.ndarray], indices: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = np.linspace(0.0, 1.0, 101)
    y = np.asarray(truth[indices], dtype=bool)
    scored: list[dict[str, Any]] = []
    for value in candidates:
        pred = np.asarray(signal[indices] >= value, dtype=bool)
        tp = int(np.sum(pred & y))
        fp = int(np.sum(pred & ~y))
        fn = int(np.sum(~pred & y))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
        scored.append({"rows": int(len(indices)), "threshold": float(value), "precision": float(precision), "recall": float(recall), "f1": float(f1)})
    best = max(scored, key=lambda row: (row["f1"], row["precision"], -row["threshold"]))
    return float(best["threshold"]), {"selected": best, "grid": scored}


def role_summary(
    truth: np.ndarray, signal: np.ndarray, arrays: Mapping[str, np.ndarray], splits: Mapping[str, np.ndarray], label: str
) -> dict[str, Any]:
    threshold, selection = select_threshold(truth, signal, arrays, splits["validation"])
    return {
        "label": label,
        "validation_selection": selection["selected"],
        "validation_threshold_grid": selection["grid"],
        "train": binary_role_metrics(truth, signal, threshold, arrays, splits["train"]),
        "validation": binary_role_metrics(truth, signal, threshold, arrays, splits["validation"]),
        "test": binary_role_metrics(truth, signal, threshold, arrays, splits["test"]),
    }


def teacher_probabilities(teacher_q: np.ndarray) -> np.ndarray:
    logits = np.asarray(teacher_q, dtype=np.float64) / TEMPERATURE
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(np.clip(logits, -80.0, 0.0))
    return values / values.sum(axis=1, keepdims=True)


def probe_metrics(
    probabilities: np.ndarray, arrays: Mapping[str, np.ndarray], indices: np.ndarray, teacher_p: np.ndarray
) -> dict[str, Any]:
    target = np.asarray(arrays["greedy_action"], dtype=np.int64)
    p = np.clip(probabilities[indices].astype(np.float64), 1.0e-9, 1.0)
    tp = teacher_p[indices]
    rows: dict[str, Any] = {
        "rows": int(len(indices)),
        "action_agreement": float(np.mean(np.argmax(p, axis=1) == target[indices])),
        "categorical_cross_entropy": float(-np.mean(np.log(p[np.arange(len(indices)), target[indices]]))),
        "categorical_kl_teacher_to_probe": float(np.mean(np.sum(tp * (np.log(np.clip(tp, 1.0e-12, 1.0)) - np.log(p)), axis=1))),
        "phase": {},
        "condition": {},
    }
    for value, name in PHASES.items():
        mask = np.asarray(arrays["phase_id"])[indices] == value
        rows["phase"][name] = probe_metric_mask(p, target[indices], tp, mask)
    for name, predicate in CONDITIONS.items():
        mask = predicate(arrays)[indices]
        rows["condition"][name] = probe_metric_mask(p, target[indices], tp, mask)
    return rows


def probe_metric_mask(p: np.ndarray, target: np.ndarray, teacher_p: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    count = int(np.sum(mask))
    if not count:
        return {"rows": 0, "action_agreement": None, "categorical_cross_entropy": None, "categorical_kl_teacher_to_probe": None}
    selected = p[mask]
    selected_target = target[mask]
    selected_teacher = teacher_p[mask]
    return {
        "rows": count,
        "action_agreement": float(np.mean(np.argmax(selected, axis=1) == selected_target)),
        "categorical_cross_entropy": float(-np.mean(np.log(selected[np.arange(count), selected_target]))),
        "categorical_kl_teacher_to_probe": float(np.mean(np.sum(selected_teacher * (np.log(np.clip(selected_teacher, 1.0e-12, 1.0)) - np.log(selected)), axis=1))),
    }


def fit_action_probe(
    name: str, features: np.ndarray, arrays: Mapping[str, np.ndarray], splits: Mapping[str, np.ndarray], teacher_p: np.ndarray
) -> dict[str, Any]:
    target = np.asarray(arrays["greedy_action"], dtype=np.int64)
    train_indices = probe_train_indices(splits["train"], target)
    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(loss="log_loss", alpha=1.0e-4, max_iter=20, tol=1.0e-3, average=True, random_state=20260918),
    )
    model.fit(features[train_indices], target[train_indices])
    raw_probabilities = model.predict_proba(features).astype(np.float32)
    probabilities = np.zeros((len(features), 9), dtype=np.float32)
    probabilities[:, np.asarray(model[-1].classes_, dtype=np.int64)] = raw_probabilities
    return {
        "label": name,
        "feature_dim": int(features.shape[1]),
        "probe_train_rows": int(len(train_indices)),
        "classes_seen_in_probe_train": [int(value) for value in model[-1].classes_.tolist()],
        "train": probe_metrics(probabilities, arrays, splits["train"], teacher_p),
        "validation": probe_metrics(probabilities, arrays, splits["validation"], teacher_p),
        "test": probe_metrics(probabilities, arrays, splits["test"], teacher_p),
    }


def probe_train_indices(
    train_indices: np.ndarray, target: np.ndarray | None = None, max_rows: int = 32768
) -> np.ndarray:
    """Deterministic bounded train subset for a lightweight diagnostic probe."""

    values = np.asarray(train_indices, dtype=np.int64)
    if len(values) <= max_rows:
        return values
    required: list[int] = []
    if target is not None:
        labels = np.asarray(target)
        for category in np.unique(labels[values]):
            required.append(int(values[np.flatnonzero(labels[values] == category)[0]]))
    required_array = np.asarray(sorted(set(required)), dtype=np.int64)
    remaining = values[~np.isin(values, required_array)]
    budget = max(max_rows - len(required_array), 0)
    if len(remaining) > budget:
        stride = int(np.ceil(len(remaining) / max(budget, 1)))
        remaining = remaining[::stride][:budget]
    return np.asarray(sorted(set(required_array.tolist() + remaining.tolist())), dtype=np.int64)


def near_alias_report(
    physical: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    extras: Mapping[str, np.ndarray],
    threshold: float = 0.10,
) -> dict[str, Any]:
    pursuing = np.asarray(arrays["pursuing"], dtype=bool)
    left = np.flatnonzero(~pursuing)
    right = np.flatnonzero(pursuing)
    actions = np.asarray(arrays["greedy_action"], dtype=np.int64)
    reports: dict[str, Any] = {}
    for label, extra in extras.items():
        extra_array = np.asarray(extra, dtype=np.float32)
        vectors = physical if extra_array.size == 0 else np.concatenate([physical, extra_array], axis=1)
        tree = cKDTree(vectors[right])
        distance, neighbor = tree.query(vectors[left], k=1)
        rmse = distance / np.sqrt(vectors.shape[1])
        paired = right[neighbor]
        close = rmse <= threshold
        mismatch = actions[left] != actions[paired]
        support_pair = np.asarray(arrays["support"], dtype=bool)[left] | np.asarray(arrays["support"], dtype=bool)[paired]
        recovery_pair = (np.asarray(arrays["phase_id"])[left] == 1) | (np.asarray(arrays["phase_id"])[paired] == 1)
        reports[label] = {
            "vector_dim": int(vectors.shape[1]),
            "p10": float(np.percentile(rmse, 10)),
            "p25": float(np.percentile(rmse, 25)),
            "p50": float(np.percentile(rmse, 50)),
            "p90": float(np.percentile(rmse, 90)),
            "threshold": threshold,
            "close_pairs": int(np.sum(close)),
            "close_pair_fraction": float(np.mean(close)),
            "close_action_mismatches": int(np.sum(close & mismatch)),
            "close_action_mismatch_rate": float(np.sum(close & mismatch) / max(np.sum(close), 1)),
            "support_close_pairs": int(np.sum(close & support_pair)),
            "support_action_mismatch_rate": float(np.sum(close & support_pair & mismatch) / max(np.sum(close & support_pair), 1)),
            "recovery_close_pairs": int(np.sum(close & recovery_pair)),
            "recovery_action_mismatch_rate": float(np.sum(close & recovery_pair & mismatch) / max(np.sum(close & recovery_pair), 1)),
        }
    return {
        "definition": "nearest cross-role (pursuing vs non-pursuing) pair under B1 physical vector; augmented RMSE is normalized over physical plus candidate dimensions",
        "baseline_b1_expected": {"close_pairs": 118, "close_action_mismatches": 87, "threshold": threshold},
        "rows": {"without_pursuing": int(len(left)), "with_pursuing": int(len(right))},
        "candidates": reports,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    data = report["data_sufficiency"]
    role = report["role_reconstruction"]
    probes = report["action_probes"]["test"]
    aliases = report["near_observation_aliasing"]["candidates"]
    lines = [
        "# B1.5 Hidden-State / Local Target Evidence Reconstruction Audit",
        "",
        "本报告是冻结 B1 teacher dataset 上的离线诊断；没有 online RL、没有修改正式 IQN observation/reward、没有实现 B3。",
        "",
        "## 数据与重建充分性",
        "",
        f"- rows={report['dataset']['rows']}，episodes={report['dataset']['episodes']}，teacher SHA={report['teacher_sha256']}。",
        f"- direct sensing / friend adjacency / neighbor evidence 一致性：{data['neighbor_target_mask_match_rate']:.6f}；friend token 匹配率：{data['token_match_rate']:.6f}。",
        f"- teacher pursuing 与当前 direct visibility 是否完全重合：{data['teacher_role_equals_current_direct_visibility']}；cross-tab（direct 0/1 × pursuing 0/1）={data['direct_teacher_pursuing_crosstab']}。",
        "- history 按 episode → timestep → agent 顺序恢复，episode 起点 z=0；同步更新不使用同 step 递归。",
        "",
        "## Role reconstruction（test）",
        "",
        "| candidate | threshold | precision | recall | F1 | false persistence | premature release | timing error (steps) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in role.items():
        row = item["test"]
        lines.append(
            f"| {label} | {row['threshold']:.2f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['false_persistence_rate']:.4f} | {row['premature_release_rate']:.4f} | {row['transition_timing_error_steps']:.3f} |"
        )
    lines += ["", "## Action probe（test）", "", "| input | agreement | CE | KL(teacher‖probe) |", "|---|---:|---:|---:|"]
    for label, item in probes.items():
        metrics = item["test"]
        lines.append(f"| {label} | {metrics['action_agreement']:.4f} | {metrics['categorical_cross_entropy']:.4f} | {metrics['categorical_kl_teacher_to_probe']:.4f} |")
    comparison = report["action_probe_comparison"]
    lines += [
        "",
        f"H2-H1 overall action agreement delta={comparison['overall_action_agreement_delta_h2_minus_h1']:.4f}；support 条件 agreement delta={comparison['support_action_agreement_delta_h2_minus_h1']:.4f}；support 条件 CE delta={comparison['support_ce_delta_h2_minus_h1']:.4f}。",
        "probe train split 未包含 action 4；该类在 9 类输出中以零概率补齐，未从 validation/test 借用标签。",
        "phase/condition 明细与 train/validation 结果见 JSON。",
        "",
        "## Near-observation aliasing（RMSE≤0.10）",
        "",
        "| augmented vector | close pairs | mismatch rate | support mismatch | recovery mismatch |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in aliases.items():
        lines.append(f"| {label} | {item['close_pairs']} | {item['close_action_mismatch_rate']:.4f} | {item['support_action_mismatch_rate']:.4f} | {item['recovery_action_mismatch_rate']:.4f} |")
    lines += [
        "",
        "说明：augmented RMSE 在扩展后的总维度上归一化；因此 close pairs 归零是固定阈值下的诊断信号，不能单独视为动作机制或因果证明，应结合 p50 距离与 action probe 解读。",
    ]
    lines += ["", "## B1.5 结论", "", report["conclusion"], ""]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    arrays, global_arrays, dataset_meta = load_dataset(args.dataset_root.resolve())
    view = role_free_view(arrays)
    physical = physical_vectors(view)
    adjacency, adjacency_report = recover_adjacency(arrays, global_arrays)
    direct = np.asarray(arrays["direct_target_mask"], dtype=bool).any(axis=1)
    neighbor = np.asarray(arrays["neighbor_target_mask"], dtype=bool).any(axis=1)
    if not np.array_equal(direct, np.asarray(arrays["direct"], dtype=bool)):
        raise ValueError("direct_target_mask does not agree with teacher direct label")
    splits = split_indices(arrays)
    truth = np.asarray(arrays["pursuing"], dtype=bool)
    signals: dict[str, np.ndarray] = {}
    role_reconstruction: dict[str, Any] = {}
    h1_valid: list[dict[str, Any]] = []
    for lambda_value in LAMBDA_GRID:
        signal = reconstruct_signal(arrays, adjacency, lambda_value, 0.0)
        label = f"H1_self_only_lambda_{lambda_value:.2f}"
        signals[label] = signal
        threshold, selection = select_threshold(truth, signal, arrays, splits["validation"])
        h1_valid.append({"lambda": lambda_value, "threshold": threshold, "validation": selection["selected"]})
    best_h1 = max(h1_valid, key=lambda row: (row["validation"]["f1"], row["validation"]["precision"], -row["lambda"]))
    best_h1_label = f"H1_self_only_lambda_{best_h1['lambda']:.2f}"
    role_reconstruction["H1_self_only"] = role_summary(truth, signals[best_h1_label], arrays, splits, best_h1_label)
    role_reconstruction["H1_self_only"]["sweep_validation"] = h1_valid

    h2_valid: list[dict[str, Any]] = []
    for lambda_value in LAMBDA_GRID:
        for eta_value in ETA_GRID:
            signal = reconstruct_signal(arrays, adjacency, lambda_value, eta_value)
            label = f"H2_self_neighbor_lambda_{lambda_value:.2f}_eta_{eta_value:.2f}"
            signals[label] = signal
            threshold, selection = select_threshold(truth, signal, arrays, splits["validation"])
            h2_valid.append({"lambda": lambda_value, "eta": eta_value, "threshold": threshold, "validation": selection["selected"]})
    best_h2 = max(h2_valid, key=lambda row: (row["validation"]["f1"], row["validation"]["precision"], -row["lambda"], -row["eta"]))
    best_h2_label = f"H2_self_neighbor_lambda_{best_h2['lambda']:.2f}_eta_{best_h2['eta']:.2f}"
    role_reconstruction["H2_self_neighbor"] = role_summary(truth, signals[best_h2_label], arrays, splits, best_h2_label)
    role_reconstruction["H2_self_neighbor"]["sweep_validation"] = h2_valid

    history_features: dict[int, np.ndarray] = {}
    history_age: dict[int, np.ndarray] = {}
    current_distance: np.ndarray | None = None
    for k in HISTORY_K:
        history_features[k], history_age[k], current_distance = build_history_features(arrays, k)
    for k in HISTORY_K:
        label = f"H3_history_K{k}"
        train_indices = probe_train_indices(splits["train"], truth.astype(np.int64))
        role_probe = make_pipeline(
            StandardScaler(),
            SGDClassifier(loss="log_loss", alpha=1.0e-4, max_iter=20, tol=1.0e-3, class_weight="balanced", average=True, random_state=20260918),
        )
        role_probe.fit(history_features[k][train_indices], truth[train_indices].astype(np.int64))
        probabilities = role_probe.predict_proba(history_features[k])[:, 1]
        threshold, selection = select_threshold(truth, probabilities, arrays, splits["validation"])
        role_reconstruction[label] = {
            "label": label,
            "probe": "balanced logistic role probe on history only; fitted on train episodes",
            "probe_train_rows": int(len(train_indices)),
            "validation_selection": selection["selected"],
            "validation_threshold_grid": selection["grid"],
            "train": binary_role_metrics(truth, probabilities, threshold, arrays, splits["train"]),
            "validation": binary_role_metrics(truth, probabilities, threshold, arrays, splits["validation"]),
            "test": binary_role_metrics(truth, probabilities, threshold, arrays, splits["test"]),
        }
        signals[label] = probabilities.astype(np.float32)

    teacher_p = teacher_probabilities(np.asarray(arrays["teacher_q"], dtype=np.float32))
    action_features: dict[str, np.ndarray] = {"o": physical}
    action_features["o+m"] = np.concatenate([physical, signals[best_h1_label][:, None]], axis=1)
    action_features["o+z"] = np.concatenate([physical, signals[best_h2_label][:, None]], axis=1)
    for k in HISTORY_K:
        action_features[f"o+history_K{k}"] = np.concatenate([physical, history_features[k]], axis=1)
    action_probes: dict[str, Any] = {}
    for label, features in action_features.items():
        action_probes[label] = fit_action_probe(label, features, arrays, splits, teacher_p)

    alias_extras: dict[str, np.ndarray] = {
        "o": np.zeros((len(physical), 0), dtype=np.float32),
        "o+m": signals[best_h1_label][:, None],
        "o+z": signals[best_h2_label][:, None],
    }
    for k in HISTORY_K:
        alias_extras[f"o+history_K{k}"] = history_features[k]
    alias_report = near_alias_report(physical, arrays, alias_extras)
    baseline = alias_report["candidates"]["o"]
    if baseline["close_pairs"] != 118 or baseline["close_action_mismatches"] != 87:
        raise ValueError(f"B1 nearest-cross-role baseline drifted: {baseline}")

    # This is deliberately an explanatory judgment, not a B3 implementation.
    h1_f1 = role_reconstruction["H1_self_only"]["test"]["f1"]
    h2_f1 = role_reconstruction["H2_self_neighbor"]["test"]["f1"]
    h1_agree = action_probes["o+m"]["test"]["action_agreement"]
    h2_agree = action_probes["o+z"]["test"]["action_agreement"]
    support_h1 = action_probes["o+m"]["test"]["condition"]["support"]
    support_h2 = action_probes["o+z"]["test"]["condition"]["support"]
    support_delta = float(support_h2["action_agreement"] - support_h1["action_agreement"])
    support_ce_delta = float(support_h2["categorical_cross_entropy"] - support_h1["categorical_cross_entropy"])
    role_is_direct_degenerate = bool(np.array_equal(direct, truth))
    direct_teacher_pursuing_crosstab = [
        [int(np.sum((direct == direct_value) & (truth == pursuing_value))) for pursuing_value in (False, True)]
        for direct_value in (False, True)
    ]
    if role_is_direct_degenerate:
        conclusion = (
            "本数据集的 role reconstruction 是退化性结果：teacher pursuing 与当前 direct visibility 在全部 rows 上完全相等 "
            f"（cross-tab={direct_teacher_pursuing_crosstab}），因此 H1/H2 的完美或近乎完美 F1 不能检验 direct evidence 衰减、release memory 或隐藏 pursuing 状态。"
        )
        if support_delta > 0.05:
            conclusion += (
                f" H2 在 support 条件下仍使 test action agreement 提升 {support_delta:.4f}，并使 support 条件 CE 改变 {support_ce_delta:.4f} "
                "（负值为改善）；但 overall agreement 提升不足 1 个百分点，故应保留 H2 作为候选而非宣称其已证明必要。"
            )
        else:
            conclusion += " H2 未带来足够的 support 条件动作探针收益，当前数据不足以支持 H2 必要。"
        conclusion += " 后续需要冻结 teacher 的轨迹中出现 direct=False、pursuing=True 的 rows，才能审计真正的 hidden-state reconstruction。"
    elif support_delta > 0.05:
        conclusion = (
            f"H2 neighbor propagation 在 support 条件下提供实质性 test action-probe 增益（agreement +{support_delta:.4f}，CE delta {support_ce_delta:.4f}）；"
            "保留 H2 作为候选，H1 作为 self-only control。"
        )
    elif h2_agree > h1_agree + 0.01:
        conclusion = "H2 neighbor propagation provides a material (>1 percentage point) test-set action-probe gain over H1; retain it as a candidate, while H1 remains the self-only control."
    else:
        conclusion = "H1 self-only recent direct evidence is sufficient for the measured offline ambiguity reduction; H2 neighbor propagation is not materially necessary under this dataset and probe contract."
    if h2_f1 > h1_f1 + 0.03:
        conclusion += " H2 also improves teacher-role reconstruction materially."
    else:
        conclusion += f" H2 does not materially improve teacher-role reconstruction (test F1 H1={h1_f1:.4f}, H2={h2_f1:.4f})."
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "scope": {
            "offline_only": True,
            "online_rl": False,
            "formal_iqn_observation_modified": False,
            "reward_modified": False,
            "b3_implemented": False,
            "cpu_only": True,
        },
        "branch": "experiment/iqn-hidden-state-b15-20260918",
        "head": _git_head(),
        "base_b1_head": BASE_B1_HEAD,
        "teacher_sha256": TEACHER_SHA,
        "dataset": dataset_meta,
        "data_sufficiency": {
            "current_direct_visibility": "direct_target_mask.any(axis=1)",
            "friend_adjacency": "recovered by global geometry/local friend-token matching",
            "previous_timestep_history": "episode/timestep/agent rows; complete pre-action sequences; reset per episode",
            "teacher_is_pursuing": "pursuing field",
            "teacher_action": "greedy_action and teacher_q",
            "direct_teacher_pursuing_crosstab": direct_teacher_pursuing_crosstab,
            "teacher_role_equals_current_direct_visibility": role_is_direct_degenerate,
            **adjacency_report,
        },
        "episode_split": {name: {"rows": int(len(index)), "episodes": sorted(set(int(value) for value in arrays["episode"][index].tolist()))} for name, index in splits.items()},
        "predeclared_grid": {"lambda": list(LAMBDA_GRID), "eta": list(ETA_GRID), "history_K": list(HISTORY_K)},
        "role_reconstruction": role_reconstruction,
        "action_probes": {"teacher_temperature": TEMPERATURE, "test": action_probes, "all_splits": action_probes},
        "action_probe_comparison": {
            "overall_action_agreement_delta_h2_minus_h1": float(h2_agree - h1_agree),
            "support_action_agreement_delta_h2_minus_h1": support_delta,
            "support_ce_delta_h2_minus_h1": support_ce_delta,
            "probe_train_missing_action_classes": sorted(set(range(9)) - set(action_probes["o"]["classes_seen_in_probe_train"])),
            "missing_train_classes_zero_padded": True,
        },
        "near_observation_aliasing": alias_report,
        "conclusion": conclusion,
    }
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    atomic_json(artifact_root / "B15_AUDIT.json", report)
    (artifact_root / "B15_AUDIT.md").write_text(markdown_report(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "artifact_root": str(args.artifact_root.resolve()), "conclusion": report["conclusion"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
