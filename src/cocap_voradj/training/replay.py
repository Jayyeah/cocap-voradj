
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


def _count_values(values: Sequence[str]) -> Dict[str, int]:
    return {value: values.count(value) for value in sorted(set(values))}


def _reward_stats(rewards: Sequence[float], metadata: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    sources = {"reward_total": [float(value) for value in rewards]}
    for key in (
        "reward_coverage",
        "reward_capture",
        "reward_safety",
        "reward_terminal",
        "reward_speed_repeat",
        "reward_early_speed",
        "reward_cell_center_speed",
        "reward_deceleration",
        "reward_settled_terminal",
        "reward_motion_penalty",
        "reward_ce_center",
        "reward_ce_control",
        "reward_ce_pbrs",
        "reward_ce_terminal_correction",
        "reward_support_blend_capture",
        "reward_support_blend_coverage",
    ):
        sources[key] = [float(item.get(key, 0.0)) for item in metadata]
    for key, values in sources.items():
        array = np.asarray(values, dtype=float)
        stats[key] = {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }
    return stats


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=int(capacity))

    def add(self, obs: Dict, action: int, reward: float, next_obs: Dict, done: bool, metadata: Optional[Dict] = None) -> None:
        self.data.append((obs, int(action), float(reward), next_obs, bool(done), dict(metadata or {})))

    def __len__(self) -> int:
        return len(self.data)

    def _batch_from_indices(self, idx: Sequence[int], device: str) -> Dict:
        batch = [self.data[int(i)] for i in idx]
        metadata = [b[5] if len(b) > 5 else {} for b in batch]
        task_labels = [str(m.get("task_label", "unknown")) for m in metadata]
        phases = [str(m.get("phase", "unknown")) for m in metadata]
        scenes = [str(m.get("scene", "unknown")) for m in metadata]
        reset_sources = [str(m.get("recovery_reset_source", "unknown")) for m in metadata]
        buffer_classes = [str(m.get("buffer_class", "unknown")) for m in metadata]
        rewards = [float(b[2]) for b in batch]
        return {
            "obs": stack_obs([b[0] for b in batch], device),
            "actions": torch.as_tensor([b[1] for b in batch], dtype=torch.long, device=device),
            "rewards": torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=device),
            "next_obs": stack_obs([b[3] for b in batch], device),
            "dones": torch.as_tensor([b[4] for b in batch], dtype=torch.float32, device=device),
            "metadata": metadata,
            "task_labels": task_labels,
            "phases": phases,
            "batch_task_counts": _count_values(task_labels),
            "batch_phase_counts": _count_values(phases),
            "batch_scene_counts": _count_values(scenes),
            "batch_reset_source_counts": _count_values(reset_sources),
            "batch_buffer_counts": _count_values(buffer_classes),
            "batch_role_switch_count": int(sum(bool(m.get("task_switched", False)) for m in metadata)),
            "batch_reward_stats": _reward_stats(rewards, metadata),
        }

    def sample(self, batch_size: int, device: str) -> Dict:
        idx = np.random.randint(0, len(self.data), size=int(batch_size))
        return self._batch_from_indices(idx, device)

    def sample_task_balanced(self, batch_size: int, device: str, task_labels: Sequence[str] = ("coverage", "capture"), phase_balance_coverage: bool = True) -> Dict:
        batch_size = int(batch_size)
        by_task = {label: [] for label in task_labels}
        fallback = []
        for i, item in enumerate(self.data):
            meta = item[5] if len(item) > 5 else {}
            label = str(meta.get("task_label", "unknown"))
            if label in by_task:
                by_task[label].append(i)
            fallback.append(i)
        chosen: List[int] = []
        target_each = max(batch_size // max(len(task_labels), 1), 1)
        for label in task_labels:
            pool = by_task.get(label, [])
            if not pool:
                continue
            if label == "coverage" and phase_balance_coverage:
                pre = [i for i in pool if str((self.data[i][5] if len(self.data[i]) > 5 else {}).get("phase", "")) != "post_capture"]
                post = [i for i in pool if str((self.data[i][5] if len(self.data[i]) > 5 else {}).get("phase", "")) == "post_capture"]
                half = target_each // 2
                if pre and post and half > 0:
                    chosen.extend(np.random.choice(pre, size=half, replace=len(pre) < half).tolist())
                    rest = target_each - half
                    chosen.extend(np.random.choice(post, size=rest, replace=len(post) < rest).tolist())
                    continue
            chosen.extend(np.random.choice(pool, size=target_each, replace=len(pool) < target_each).tolist())
        while len(chosen) < batch_size:
            chosen.append(int(np.random.choice(fallback)))
        if len(chosen) > batch_size:
            chosen = chosen[:batch_size]
        return self._batch_from_indices(chosen, device)


def concat_replay_batches(batches: Sequence[Dict], scene_labels: Optional[Sequence[str]] = None) -> Dict:
    if not batches:
        raise ValueError("concat_replay_batches requires at least one batch")
    obs_keys = batches[0]["obs"].keys()
    out = {
        "obs": {key: torch.cat([batch["obs"][key] for batch in batches], dim=0) for key in obs_keys},
        "actions": torch.cat([batch["actions"] for batch in batches], dim=0),
        "rewards": torch.cat([batch["rewards"] for batch in batches], dim=0),
        "next_obs": {key: torch.cat([batch["next_obs"][key] for batch in batches], dim=0) for key in obs_keys},
        "dones": torch.cat([batch["dones"] for batch in batches], dim=0),
    }
    metadata: List[Dict] = []
    task_labels: List[str] = []
    phases: List[str] = []
    scene_counts: Dict[str, int] = {}
    for batch_idx, batch in enumerate(batches):
        batch_meta = [dict(item) for item in batch.get("metadata", [])]
        label = scene_labels[batch_idx] if scene_labels is not None and batch_idx < len(scene_labels) else None
        if label is not None:
            scene_counts[str(label)] = scene_counts.get(str(label), 0) + int(batch["actions"].shape[0])
            for item in batch_meta:
                item.setdefault("scene", str(label))
        metadata.extend(batch_meta)
        task_labels.extend([str(x) for x in batch.get("task_labels", [])])
        phases.extend([str(x) for x in batch.get("phases", [])])
    if not task_labels:
        task_labels = [str(m.get("task_label", "unknown")) for m in metadata]
    if not phases:
        phases = [str(m.get("phase", "unknown")) for m in metadata]
    out["metadata"] = metadata
    out["task_labels"] = task_labels
    out["phases"] = phases
    out["batch_task_counts"] = _count_values(task_labels)
    out["batch_phase_counts"] = _count_values(phases)
    if scene_counts:
        out["batch_scene_counts"] = scene_counts
    else:
        scenes = [str(m.get("scene", "unknown")) for m in metadata]
        out["batch_scene_counts"] = _count_values(scenes)
    buffer_classes = [str(m.get("buffer_class", "unknown")) for m in metadata]
    reset_sources = [str(m.get("recovery_reset_source", "unknown")) for m in metadata]
    out["batch_buffer_counts"] = _count_values(buffer_classes)
    out["batch_reset_source_counts"] = _count_values(reset_sources)
    out["batch_role_switch_count"] = int(sum(bool(m.get("task_switched", False)) for m in metadata))
    out["batch_reward_stats"] = _reward_stats(out["rewards"].detach().cpu().tolist(), metadata)
    return out


def stack_obs(items: List[Dict], device: str) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.stack([x[k] for x in items]), dtype=(torch.long if k == "types" else torch.bool if k == "masks" else torch.float32), device=device) for k in items[0]}
