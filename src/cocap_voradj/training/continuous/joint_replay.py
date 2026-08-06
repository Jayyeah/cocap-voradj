"""Joint replay contract and deterministic 70/20/10 sampling for continuous MARL."""
from __future__ import annotations

import pickle
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch


_FORBIDDEN_METADATA_KEYS = {
    "v_raw",
    "v_candidate",
    "v_world",
    "raw_action",
    "candidate_action",
    "global_phase_label",
    "oracle_event_label",
}
_ALLOWED_METADATA_KEYS = {
    "regime",
    "event_ids",
    "phase",
    "scene",
    "task_label",
    "coverage_only",
    "active_target",
}


def _array(value: Any, dtype: np.dtype, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} shape mismatch: expected {shape}, got {result.shape}")
    if not np.all(np.isfinite(result.astype(float, copy=False))):
        raise ValueError(f"{name} contains NaN or Inf")
    return result.copy()


def _copy_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _copy_tree(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _metadata(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    raw = dict(value or {})
    forbidden = sorted(_FORBIDDEN_METADATA_KEYS.intersection(raw))
    if forbidden:
        raise ValueError(f"joint replay metadata cannot contain action candidates: {forbidden}")
    unknown = sorted(set(raw).difference(_ALLOWED_METADATA_KEYS))
    if unknown:
        raise ValueError(f"joint replay metadata contains non-contract keys: {unknown}")
    result = {key: _copy_tree(raw[key]) for key in raw}
    event_ids = result.get("event_ids", [])
    if isinstance(event_ids, str):
        event_ids = [event_ids]
    if not isinstance(event_ids, (list, tuple)):
        raise ValueError("metadata.event_ids must be a list of IDs")
    result["event_ids"] = [str(item) for item in event_ids]
    result["regime"] = str(result.get("regime", "uniform"))
    result["coverage_only"] = bool(result.get("coverage_only", result["regime"] == "coverage_only"))
    result["active_target"] = bool(result.get("active_target", result["regime"] == "active_target"))
    return result


@dataclass(frozen=True)
class JointTransition:
    """One atomic multi-agent transition; actions are body-frame accelerations."""

    local_obs: Dict[str, np.ndarray]
    next_local_obs: Dict[str, np.ndarray]
    global_state: Dict[str, np.ndarray] | np.ndarray
    next_global_state: Dict[str, np.ndarray] | np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    active_mask: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    metadata: Dict[str, Any]
    transition_id: int = -1


class JointReplayBuffer:
    """Persistent-capable ring of joint transitions with atomic event indices."""

    schema_version = 3

    def __init__(self, capacity: int, max_agents: int, seed: int = 0):
        if int(capacity) <= 0 or int(max_agents) <= 0:
            raise ValueError("capacity and max_agents must be positive")
        self.capacity = int(capacity)
        self.max_agents = int(max_agents)
        self._records: "OrderedDict[int, JointTransition]" = OrderedDict()
        self._order: deque[int] = deque()
        self._next_id = 0
        self._regime_index: MutableMapping[str, set[int]] = defaultdict(set)
        self._event_index: MutableMapping[str, set[int]] = defaultdict(set)
        self.rng = np.random.default_rng(int(seed))
        self.last_sample_stats: Dict[str, Any] = {}
        self.runtime_state: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self._records)

    @property
    def transition_ids(self) -> tuple[int, ...]:
        return tuple(self._order)

    def _validate_obs_tree(self, value: Mapping[str, Any], name: str) -> Dict[str, np.ndarray]:
        if not value:
            raise ValueError(f"{name} cannot be empty")
        result: Dict[str, np.ndarray] = {}
        for key, array in value.items():
            result[str(key)] = np.asarray(array).copy()
            if not np.all(np.isfinite(result[str(key)].astype(float, copy=False))):
                raise ValueError(f"{name}.{key} contains NaN or Inf")
        return result

    def add(
        self,
        *,
        local_obs: Mapping[str, Any],
        next_local_obs: Mapping[str, Any],
        global_state: Mapping[str, Any] | Any,
        next_global_state: Mapping[str, Any] | Any,
        actions: Any,
        rewards: Any,
        active_mask: Any,
        terminated: Any,
        truncated: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> int:
        local = self._validate_obs_tree(local_obs, "local_obs")
        next_local = self._validate_obs_tree(next_local_obs, "next_local_obs")
        if set(local) != set(next_local):
            raise ValueError("local_obs and next_local_obs keys must match")
        global_value = _copy_tree(global_state)
        next_global_value = _copy_tree(next_global_state)
        if isinstance(global_value, Mapping) != isinstance(next_global_value, Mapping):
            raise ValueError("global_state and next_global_state container types must match")
        if isinstance(global_value, Mapping):
            global_value = self._validate_obs_tree(global_value, "global_state")
            next_global_value = self._validate_obs_tree(next_global_value, "next_global_state")
            if set(global_value) != set(next_global_value):
                raise ValueError("global_state keys must match next_global_state keys")
        else:
            global_value = np.asarray(global_value, dtype=np.float32).copy()
            next_global_value = np.asarray(next_global_value, dtype=np.float32).copy()
            if global_value.shape != next_global_value.shape:
                raise ValueError("global_state shape must match next_global_state shape")
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.shape != (self.max_agents, 2):
            raise ValueError(f"actions must have shape ({self.max_agents}, 2)")
        if not np.all(np.isfinite(action_array)):
            raise ValueError("actions contains NaN or Inf")
        transition = JointTransition(
            local_obs=local,
            next_local_obs=next_local,
            global_state=global_value,
            next_global_state=next_global_value,
            actions=action_array.copy(),
            rewards=_array(rewards, np.float32, (self.max_agents,), "rewards"),
            active_mask=_array(active_mask, np.bool_, (self.max_agents,), "active_mask").astype(bool),
            terminated=_array(terminated, np.bool_, (self.max_agents,), "terminated").astype(bool),
            truncated=_array(truncated, np.bool_, (self.max_agents,), "truncated").astype(bool),
            metadata=_metadata(metadata),
            transition_id=self._next_id,
        )
        transition_id = self._next_id
        self._next_id += 1
        if len(self._order) >= self.capacity:
            self._remove(self._order[0])
        self._records[transition_id] = transition
        self._order.append(transition_id)
        regime = transition.metadata["regime"]
        self._regime_index[regime].add(transition_id)
        for event_id in transition.metadata["event_ids"]:
            self._event_index[event_id].add(transition_id)
        return transition_id

    def _remove(self, transition_id: int) -> None:
        transition = self._records.pop(int(transition_id))
        self._order.remove(int(transition_id))
        self._regime_index[transition.metadata["regime"]].discard(int(transition_id))
        for event_id in transition.metadata["event_ids"]:
            self._event_index[event_id].discard(int(transition_id))

    def _pool(self, kind: str, key: Optional[str] = None) -> List[int]:
        if kind == "uniform":
            return list(self._order)
        if kind == "regime":
            return [idx for idx in self._order if self._records[idx].metadata["regime"] == str(key)]
        if kind == "event":
            return sorted(set().union(*(self._event_index[event] for event in self._event_index))) if key is None else sorted(self._event_index[str(key)])
        if kind == "coverage_only":
            return [idx for idx in self._order if self._records[idx].metadata["coverage_only"]]
        raise ValueError(f"unknown joint replay pool: {kind}")

    def _draw(self, pool: Sequence[int], count: int) -> List[int]:
        if count <= 0:
            return []
        if not pool:
            return []
        return [int(value) for value in self.rng.choice(np.asarray(pool, dtype=np.int64), size=int(count), replace=len(pool) < count)]

    def get(self, transition_id: int) -> JointTransition:
        return self._records[int(transition_id)]

    def sample_ids(self, batch_size: int, sampler: "JointReplaySampler" | None = None) -> List[int]:
        if len(self) == 0:
            raise ValueError("cannot sample an empty joint replay")
        sampler = sampler or JointReplaySampler()
        return sampler.sample_ids(self, int(batch_size))

    def batch_from_ids(self, ids: Sequence[int], device: str = "cpu") -> Dict[str, Any]:
        transitions = [self.get(idx) for idx in ids]
        if not transitions:
            raise ValueError("batch ids cannot be empty")
        def stack_tree(name: str) -> Dict[str, torch.Tensor]:
            keys = transitions[0].__dict__[name].keys()
            return {key: torch.as_tensor(np.stack([getattr(item, name)[key] for item in transitions]), dtype=torch.float32, device=device) for key in keys}
        def stack_global(name: str) -> Any:
            value = getattr(transitions[0], name)
            if isinstance(value, Mapping):
                return {key: torch.as_tensor(np.stack([getattr(item, name)[key] for item in transitions]), dtype=torch.float32, device=device) for key in value}
            return torch.as_tensor(np.stack([getattr(item, name) for item in transitions]), dtype=torch.float32, device=device)
        return {
            "local_obs": stack_tree("local_obs"),
            "next_local_obs": stack_tree("next_local_obs"),
            "global_state": stack_global("global_state"),
            "next_global_state": stack_global("next_global_state"),
            "actions": torch.as_tensor(np.stack([item.actions for item in transitions]), dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(np.stack([item.rewards for item in transitions]), dtype=torch.float32, device=device),
            "active_mask": torch.as_tensor(np.stack([item.active_mask for item in transitions]), dtype=torch.bool, device=device),
            "terminated": torch.as_tensor(np.stack([item.terminated for item in transitions]), dtype=torch.bool, device=device),
            "truncated": torch.as_tensor(np.stack([item.truncated for item in transitions]), dtype=torch.bool, device=device),
            "metadata": [dict(item.metadata) for item in transitions],
            "transition_ids": torch.as_tensor(list(ids), dtype=torch.long, device=device),
        }

    def sample(self, batch_size: int, device: str = "cpu", sampler: "JointReplaySampler" | None = None) -> Dict[str, Any]:
        ids = self.sample_ids(batch_size, sampler=sampler)
        batch = self.batch_from_ids(ids, device=device)
        batch["sampling_stats"] = dict(self.last_sample_stats)
        return batch

    def save(
        self,
        path: str | Path,
        manifest: Mapping[str, Any],
        runtime_state: Mapping[str, Any] | None = None,
    ) -> None:
        payload = {
            "schema_version": self.schema_version,
            "capacity": self.capacity,
            "max_agents": self.max_agents,
            "next_id": self._next_id,
            "records": list(self._records.items()),
            "rng_state": self.rng.bit_generator.state,
            "manifest": dict(manifest),
            "runtime_state": dict(runtime_state or {}),
        }
        with Path(path).open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path, expected_manifest: Mapping[str, Any]) -> "JointReplayBuffer":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("schema_version") != cls.schema_version:
            raise ValueError("joint replay schema version mismatch")
        if dict(payload.get("manifest", {})) != dict(expected_manifest):
            raise ValueError("joint replay manifest mismatch; refusing unsafe resume")
        replay = cls(payload["capacity"], payload["max_agents"])
        replay._next_id = int(payload["next_id"])
        for transition_id, transition in payload["records"]:
            replay._records[int(transition_id)] = transition
            replay._order.append(int(transition_id))
            replay._regime_index[transition.metadata["regime"]].add(int(transition_id))
            for event_id in transition.metadata["event_ids"]:
                replay._event_index[event_id].add(int(transition_id))
        replay.rng.bit_generator.state = payload["rng_state"]
        replay.runtime_state = dict(payload.get("runtime_state", {}) or {})
        return replay


class JointReplaySampler:
    """Sample 70% uniform, 20% active-target/coverage regime, 10% events."""

    def __init__(self, uniform_ratio: float = 0.70, regime_ratio: float = 0.20, event_ratio: float = 0.10, coverage_floor: float = 0.25):
        ratios = np.asarray([uniform_ratio, regime_ratio, event_ratio], dtype=float)
        if np.any(ratios < 0) or not np.isclose(float(ratios.sum()), 1.0):
            raise ValueError("uniform/regime/event ratios must be non-negative and sum to 1")
        self.uniform_ratio, self.regime_ratio, self.event_ratio = [float(item) for item in ratios]
        self.coverage_floor = float(coverage_floor)
        if not 0.0 <= self.coverage_floor <= 1.0:
            raise ValueError("coverage_floor must be in [0,1]")

    def sample_ids(self, replay: JointReplayBuffer, batch_size: int) -> List[int]:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        uniform_count = int(round(batch_size * self.uniform_ratio))
        regime_count = int(round(batch_size * self.regime_ratio))
        event_count = batch_size - uniform_count - regime_count
        uniform_ids = replay._draw(replay._pool("uniform"), uniform_count)
        regime_pool = replay._pool("regime", "active_target") + replay._pool("regime", "coverage_only")
        regime_ids = replay._draw(regime_pool, regime_count)
        event_pool = replay._pool("event")
        event_ids = replay._draw(event_pool, event_count)
        event_fallback_count = int(event_count > 0 and not event_pool)
        chosen = uniform_ids + regime_ids + event_ids
        source_slots = (
            ["uniform"] * len(uniform_ids)
            + ["regime"] * len(regime_ids)
            + ["event"] * len(event_ids)
        )
        coverage_target = int(np.ceil(batch_size * self.coverage_floor))
        coverage_pool = replay._pool("coverage_only")
        coverage_count = sum(replay.get(idx).metadata["coverage_only"] for idx in chosen)
        if coverage_pool and coverage_count < coverage_target:
            replacement_positions = [pos for pos, idx in enumerate(chosen) if not replay.get(idx).metadata["coverage_only"]]
            needed = min(coverage_target - int(coverage_count), len(replacement_positions))
            replacements = replay._draw(coverage_pool, needed)
            for pos, replacement in zip(replacement_positions[:needed], replacements):
                chosen[pos] = replacement
        while len(chosen) < batch_size:
            chosen.extend(replay._draw(replay._pool("uniform"), 1))
            source_slots.append("fallback_uniform")
        chosen = chosen[:batch_size]
        source_slots = source_slots[:batch_size]
        labels = [replay.get(idx).metadata["regime"] for idx in chosen]
        events = [bool(replay.get(idx).metadata["event_ids"]) for idx in chosen]
        replay.last_sample_stats = {
            "requested": batch_size,
            "actual": len(chosen),
            "uniform_ratio_target": self.uniform_ratio,
            "regime_ratio_target": self.regime_ratio,
            "event_ratio_target": self.event_ratio,
            "coverage_floor": self.coverage_floor,
            "coverage_only_count": int(sum(replay.get(idx).metadata["coverage_only"] for idx in chosen)),
            "regime_counts": {label: int(labels.count(label)) for label in sorted(set(labels))},
            "event_count": int(sum(events)),
            "event_pool_available": bool(event_pool),
            "event_fallback_count": event_fallback_count,
            "source_slot_counts": {
                source: int(source_slots.count(source)) for source in sorted(set(source_slots))
            },
        }
        return chosen
