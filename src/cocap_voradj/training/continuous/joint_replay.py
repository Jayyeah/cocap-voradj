"""Joint replay contract for focal-item and standard all-agent CTDE MASAC.

Storage unit: a complete joint transition ``(S,O,A,R,S',O',M,meta)``.
Optimizer unit is selected explicitly by the formal config: either a focal
``(joint_transition_id, focal_agent_id, bucket_id)`` item or a complete joint
transition whose active agents all enter the losses.
The legacy 70/20/10 ``JointReplaySampler`` is retained only for old smoke
tests. Formal focal training uses ``FocalReplaySampler``; the all-agent
ablation uses ``UniformJointReplaySampler``.
"""
from __future__ import annotations

import copy
import pickle
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

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
    "origin",
    "task_label",
    "coverage_only",
    "active_target",
    "recovery_reset_source",
    "support_candidate",
}

ROLE_INACTIVE = 0
ROLE_PURSUING = 1
ROLE_SUPPORT = 2
ROLE_COVERAGE = 3

FOCAL_BUCKETS = (
    "pre_capture_pursuing",
    "pre_capture_support",
    "pre_capture_coverage",
    "post_capture_coverage",
    "pure_recovery_coverage",
)

DEFAULT_FALLBACK_MATRIX: Dict[str, Tuple[str, ...]] = {
    "pre_capture_pursuing": ("pre_capture_coverage", "pre_capture_support", "post_capture_coverage", "pure_recovery_coverage"),
    "pre_capture_support": ("pre_capture_coverage", "pre_capture_pursuing", "post_capture_coverage", "pure_recovery_coverage"),
    "pre_capture_coverage": ("pre_capture_support", "pre_capture_pursuing", "post_capture_coverage", "pure_recovery_coverage"),
    "post_capture_coverage": ("pure_recovery_coverage", "pre_capture_coverage", "pre_capture_support", "pre_capture_pursuing"),
    "pure_recovery_coverage": ("post_capture_coverage", "pre_capture_coverage", "pre_capture_support", "pre_capture_pursuing"),
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
    result.setdefault("phase", "pre_capture")
    result.setdefault("scene", "unknown")
    result.setdefault("origin", "online")
    result.setdefault("task_label", "")
    return result


@dataclass(frozen=True)
class JointTransition:
    """One atomic multi-agent transition stored exactly once in the ring."""

    local_obs: Dict[str, np.ndarray]
    next_local_obs: Dict[str, np.ndarray]
    global_state: Dict[str, np.ndarray] | np.ndarray
    next_global_state: Dict[str, np.ndarray] | np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    active_mask: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    agent_role_id: np.ndarray
    phase: str
    scene: str
    origin: str
    metadata: Dict[str, Any]
    transition_id: int = -1
    slot_id: int = -1
    generation_id: int = 0


@dataclass(frozen=True)
class FocalItem:
    slot_id: int
    generation_id: int
    agent_id: int
    bucket_id: str


class JointReplayBuffer:
    """Persistent-capable ring of joint transitions with role reference indices."""

    schema_version = 4

    def __init__(self, capacity: int, max_agents: int, seed: int = 0):
        if int(capacity) <= 0 or int(max_agents) <= 0:
            raise ValueError("capacity and max_agents must be positive")
        self.capacity = int(capacity)
        self.max_agents = int(max_agents)
        self._records: "OrderedDict[int, JointTransition]" = OrderedDict()
        self._order: deque[int] = deque()
        self._next_id = 0
        self._next_slot = 0
        self._slot_record: Dict[int, int] = {}
        self._generation: Dict[int, int] = {}
        self._regime_index: MutableMapping[str, set[int]] = defaultdict(set)
        self._event_index: MutableMapping[str, set[int]] = defaultdict(set)
        # Derived role indexes are kept as dense pools plus O(1) key->position
        # maps. They are rebuilt from records on load and therefore do not
        # change the persistent replay schema.
        self._role_index: Dict[str, Dict[Tuple[int, int, int], int]] = {
            name: {} for name in FOCAL_BUCKETS
        }
        self._role_items: Dict[str, List[FocalItem]] = {
            name: [] for name in FOCAL_BUCKETS
        }
        self.rng = np.random.default_rng(int(seed))
        self.last_sample_stats: Dict[str, Any] = {}
        self.runtime_state: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self._records)

    @property
    def transition_ids(self) -> tuple[int, ...]:
        return tuple(self._order)

    @property
    def focal_index_sizes(self) -> Dict[str, int]:
        # Role references are maintained atomically by add/remove/load, so their
        # cardinality is already the focal pool size. Materializing every
        # FocalItem here made a metrics-only query O(replay_size).
        return {name: len(self._role_index[name]) for name in FOCAL_BUCKETS}

    def _validate_obs_tree(self, value: Mapping[str, Any], name: str) -> Dict[str, np.ndarray]:
        if not value:
            raise ValueError(f"{name} cannot be empty")
        result: Dict[str, np.ndarray] = {}
        for key, array in value.items():
            result[str(key)] = np.asarray(array).copy()
            if not np.all(np.isfinite(result[str(key)].astype(float, copy=False))):
                raise ValueError(f"{name}.{key} contains NaN or Inf")
        return result

    @staticmethod
    def _default_roles(
        active_mask: np.ndarray,
        phase: str,
        metadata: Mapping[str, Any],
    ) -> np.ndarray:
        roles = np.full(len(active_mask), ROLE_INACTIVE, dtype=np.uint8)
        active = np.asarray(active_mask, dtype=bool)
        if phase in {"post_capture", "pure_coverage", "pure_recovery"}:
            roles[active] = ROLE_COVERAGE
        else:
            task = str(metadata.get("task_label", ""))
            support = bool(metadata.get("support_candidate", False))
            roles[active] = ROLE_SUPPORT if support else (ROLE_PURSUING if task == "capture" else ROLE_COVERAGE)
        return roles

    def _role_bucket(self, phase: str, role: int) -> Optional[str]:
        if role == ROLE_INACTIVE:
            return None
        phase = str(phase).strip().lower()
        if phase == "pre_capture":
            return {
                ROLE_PURSUING: "pre_capture_pursuing",
                ROLE_SUPPORT: "pre_capture_support",
                ROLE_COVERAGE: "pre_capture_coverage",
            }.get(int(role))
        if phase in {"post_capture", "post"}:
            return "post_capture_coverage"
        if phase in {"pure_coverage", "pure_recovery", "pure_ce"}:
            return "pure_recovery_coverage"
        return None

    def _add_role_reference(
        self,
        bucket: str,
        slot: int,
        generation: int,
        agent_id: int,
    ) -> None:
        key = (int(slot), int(generation), int(agent_id))
        positions = self._role_index[bucket]
        if key in positions:
            return
        positions[key] = len(self._role_items[bucket])
        self._role_items[bucket].append(
            FocalItem(key[0], key[1], key[2], bucket)
        )

    def _remove_role_reference(
        self,
        bucket: str,
        slot: int,
        generation: int,
        agent_id: int,
    ) -> None:
        key = (int(slot), int(generation), int(agent_id))
        positions = self._role_index[bucket]
        position = positions.pop(key, None)
        if position is None:
            return
        items = self._role_items[bucket]
        last = items.pop()
        if position < len(items):
            items[position] = last
            positions[(last.slot_id, last.generation_id, last.agent_id)] = position

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
        agent_role_id: Any = None,
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
        active = _array(active_mask, np.bool_, (self.max_agents,), "active_mask").astype(bool)
        term = _array(terminated, np.bool_, (self.max_agents,), "terminated").astype(bool)
        trunc = _array(truncated, np.bool_, (self.max_agents,), "truncated").astype(bool)
        meta = _metadata(metadata)
        phase = str(meta["phase"])
        scene = str(meta["scene"])
        origin = str(meta["origin"])
        if agent_role_id is None:
            role_value = meta.get("agent_role_id")
            if role_value is None:
                roles = self._default_roles(active, phase, meta)
            else:
                roles = np.asarray(role_value, dtype=np.uint8)
        else:
            roles = np.asarray(agent_role_id, dtype=np.uint8)
        if roles.shape != (self.max_agents,):
            raise ValueError(f"agent_role_id must have shape ({self.max_agents},)")
        if not np.isin(roles, [0, 1, 2, 3]).all():
            raise ValueError("agent_role_id values must be 0..3")
        if not np.array_equal(roles == ROLE_INACTIVE, ~active):
            raise ValueError("inactive agents must have role 0 and active agents a non-zero primary role")

        slot = self._next_slot
        if slot in self._slot_record:
            self._remove(self._slot_record[slot])
        generation = int(self._generation.get(slot, 0)) + 1
        self._generation[slot] = generation
        transition_id = self._next_id
        self._next_id += 1
        transition = JointTransition(
            local_obs=local,
            next_local_obs=next_local,
            global_state=global_value,
            next_global_state=next_global_value,
            actions=action_array.copy(),
            rewards=_array(rewards, np.float32, (self.max_agents,), "rewards"),
            active_mask=active,
            terminated=term,
            truncated=trunc,
            agent_role_id=roles.copy(),
            phase=phase,
            scene=scene,
            origin=origin,
            metadata=meta,
            transition_id=transition_id,
            slot_id=slot,
            generation_id=generation,
        )
        self._records[transition_id] = transition
        self._order.append(transition_id)
        self._slot_record[slot] = transition_id
        self._next_slot = (slot + 1) % self.capacity
        regime = transition.metadata["regime"]
        self._regime_index[regime].add(transition_id)
        for event_id in transition.metadata["event_ids"]:
            self._event_index[event_id].add(transition_id)
        for agent_id, role in enumerate(roles):
            bucket = self._role_bucket(phase, int(role))
            if bucket is not None:
                self._add_role_reference(bucket, slot, generation, int(agent_id))
        return transition_id

    def _remove(self, transition_id: int) -> None:
        transition = self._records.pop(int(transition_id))
        self._order.remove(int(transition_id))
        slot = int(transition.slot_id)
        if self._slot_record.get(slot) == int(transition_id):
            del self._slot_record[slot]
        self._regime_index[transition.metadata["regime"]].discard(int(transition_id))
        for event_id in transition.metadata["event_ids"]:
            self._event_index[event_id].discard(int(transition_id))
        for agent_id, role in enumerate(transition.agent_role_id):
            bucket = self._role_bucket(transition.phase, int(role))
            if bucket is not None:
                self._remove_role_reference(
                    bucket,
                    slot,
                    int(transition.generation_id),
                    int(agent_id),
                )

    def _is_index_valid(self, slot_id: int, generation_id: int) -> bool:
        current_id = self._slot_record.get(int(slot_id))
        if current_id is None or current_id not in self._records:
            return False
        return int(self._generation.get(int(slot_id), 0)) == int(generation_id)

    def _role_pool(self, bucket: str) -> List[FocalItem]:
        if bucket not in self._role_index:
            raise ValueError(f"unknown focal bucket: {bucket}")
        return self._role_items[bucket]

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

    def get_item_transition(self, item: FocalItem) -> JointTransition:
        if not self._is_index_valid(item.slot_id, item.generation_id):
            raise ValueError("stale focal item reference")
        transition = self._records[self._slot_record[int(item.slot_id)]]
        if int(transition.slot_id) != int(item.slot_id) or int(transition.generation_id) != int(item.generation_id):
            raise ValueError("focal item slot/generation mismatch")
        if not bool(transition.active_mask[int(item.agent_id)]):
            raise ValueError("focal item refers to an inactive agent")
        return transition

    def sample_ids(
        self,
        batch_size: int,
        sampler: "JointReplaySampler | UniformJointReplaySampler | FocalReplaySampler | None" = None,
    ) -> List[int]:
        if len(self) == 0:
            raise ValueError("cannot sample an empty joint replay")
        if sampler is not None and hasattr(sampler, "sample_items"):
            items = sampler.sample_items(self, int(batch_size))
            return list(dict.fromkeys(int(self.get_item_transition(item).transition_id) for item in items))
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

    def batch_from_items(self, items: Sequence[FocalItem], device: str = "cpu") -> Dict[str, Any]:
        if not items:
            raise ValueError("focal items cannot be empty")
        transitions = [self.get_item_transition(item) for item in items]
        agents = self.max_agents

        def stack_tree(name: str) -> Dict[str, torch.Tensor]:
            keys = transitions[0].__dict__[name].keys()
            return {
                key: torch.as_tensor(np.stack([getattr(item, name)[key] for item in transitions]), dtype=torch.float32, device=device)
                for key in keys
            }

        def stack_global(name: str) -> Any:
            value = getattr(transitions[0], name)
            if isinstance(value, Mapping):
                return {
                    key: torch.as_tensor(
                        np.stack([getattr(item, name)[key] for item in transitions]),
                        dtype=torch.float32,
                        device=device,
                    )
                    for key in value
                }
            return torch.as_tensor(np.stack([getattr(item, name) for item in transitions]), dtype=torch.float32, device=device)

        rewards = np.stack([item.rewards for item in transitions])
        terminated = np.stack([item.terminated for item in transitions])
        truncated = np.stack([item.truncated for item in transitions])
        focal_ids = np.asarray([item.agent_id for item in items], dtype=np.int64)
        focal_mask = np.zeros((len(items), agents), dtype=bool)
        focal_mask[np.arange(len(items)), focal_ids] = True
        return {
            "local_obs": stack_tree("local_obs"),
            "next_local_obs": stack_tree("next_local_obs"),
            "global_state": stack_global("global_state"),
            "next_global_state": stack_global("next_global_state"),
            "actions": torch.as_tensor(np.stack([item.actions for item in transitions]), dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(rewards, dtype=torch.float32, device=device),
            "active_mask": torch.as_tensor(np.stack([item.active_mask for item in transitions]), dtype=torch.bool, device=device),
            "terminated": torch.as_tensor(terminated, dtype=torch.bool, device=device),
            "truncated": torch.as_tensor(truncated, dtype=torch.bool, device=device),
            "metadata": [dict(item.metadata) for item in transitions],
            "transition_ids": torch.as_tensor([item.transition_id for item in transitions], dtype=torch.long, device=device),
            "focal_agent_id": torch.as_tensor(focal_ids, dtype=torch.long, device=device),
            "focal_mask": torch.as_tensor(focal_mask, dtype=torch.bool, device=device),
            "focal_reward": torch.as_tensor(rewards[np.arange(len(items)), focal_ids], dtype=torch.float32, device=device),
            "focal_terminated": torch.as_tensor(terminated[np.arange(len(items)), focal_ids], dtype=torch.bool, device=device),
            "focal_truncated": torch.as_tensor(truncated[np.arange(len(items)), focal_ids], dtype=torch.bool, device=device),
            "bucket_id": [item.bucket_id for item in items],
            "phase_id": [item.phase for item in transitions],
            "origin_id": [item.origin for item in transitions],
            "scene_id": [item.scene for item in transitions],
            "slot_ids": torch.as_tensor([item.slot_id for item in items], dtype=torch.long, device=device),
            "generation_ids": torch.as_tensor([item.generation_id for item in items], dtype=torch.long, device=device),
        }

    def sample(
        self,
        batch_size: int,
        device: str = "cpu",
        sampler: "JointReplaySampler | UniformJointReplaySampler | FocalReplaySampler | None" = None,
    ) -> Dict[str, Any]:
        if len(self) == 0:
            raise ValueError("cannot sample an empty joint replay")
        if sampler is not None and hasattr(sampler, "sample_items"):
            items = sampler.sample_items(self, int(batch_size))
            batch = self.batch_from_items(items, device=device)
        else:
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
            "next_slot": self._next_slot,
            "records": list(self._records.items()),
            # Dense pool order affects how sampler RNG indexes map to focal
            # items after ring overwrite. Persist the derived ordering so a
            # resumed run reproduces the next batch exactly. Older schema-4
            # payloads omit this optional field and rebuild in record order.
            "role_items": (
                {
                    bucket: [
                        (item.slot_id, item.generation_id, item.agent_id)
                        for item in self._role_items[bucket]
                    ]
                    for bucket in FOCAL_BUCKETS
                }
                if self._next_id > self.capacity
                else None
            ),
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
        replay._next_slot = int(payload.get("next_slot", 0))
        saved_role_items = payload.get("role_items")
        expected_role_counts = {bucket: 0 for bucket in FOCAL_BUCKETS}
        for transition_id, transition in payload["records"]:
            replay._records[int(transition_id)] = transition
            replay._order.append(int(transition_id))
            replay._slot_record[int(transition.slot_id)] = int(transition_id)
            replay._generation[int(transition.slot_id)] = int(transition.generation_id)
            replay._regime_index[transition.metadata["regime"]].add(int(transition_id))
            for event_id in transition.metadata["event_ids"]:
                replay._event_index[event_id].add(int(transition_id))
            for agent_id, role in enumerate(transition.agent_role_id):
                bucket = replay._role_bucket(transition.phase, int(role))
                if bucket is not None:
                    expected_role_counts[bucket] += 1
                    if saved_role_items is None:
                        replay._add_role_reference(
                            bucket,
                            int(transition.slot_id),
                            int(transition.generation_id),
                            int(agent_id),
                        )
        if saved_role_items is not None:
            if set(saved_role_items) != set(FOCAL_BUCKETS):
                raise ValueError("joint replay role_items buckets mismatch")
            for bucket in FOCAL_BUCKETS:
                for raw_key in saved_role_items[bucket]:
                    if len(raw_key) != 3:
                        raise ValueError("joint replay role_items key must have three integers")
                    slot, generation, agent_id = (int(value) for value in raw_key)
                    if not replay._is_index_valid(slot, generation):
                        raise ValueError("joint replay role_items contains a stale reference")
                    transition = replay._records[replay._slot_record[slot]]
                    expected_bucket = replay._role_bucket(
                        transition.phase,
                        int(transition.agent_role_id[agent_id]),
                    )
                    if expected_bucket != bucket:
                        raise ValueError("joint replay role_items bucket mismatch")
                    replay._add_role_reference(bucket, slot, generation, agent_id)
                if len(replay._role_items[bucket]) != expected_role_counts[bucket]:
                    raise ValueError("joint replay role_items cardinality mismatch")
        replay.rng.bit_generator.state = payload["rng_state"]
        replay.runtime_state = dict(payload.get("runtime_state", {}) or {})
        return replay


class UniformJointReplaySampler:
    """Uniformly sample stored joint transitions without role/phase balancing.

    Sampling is without replacement whenever the replay contains at least one
    full batch. Role and phase metadata are summarized only after selection;
    they never influence which transition enters the optimizer.
    """

    def sample_ids(self, replay: JointReplayBuffer, batch_size: int) -> List[int]:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        pool = replay._pool("uniform")
        if not pool:
            raise ValueError("cannot sample an empty joint replay")
        replace = len(pool) < batch_size
        chosen = [
            int(value)
            for value in replay.rng.choice(
                np.asarray(pool, dtype=np.int64),
                size=batch_size,
                replace=replace,
            )
        ]
        transitions = [replay.get(idx) for idx in chosen]
        phase_counts: Dict[str, int] = {}
        scene_counts: Dict[str, int] = {}
        role_counts = {"pursuing": 0, "support": 0, "coverage": 0}
        active_agent_count = 0
        for transition in transitions:
            phase = str(transition.phase)
            scene = str(transition.scene)
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
            active = np.asarray(transition.active_mask, dtype=bool)
            roles = np.asarray(transition.agent_role_id, dtype=np.uint8)
            active_agent_count += int(np.count_nonzero(active))
            role_counts["pursuing"] += int(np.count_nonzero(active & (roles == ROLE_PURSUING)))
            role_counts["support"] += int(np.count_nonzero(active & (roles == ROLE_SUPPORT)))
            role_counts["coverage"] += int(np.count_nonzero(active & (roles == ROLE_COVERAGE)))
        replay.last_sample_stats = {
            "sampler": "uniform_joint",
            "requested_batch_size": batch_size,
            "actual_batch_size": len(chosen),
            "unique_joint_transitions": len(set(chosen)),
            "replacement_count": len(chosen) - len(set(chosen)),
            "pre_capture_transition_count": int(sum(str(item.phase) == "pre_capture" for item in transitions)),
            "post_capture_transition_count": int(sum(str(item.phase) == "post_capture" for item in transitions)),
            "pure_ce_transition_count": int(sum(str(item.scene) == "pure_ce" for item in transitions)),
            "sampled_phase_counts": phase_counts,
            "sampled_scene_counts": scene_counts,
            "sampled_active_agent_role_counts": role_counts,
            "active_agent_loss_terms": int(active_agent_count),
            "role_metadata_used_for_sampling": False,
        }
        return chosen


class JointReplaySampler:
    """Legacy 70/20/10 sampler kept for old smoke tests (not formal CTDE)."""

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


class FocalReplaySampler:
    """Quota-driven focal-agent sampler with explicit fallback, no silent uniform."""

    def __init__(
        self,
        quotas: Mapping[str, int],
        *,
        max_focal_items_per_joint_transition: int = 4,
        fallback_matrix: Mapping[str, Sequence[str]] | None = None,
        seed: int = 0,
        batch_size: int | None = None,
    ):
        unknown = set(quotas).difference(FOCAL_BUCKETS)
        if unknown:
            raise ValueError(f"unknown focal quotas: {sorted(unknown)}")
        self.quotas = {name: int(quotas.get(name, 0)) for name in FOCAL_BUCKETS}
        if sum(self.quotas.values()) <= 0:
            raise ValueError("focal quotas must sum to a positive batch size")
        if batch_size is not None and int(batch_size) != sum(self.quotas.values()):
            raise ValueError("explicit batch_size must equal the sum of focal quotas")
        self.batch_size = int(batch_size) if batch_size is not None else sum(self.quotas.values())
        self.max_focal_items_per_joint_transition = int(max_focal_items_per_joint_transition)
        if self.max_focal_items_per_joint_transition <= 0:
            raise ValueError("max_focal_items_per_joint_transition must be positive")
        fallback_matrix = fallback_matrix or {}
        self.fallback_matrix = {
            name: tuple(fallback_matrix.get(name, DEFAULT_FALLBACK_MATRIX[name]))
            for name in FOCAL_BUCKETS
        }
        self.rng = np.random.default_rng(int(seed))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != 1:
            raise ValueError("focal sampler state schema version mismatch")
        if "rng_state" not in state:
            raise ValueError("focal sampler state is missing rng_state")
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])

    @staticmethod
    def _candidate_pool(
        pool: Sequence[FocalItem],
        selected: Set[Tuple[int, int]],
        slot_counts: Dict[int, int],
        max_items: int,
    ) -> List[FocalItem]:
        return [
            item
            for item in pool
            if (item.slot_id, item.agent_id) not in selected
            and slot_counts.get(item.slot_id, 0) < max_items
        ]

    def _draw_unique(
        self,
        pool: List[FocalItem],
        need: int,
        selected: Set[Tuple[int, int]],
        slot_counts: Dict[int, int],
    ) -> Tuple[List[FocalItem], int]:
        """Draw unique focal items; count with-replacement attempts that collide."""
        chosen: List[FocalItem] = []
        # sample_items passes an already filtered candidate list. Repeating
        # this full scan doubled the hot-path work for every primary quota.
        candidates = pool
        take = min(need, len(candidates))
        if take:
            indexes = self.rng.choice(len(candidates), size=take, replace=False)
            for pos in indexes:
                item = candidates[int(pos)]
                chosen.append(item)
                selected.add((item.slot_id, item.agent_id))
                slot_counts[item.slot_id] = slot_counts.get(item.slot_id, 0) + 1
        remaining = need - take
        replacement_attempts = 0
        attempts = 0
        while remaining > 0 and attempts < 200 * max(remaining, 1):
            item = pool[int(self.rng.integers(0, len(pool)))]
            attempts += 1
            if (
                (item.slot_id, item.agent_id) in selected
                or slot_counts.get(item.slot_id, 0) >= self.max_focal_items_per_joint_transition
            ):
                replacement_attempts += 1
                continue
            chosen.append(item)
            selected.add((item.slot_id, item.agent_id))
            slot_counts[item.slot_id] = slot_counts.get(item.slot_id, 0) + 1
            remaining -= 1
        return chosen, replacement_attempts

    def _draw_sparse_unique(
        self,
        pool: Sequence[FocalItem],
        need: int,
        selected: Set[Tuple[int, int]],
        slot_counts: Dict[int, int],
    ) -> List[FocalItem] | None:
        """Uniformly draw from a large dense pool without scanning it.

        Rejection sampling is uniform over the currently eligible items. Small
        or dense-exclusion pools return None so the exact exhaustive path
        remains available for edge cases.
        """
        if need <= 0:
            return []
        if len(pool) < max(4096, 16 * int(need)):
            return None
        chosen: List[FocalItem] = []
        used_positions: Set[int] = set()
        attempts = 0
        attempt_limit = max(512, 32 * int(need))
        while len(chosen) < need and attempts < attempt_limit:
            position = int(self.rng.integers(0, len(pool)))
            attempts += 1
            if position in used_positions:
                continue
            used_positions.add(position)
            item = pool[position]
            if (
                (item.slot_id, item.agent_id) in selected
                or slot_counts.get(item.slot_id, 0)
                >= self.max_focal_items_per_joint_transition
            ):
                continue
            chosen.append(item)
            selected.add((item.slot_id, item.agent_id))
            slot_counts[item.slot_id] = slot_counts.get(item.slot_id, 0) + 1
        if len(chosen) < need:
            candidates = self._candidate_pool(
                pool,
                selected,
                slot_counts,
                self.max_focal_items_per_joint_transition,
            )
            take = min(need - len(chosen), len(candidates))
            if take:
                indexes = self.rng.choice(len(candidates), size=take, replace=False)
                for position in indexes:
                    item = candidates[int(position)]
                    chosen.append(item)
                    selected.add((item.slot_id, item.agent_id))
                    slot_counts[item.slot_id] = slot_counts.get(item.slot_id, 0) + 1
        return chosen

    def sample_items(self, replay: JointReplayBuffer, batch_size: int | None = None) -> List[FocalItem]:
        batch_size = self.batch_size if batch_size is None else int(batch_size)
        if batch_size != self.batch_size:
            raise ValueError("FocalReplaySampler batch size must match configured quota sum")
        if len(replay) == 0:
            raise ValueError("cannot sample an empty joint replay")
        selected: Set[Tuple[int, int]] = set()
        slot_counts: Dict[int, int] = {}
        chosen: List[FocalItem] = []
        actual_counts: Dict[str, int] = {name: 0 for name in FOCAL_BUCKETS}
        requested_counts: Dict[str, int] = dict(self.quotas)
        replacement_count = 0
        fallback_count = 0
        fallback_targets: List[str] = []
        # A replay does not mutate during one synchronous sample call. Reuse
        # each dense role pool across primary and fallback quota paths instead
        # of rebuilding hundreds of thousands of FocalItem objects.
        role_pools = {bucket: replay._role_pool(bucket) for bucket in FOCAL_BUCKETS}
        for bucket in FOCAL_BUCKETS:
            target = self.quotas[bucket]
            if target <= 0:
                continue
            need = target
            raw_pool = role_pools[bucket]
            sparse_part = self._draw_sparse_unique(
                raw_pool,
                need,
                selected,
                slot_counts,
            )
            if sparse_part is None:
                pool = self._candidate_pool(
                    raw_pool,
                    selected,
                    slot_counts,
                    self.max_focal_items_per_joint_transition,
                )
                part, replacements = (
                    self._draw_unique(pool, need, selected, slot_counts)
                    if pool
                    else ([], 0)
                )
            else:
                part, replacements = sparse_part, 0
            if part:
                chosen.extend(part)
                actual_counts[bucket] = actual_counts.get(bucket, 0) + len(part)
                replacement_count += replacements
                need -= len(part)
            if need <= 0:
                continue
            for fallback_bucket in self.fallback_matrix.get(bucket, ()):
                if need <= 0:
                    break
                raw_fallback_pool = role_pools[fallback_bucket]
                sparse_part = self._draw_sparse_unique(
                    raw_fallback_pool,
                    need,
                    selected,
                    slot_counts,
                )
                if sparse_part is None:
                    fallback_pool = self._candidate_pool(
                        raw_fallback_pool,
                        selected,
                        slot_counts,
                        self.max_focal_items_per_joint_transition,
                    )
                    take = min(need, len(fallback_pool))
                    indexes = (
                        self.rng.choice(
                            len(fallback_pool),
                            size=take,
                            replace=False,
                        )
                        if take
                        else []
                    )
                    part = [fallback_pool[int(pos)] for pos in indexes]
                    for item in part:
                        selected.add((item.slot_id, item.agent_id))
                        slot_counts[item.slot_id] = slot_counts.get(item.slot_id, 0) + 1
                else:
                    part = sparse_part
                if not part:
                    continue
                for item in part:
                    chosen.append(item)
                    actual_counts[item.bucket_id] = actual_counts.get(item.bucket_id, 0) + 1
                    fallback_count += 1
                    fallback_targets.append(bucket)
                need -= len(part)
            if need > 0:
                raise RuntimeError(
                    f"focal bucket {bucket} cannot be filled through the explicit fallback matrix; "
                    "silent uniform fallback is forbidden"
                )
        if len(chosen) != batch_size:
            raise RuntimeError(f"focal sampler produced {len(chosen)} items, expected {batch_size}")
        unique_transitions = {replay.get_item_transition(item).transition_id for item in chosen}
        role_counts: Dict[str, int] = {name: 0 for name in ("pursuing", "support", "coverage")}
        for item in chosen:
            transition = replay.get_item_transition(item)
            role = int(transition.agent_role_id[item.agent_id])
            role_counts["pursuing" if role == ROLE_PURSUING else ("support" if role == ROLE_SUPPORT else "coverage")] += 1
        replay.last_sample_stats = {
            "requested_batch_size": batch_size,
            "actual_batch_size": len(chosen),
            "requested_counts": requested_counts,
            "actual_counts": actual_counts,
            "sampled_role_counts": role_counts,
            "unique_joint_transitions": len(unique_transitions),
            "max_items_per_joint_transition": max(slot_counts.values()) if slot_counts else 0,
            "replacement_count": replacement_count,
            "fallback_count": fallback_count,
            "fallback_targets": fallback_targets,
            "silent_uniform_fallback": False,
        }
        return chosen
