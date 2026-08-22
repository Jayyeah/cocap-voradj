"""Fixed-slot joint replay whose action is exactly the environment action."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import Tensor


@dataclass(frozen=True)
class JointBatch:
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    next_observations: Tensor
    terminated: Tensor
    truncated: Tensor

    @property
    def bootstrap_mask(self) -> Tensor:
        """Horizon truncation bootstraps; only true termination zeros the target."""

        return (~self.terminated).to(dtype=self.rewards.dtype)


class JointReplayBuffer:
    def __init__(
        self,
        capacity: int,
        *,
        agent_count: int = 3,
        observation_dim: int = 26,
        action_dim: int = 2,
        seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.agent_count = int(agent_count)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.observations = np.empty(
            (capacity, agent_count, observation_dim), dtype=np.float32
        )
        self.actions = np.empty((capacity, agent_count, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, agent_count), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.truncated = np.empty(capacity, dtype=np.bool_)
        self.cursor = 0
        self.size = 0
        self.total_insertions = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def _array(
        self, value: ArrayLike, shape: tuple[int, ...], *, name: str
    ) -> NDArray[np.float32]:
        result = np.asarray(value, dtype=np.float32)
        if result.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be finite")
        return result

    def add(
        self,
        observation: ArrayLike,
        executed_action: ArrayLike,
        reward: ArrayLike,
        next_observation: ArrayLike,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        obs = self._array(
            observation,
            (self.agent_count, self.observation_dim),
            name="observation",
        )
        action = self._array(
            executed_action,
            (self.agent_count, self.action_dim),
            name="executed_action",
        )
        action_norms = np.linalg.norm(action, axis=-1)
        if np.any(action_norms > 1.0 + 1.0e-6):
            raise ValueError("replay action is not a canonical executable action")
        rewards = self._array(reward, (self.agent_count,), name="reward")
        next_obs = self._array(
            next_observation,
            (self.agent_count, self.observation_dim),
            name="next_observation",
        )
        index = self.cursor
        self.observations[index] = obs
        self.actions[index] = action
        self.rewards[index] = rewards
        self.next_observations[index] = next_obs
        self.terminated[index] = bool(terminated)
        self.truncated[index] = bool(truncated)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_insertions += 1

    def sample_indices(self, batch_size: int) -> NDArray[np.int64]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size < batch_size:
            raise ValueError(f"replay contains {self.size}, cannot sample {batch_size}")
        return self._rng.integers(0, self.size, size=batch_size, dtype=np.int64)

    def batch_from_indices(
        self, indices: ArrayLike, *, device: torch.device | str = "cpu"
    ) -> JointBatch:
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= self.size):
            raise ValueError("replay indices are invalid")
        return JointBatch(
            observations=torch.as_tensor(
                self.observations[selected], device=device, dtype=torch.float32
            ),
            actions=torch.as_tensor(
                self.actions[selected], device=device, dtype=torch.float32
            ),
            rewards=torch.as_tensor(
                self.rewards[selected], device=device, dtype=torch.float32
            ),
            next_observations=torch.as_tensor(
                self.next_observations[selected], device=device, dtype=torch.float32
            ),
            terminated=torch.as_tensor(
                self.terminated[selected], device=device, dtype=torch.bool
            ),
            truncated=torch.as_tensor(
                self.truncated[selected], device=device, dtype=torch.bool
            ),
        )

    def sample(
        self, batch_size: int, *, device: torch.device | str = "cpu"
    ) -> JointBatch:
        return self.batch_from_indices(self.sample_indices(batch_size), device=device)

    def state_dict(self) -> dict[str, Any]:
        valid = self.capacity if self.size == self.capacity else self.size
        return {
            "capacity": self.capacity,
            "agent_count": self.agent_count,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "cursor": self.cursor,
            "size": self.size,
            "total_insertions": self.total_insertions,
            "observations": self.observations[:valid].copy(),
            "actions": self.actions[:valid].copy(),
            "rewards": self.rewards[:valid].copy(),
            "next_observations": self.next_observations[:valid].copy(),
            "terminated": self.terminated[:valid].copy(),
            "truncated": self.truncated[:valid].copy(),
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        metadata = (
            state["capacity"],
            state["agent_count"],
            state["observation_dim"],
            state["action_dim"],
        )
        expected = (
            self.capacity,
            self.agent_count,
            self.observation_dim,
            self.action_dim,
        )
        if metadata != expected:
            raise ValueError(f"replay metadata mismatch: {metadata} != {expected}")
        size = int(state["size"])
        cursor = int(state["cursor"])
        total_insertions = int(state.get("total_insertions", size))
        valid = self.capacity if size == self.capacity else size
        if (
            not 0 <= size <= self.capacity
            or not 0 <= cursor < self.capacity
            or total_insertions < size
            or cursor != total_insertions % self.capacity
            or size != min(total_insertions, self.capacity)
        ):
            raise ValueError("invalid replay size/cursor/total_insertions")
        fields = (
            ("observations", self.observations),
            ("actions", self.actions),
            ("rewards", self.rewards),
            ("next_observations", self.next_observations),
            ("terminated", self.terminated),
            ("truncated", self.truncated),
        )
        for name, destination in fields:
            source = np.asarray(state[name])
            if source.shape != destination[:valid].shape:
                raise ValueError(f"invalid replay field shape for {name}")
            destination[:valid] = source
        self.size = size
        self.cursor = cursor
        self.total_insertions = total_insertions
        self._rng.bit_generator.state = deepcopy(state["rng_state"])
