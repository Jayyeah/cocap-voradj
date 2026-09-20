"""Replay, update, checkpoint, and recovery helpers for PS-Local-Discrete-AC."""
from __future__ import annotations

import copy
import hashlib
import io
import os
import pickle
import random
import tempfile
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from cocap_voradj.models.shared_local_ac import (
    SharedLocalACNetworkConfig,
    SharedLocalActor,
    SharedLocalQ,
    hard_copy,
)


REPLAY_CLASSES = ("pursuing", "pre_capture_cover", "post_capture_real", "recovery_pure")


def clone_obs(obs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {str(key): np.asarray(value).copy() for key, value in obs.items()}


def tensor_obs(items: Sequence[Mapping[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(
            np.stack([item[key] for item in items]),
            dtype=torch.long if key == "types" else torch.bool if key == "masks" else torch.float32,
            device=device,
        )
        for key in items[0]
    }


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def global_grad_norm(module: torch.nn.Module) -> float:
    values = [parameter.grad.detach().float().square().sum() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.sqrt(torch.stack(values).sum()).detach()) if values else 0.0


@dataclass
class Transition:
    obs: dict[str, np.ndarray]
    action: int
    reward: float
    next_obs: dict[str, np.ndarray]
    done: bool
    terminated: bool
    truncated: bool
    phase: str
    replay_class: str
    behavior_probability: float
    epsilon: float
    actor_probability: float
    episode_id: int
    agent_id: int
    metadata: dict[str, Any]


class ReplayWarmupError(RuntimeError):
    pass


class SharedLocalReplay:
    """A single row-level replay with metadata-indexed semantic sampling."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data: deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, transition: Transition) -> None:
        if transition.replay_class not in REPLAY_CLASSES:
            raise ValueError(f"unknown replay class: {transition.replay_class}")
        self.data.append(transition)

    def class_counts(self) -> dict[str, int]:
        counts = Counter(item.replay_class for item in self.data)
        return {name: int(counts.get(name, 0)) for name in REPLAY_CLASSES}

    def _indices(self, replay_class: Optional[str] = None) -> list[int]:
        if replay_class is None:
            return list(range(len(self.data)))
        return [index for index, item in enumerate(self.data) if item.replay_class == replay_class]

    def sample_uniform(self, batch_size: int, rng: np.random.RandomState) -> list[Transition]:
        if len(self.data) < int(batch_size):
            raise ReplayWarmupError(f"uniform replay warmup: {len(self.data)} < {batch_size}")
        indices = rng.choice(len(self.data), size=int(batch_size), replace=False)
        return [self.data[int(index)] for index in indices]

    def sample_stratified(
        self,
        counts: Mapping[str, int],
        rng: np.random.RandomState,
        *,
        fallback: str = "defer",
    ) -> tuple[list[Transition], dict[str, Any]]:
        requested = {name: int(counts.get(name, 0)) for name in REPLAY_CLASSES}
        if any(value < 0 for value in requested.values()):
            raise ValueError("replay sample counts must be non-negative")
        available = self.class_counts()
        missing = {name: requested[name] - available[name] for name in REPLAY_CLASSES if requested[name] > available[name]}
        if missing:
            if fallback == "defer":
                raise ReplayWarmupError(f"semantic replay warmup: missing={missing}, available={available}")
            if fallback != "uniform":
                raise ValueError(f"unknown replay fallback: {fallback}")
            batch = self.sample_uniform(sum(requested.values()), rng)
            return batch, {
                "sampler": "uniform_fallback",
                "requested_counts": requested,
                "actual_counts": dict(Counter(item.replay_class for item in batch)),
                "missing": missing,
            }
        batch: list[Transition] = []
        for name, count in requested.items():
            if count <= 0:
                continue
            indices = rng.choice(self._indices(name), size=count, replace=False)
            batch.extend(self.data[int(index)] for index in indices)
        return batch, {
            "sampler": "semantic_stratified",
            "requested_counts": requested,
            "actual_counts": dict(Counter(item.replay_class for item in batch)),
            "missing": {},
        }

    def serialized_size_estimate(self) -> int:
        if not self.data:
            return 0
        sample = list(self.data)[: min(32, len(self.data))]
        return int(len(pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)) * len(self.data) / len(sample))


class RecoveryInitPool:
    """In-memory Final IQN recovery pool; snapshots never become separate files."""

    def __init__(self, capacity: int = 1000, captured_ratio: float = 0.75, map_random_ratio: float = 0.5):
        self.data: deque[dict[str, Any]] = deque(maxlen=int(capacity))
        self.captured_ratio = float(captured_ratio)
        self.map_random_ratio = float(map_random_ratio)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, snapshot: Mapping[str, Any]) -> None:
        self.data.append(
            {
                "step": int(snapshot.get("step", 0)),
                "positions": copy.deepcopy(snapshot.get("positions", [])),
                "active_mask": copy.deepcopy(snapshot.get("active_mask", [])),
            }
        )

    def choose(self, rng: random.Random) -> tuple[str, Optional[dict[str, Any]]]:
        if self.data and rng.random() < self.captured_ratio:
            return "capture_snapshot", copy.deepcopy(rng.choice(list(self.data)))
        if rng.random() < self.map_random_ratio:
            return "ordinary_map_random", None
        return "synthetic_cluster", None


@dataclass(frozen=True)
class SharedLocalACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    adam_eps: float = 1e-8
    batch_size: int = 128
    min_replay_size: int = 3000
    train_freq: int = 4
    max_grad_norm: float = 0.5
    replay_capacity: int = 1_000_000
    replay_fallback: str = "defer"


class SharedLocalACTrainer:
    """Single-Q discrete off-policy AC with explicit target actor and critic."""

    schema_version = "ps-local-discrete-ac-v1"

    def __init__(self, network: SharedLocalACNetworkConfig, config: SharedLocalACConfig, device: str = "cpu"):
        self.device = torch.device(device)
        self.network_config = network
        self.config = config
        self.actor = SharedLocalActor(network).to(self.device)
        self.critic = SharedLocalQ(network).to(self.device)
        self.target_actor = hard_copy(self.actor).to(self.device)
        self.target_critic = hard_copy(self.critic).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr, eps=config.adam_eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr, eps=config.adam_eps)
        self.update_count = 0
        self._set_eval_modes()

    def _set_eval_modes(self) -> None:
        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

    @staticmethod
    @torch.no_grad()
    def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.mul_(1.0 - float(tau)).add_(source_parameter, alpha=float(tau))
        for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
            target_buffer.copy_(source_buffer)

    def update(self, transitions: Sequence[Transition]) -> dict[str, float]:
        if len(transitions) != int(self.config.batch_size):
            raise ValueError("AC update batch does not match configured batch_size")
        obs = tensor_obs([item.obs for item in transitions], self.device)
        next_obs = tensor_obs([item.next_obs for item in transitions], self.device)
        actions = torch.as_tensor([item.action for item in transitions], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([item.reward for item in transitions], dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor([item.terminated for item in transitions], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_pi = self.target_actor.probabilities(next_obs)
            next_q = self.target_critic(next_obs)
            target_value = (next_pi * next_q).sum(dim=-1)
            td_target = rewards + float(self.config.gamma) * (1.0 - terminated) * target_value
        q_values = self.critic(obs)
        q_taken = q_values.gather(1, actions[:, None]).squeeze(1)
        critic_loss = F.smooth_l1_loss(q_taken, td_target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm))
        self.critic_optimizer.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        pi = self.actor.probabilities(obs)
        q_for_actor = self.critic(obs)
        actor_objective = (pi * q_for_actor).sum(dim=-1).mean()
        actor_loss = -actor_objective
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm))
        self.actor_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)
        self.soft_update(self.actor, self.target_actor, self.config.tau)
        self.soft_update(self.critic, self.target_critic, self.config.tau)
        self.update_count += 1
        with torch.no_grad():
            entropy = -(pi.clamp_min(1e-8) * pi.clamp_min(1e-8).log()).sum(dim=-1).mean()
            residual = q_taken - td_target
            q_all = q_values.detach()
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "actor_objective": float(actor_objective.detach()),
            "actor_entropy": float(entropy.detach()),
            "critic_grad_norm": critic_grad,
            "actor_grad_norm": actor_grad,
            "q_mean": float(q_all.mean()),
            "q_std": float(q_all.std(unbiased=False)),
            "q_min": float(q_all.min()),
            "q_max": float(q_all.max()),
            "td_target_mean": float(td_target.mean()),
            "td_target_std": float(td_target.std(unbiased=False)),
            "bellman_residual_mean": float(residual.mean()),
            "bellman_residual_abs_mean": float(residual.abs().mean()),
            "finite": float(
                all(
                    np.isfinite(value)
                    for value in (
                        float(critic_loss.detach()),
                        float(actor_loss.detach()),
                        float(q_all.mean().detach()),
                        float(td_target.mean().detach()),
                    )
                )
            ),
            "update_count": float(self.update_count),
        }

    def checkpoint_payload(
        self,
        *,
        config: Mapping[str, Any],
        global_step: int,
        episode: int,
        include_replay: bool = False,
        include_runtime: bool = False,
        replay: Optional[SharedLocalReplay] = None,
        recovery_pool: Optional[RecoveryInitPool] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema_version,
            "network_config": asdict(self.network_config),
            "trainer_config": asdict(self.config),
            "config": copy.deepcopy(dict(config)),
            "global_step": int(global_step),
            "episode": int(episode),
            "update_count": int(self.update_count),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_hash": state_hash(self.actor),
            "critic_hash": state_hash(self.critic),
        }
        if include_runtime or include_replay:
            payload["actor_optimizer"] = self.actor_optimizer.state_dict()
            payload["critic_optimizer"] = self.critic_optimizer.state_dict()
            payload["python_rng"] = random.getstate()
            payload["numpy_rng"] = np.random.get_state()
            payload["torch_rng"] = torch.get_rng_state()
            if torch.cuda.is_available():
                payload["cuda_rng"] = torch.cuda.get_rng_state_all()
        if include_replay:
            payload["replay"] = replay
            payload["recovery_pool"] = recovery_pool
        return payload

    def load_payload(self, payload: Mapping[str, Any], *, load_optimizer: bool = True) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.target_actor.load_state_dict(payload["target_actor"])
        self.target_critic.load_state_dict(payload["target_critic"])
        if load_optimizer and "actor_optimizer" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.update_count = int(payload.get("update_count", 0))
        self._set_eval_modes()


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Write one checkpoint atomically without retaining rotating large copies."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temp = Path(raw)
    try:
        torch.save(dict(payload), temp)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def estimate_replay_row_bytes(transition: Transition) -> int:
    return len(pickle.dumps([transition], protocol=pickle.HIGHEST_PROTOCOL))
