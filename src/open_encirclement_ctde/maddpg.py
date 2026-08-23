"""Minimal audited MADDPG for three hunters on ``corrected_roundup_v1``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MADDPGConfig:
    obs_dim: int = 26
    action_dim: int = 2
    num_agents: int = 3
    hidden_dim: int = 128
    actor_lr: float = 1e-4
    critic_lr: float = 3e-3
    gamma: float = 0.99
    tau: float = 0.01
    replay_capacity: int = 1_000_000
    batch_size: int = 256
    warmup_steps: int = 1_024
    updates_per_transition: float = 0.1
    action_limit: float = 0.04
    noise_initial: float = 0.75
    noise_minimum: float = 0.01
    noise_decay: float = 0.999995
    actor_scheduler_step: int = 1_000
    actor_scheduler_gamma: float = 0.8
    critic_scheduler_step: int = 5_000
    critic_scheduler_gamma: float = 0.33

    @property
    def joint_obs_dim(self) -> int:
        return self.obs_dim * self.num_agents

    @property
    def joint_action_dim(self) -> int:
        return self.action_dim * self.num_agents


class Actor(nn.Module):
    def __init__(self, config: MADDPGConfig) -> None:
        super().__init__()
        self.action_limit = config.action_limit
        self.fc1 = nn.Linear(config.obs_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = F.leaky_relu(self.fc1(observation))
        hidden = F.leaky_relu(self.fc2(hidden))
        return torch.tanh(self.output(hidden)) * self.action_limit


class CentralCritic(nn.Module):
    def __init__(self, config: MADDPGConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.joint_obs_dim + config.joint_action_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, 1)

    def forward(self, joint_observation: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.fc1(torch.cat((joint_observation, joint_action), dim=-1)))
        hidden = F.relu(self.fc2(hidden))
        return self.output(hidden)


class ReplayBuffer:
    """Transition-level ring buffer with separate terminated/truncated bits."""

    def __init__(self, config: MADDPGConfig, seed: int) -> None:
        self.config = config
        capacity = config.replay_capacity
        self.observations = np.zeros((capacity, config.num_agents, config.obs_dim), dtype=np.float32)
        self.next_observations = np.zeros_like(self.observations)
        self.actions = np.zeros((capacity, config.num_agents, config.action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, config.num_agents), dtype=np.float32)
        self.terminated = np.zeros((capacity,), dtype=np.bool_)
        self.truncated = np.zeros((capacity,), dtype=np.bool_)
        self.count = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return min(self.count, self.config.replay_capacity)

    def add(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        index = self.count % self.config.replay_capacity
        self.observations[index] = observations
        self.actions[index] = actions
        self.rewards[index] = rewards
        self.next_observations[index] = next_observations
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.count += 1

    def sample(self, batch_size: int | None = None) -> dict[str, np.ndarray]:
        size = batch_size or self.config.batch_size
        if len(self) < size:
            raise RuntimeError(f"replay has {len(self)} transitions, needs {size}")
        indices = self.rng.choice(len(self), size=size, replace=False)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "terminated": self.terminated[indices],
            "truncated": self.truncated[indices],
        }

    def state_dict(self) -> dict[str, Any]:
        valid = len(self)
        return {
            "config": asdict(self.config),
            "count": self.count,
            "rng_state": self.rng.bit_generator.state,
            "observations": self.observations[:valid].copy(),
            "actions": self.actions[:valid].copy(),
            "rewards": self.rewards[:valid].copy(),
            "next_observations": self.next_observations[:valid].copy(),
            "terminated": self.terminated[:valid].copy(),
            "truncated": self.truncated[:valid].copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["config"] != asdict(self.config):
            raise ValueError("replay config mismatch")
        valid = len(state["observations"])
        if valid > self.config.replay_capacity:
            raise ValueError("checkpoint replay exceeds configured capacity")
        self.count = int(state["count"])
        self.rng.bit_generator.state = state["rng_state"]
        for name in ("observations", "actions", "rewards", "next_observations", "terminated", "truncated"):
            getattr(self, name)[:valid] = state[name]


class MADDPG:
    def __init__(self, config: MADDPGConfig, device: torch.device, seed: int) -> None:
        self.config = config
        self.device = device
        self.actors = nn.ModuleList([Actor(config) for _ in range(config.num_agents)]).to(device)
        self.target_actors = nn.ModuleList([Actor(config) for _ in range(config.num_agents)]).to(device)
        self.critics = nn.ModuleList([CentralCritic(config) for _ in range(config.num_agents)]).to(device)
        self.target_critics = nn.ModuleList([CentralCritic(config) for _ in range(config.num_agents)]).to(device)
        self.target_actors.load_state_dict(self.actors.state_dict())
        self.target_critics.load_state_dict(self.critics.state_dict())
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=config.actor_lr) for actor in self.actors]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=config.critic_lr) for critic in self.critics]
        self.actor_schedulers = [
            torch.optim.lr_scheduler.StepLR(opt, step_size=config.actor_scheduler_step, gamma=config.actor_scheduler_gamma)
            for opt in self.actor_optimizers
        ]
        self.critic_schedulers = [
            torch.optim.lr_scheduler.StepLR(opt, step_size=config.critic_scheduler_step, gamma=config.critic_scheduler_gamma)
            for opt in self.critic_optimizers
        ]
        self.replay = ReplayBuffer(config, seed + 55_000)
        self.update_count = 0

    @torch.no_grad()
    def actions(self, observations: np.ndarray, *, deterministic: bool, env_steps: int) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float32)
        single = obs.ndim == 2
        if single:
            obs = obs[None, ...]
        if obs.shape[1:] != (self.config.num_agents, self.config.obs_dim):
            raise ValueError(f"unexpected observation shape {obs.shape}")
        outputs = []
        for idx, actor in enumerate(self.actors):
            action = actor(torch.as_tensor(obs[:, idx], device=self.device)).cpu().numpy()
            outputs.append(action)
        result = np.stack(outputs, axis=1)
        if not deterministic:
            scale = max(
                self.config.noise_minimum,
                self.config.noise_initial * self.config.noise_decay ** env_steps,
            )
            normalized_noise = np.random.uniform(-scale, scale, size=result.shape)
            result = result + normalized_noise * self.config.action_limit
        result = np.clip(result, -self.config.action_limit, self.config.action_limit).astype(np.float32)
        return result[0] if single else result

    def train_step(self) -> dict[str, float]:
        batch = self.replay.sample()
        observations = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(batch["next_observations"], dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.float32, device=self.device)
        joint_obs = observations.flatten(start_dim=1)
        joint_actions = actions.flatten(start_dim=1)
        joint_next_obs = next_observations.flatten(start_dim=1)

        with torch.no_grad():
            next_actions = torch.cat(
                [self.target_actors[idx](next_observations[:, idx]) for idx in range(self.config.num_agents)], dim=1
            )

        critic_losses = []
        actor_losses = []
        critic_grad_norms = []
        actor_grad_norms = []
        q_means = []
        td_abs_means = []
        for idx in range(self.config.num_agents):
            with torch.no_grad():
                next_q = self.target_critics[idx](joint_next_obs, next_actions).squeeze(-1)
                target = rewards[:, idx] + self.config.gamma * (1.0 - terminated) * next_q
            q_value = self.critics[idx](joint_obs, joint_actions).squeeze(-1)
            critic_loss = F.mse_loss(q_value, target)
            self.critic_optimizers[idx].zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_grad = torch.nn.utils.clip_grad_norm_(self.critics[idx].parameters(), max_norm=1e9)
            self.critic_optimizers[idx].step()
            self.critic_schedulers[idx].step()

            for parameter in self.critics[idx].parameters():
                parameter.requires_grad_(False)
            policy_actions = actions.detach().clone()
            policy_actions[:, idx] = self.actors[idx](observations[:, idx])
            actor_loss = -self.critics[idx](joint_obs, policy_actions.flatten(start_dim=1)).mean()
            self.actor_optimizers[idx].zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(self.actors[idx].parameters(), max_norm=1e9)
            self.actor_optimizers[idx].step()
            self.actor_schedulers[idx].step()
            for parameter in self.critics[idx].parameters():
                parameter.requires_grad_(True)

            critic_losses.append(float(critic_loss.detach().cpu()))
            actor_losses.append(float(actor_loss.detach().cpu()))
            critic_grad_norms.append(float(critic_grad.detach().cpu()))
            actor_grad_norms.append(float(actor_grad.detach().cpu()))
            q_means.append(float(q_value.detach().mean().cpu()))
            td_abs_means.append(float((target - q_value.detach()).abs().mean().cpu()))

        self._soft_update()
        self.update_count += 1
        metrics = {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(np.mean(actor_losses)),
            "critic_grad_norm": float(np.mean(critic_grad_norms)),
            "actor_grad_norm": float(np.mean(actor_grad_norms)),
            "q_mean": float(np.mean(q_means)),
            "td_abs_mean": float(np.mean(td_abs_means)),
            "actor_lr": float(self.actor_optimizers[0].param_groups[0]["lr"]),
            "critic_lr": float(self.critic_optimizers[0].param_groups[0]["lr"]),
        }
        if not np.isfinite(np.asarray(list(metrics.values()), dtype=np.float64)).all():
            raise FloatingPointError(f"non-finite MADDPG metrics: {metrics}")
        return metrics

    def _soft_update(self) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for target_module, module in zip(self.target_actors, self.actors):
                for target, source in zip(target_module.parameters(), module.parameters()):
                    target.mul_(1.0 - tau).add_(source, alpha=tau)
            for target_module, module in zip(self.target_critics, self.critics):
                for target, source in zip(target_module.parameters(), module.parameters()):
                    target.mul_(1.0 - tau).add_(source, alpha=tau)

    def model_state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "actors": self.actors.state_dict(),
            "critics": self.critics.state_dict(),
            "target_actors": self.target_actors.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "update_count": self.update_count,
        }

    def full_state_dict(self) -> dict[str, Any]:
        return {
            **self.model_state_dict(),
            "actor_optimizers": [optimizer.state_dict() for optimizer in self.actor_optimizers],
            "critic_optimizers": [optimizer.state_dict() for optimizer in self.critic_optimizers],
            "actor_schedulers": [scheduler.state_dict() for scheduler in self.actor_schedulers],
            "critic_schedulers": [scheduler.state_dict() for scheduler in self.critic_schedulers],
            "replay": self.replay.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any], *, full: bool) -> None:
        if state["config"] != asdict(self.config):
            raise ValueError("MADDPG config mismatch")
        self.actors.load_state_dict(state["actors"])
        self.critics.load_state_dict(state["critics"])
        self.target_actors.load_state_dict(state["target_actors"])
        self.target_critics.load_state_dict(state["target_critics"])
        self.update_count = int(state["update_count"])
        if full:
            required = ("actor_optimizers", "critic_optimizers", "actor_schedulers", "critic_schedulers", "replay")
            if any(key not in state for key in required):
                raise ValueError("requested full MADDPG resume from model-only checkpoint")
            for optimizer, saved in zip(self.actor_optimizers, state["actor_optimizers"]):
                optimizer.load_state_dict(saved)
            for optimizer, saved in zip(self.critic_optimizers, state["critic_optimizers"]):
                optimizer.load_state_dict(saved)
            for scheduler, saved in zip(self.actor_schedulers, state["actor_schedulers"]):
                scheduler.load_state_dict(saved)
            for scheduler, saved in zip(self.critic_schedulers, state["critic_schedulers"]):
                scheduler.load_state_dict(saved)
            self.replay.load_state_dict(state["replay"])
