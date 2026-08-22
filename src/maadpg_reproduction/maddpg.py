"""Standard MADDPG: independent actors and one joint critic per pursuer."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import Tensor, nn

from .networks import Actor, JointCritic
from .replay import JointBatch
from .training_config import MADDPGConfig, default_maddpg_config


@dataclass(frozen=True)
class AgentUpdateDiagnostics:
    critic_loss: float
    actor_loss: float
    q_mean: float
    target_q_mean: float
    td_abs_mean: float
    td_abs_max: float
    critic_gradient_norm: float
    actor_gradient_norm: float
    action_norm_mean: float
    action_norm_max: float
    policy_q_mean: float
    shuffled_action_q_mean: float
    policy_minus_replay_q: float
    policy_minus_shuffled_q: float


@dataclass(frozen=True)
class MADDPGUpdateDiagnostics:
    update_step: int
    agents: tuple[AgentUpdateDiagnostics, ...]


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not torch.all(torch.isfinite(gradient)):
            return math.inf
        squared += float(torch.sum(gradient * gradient).item())
    return math.sqrt(squared)


def _require_finite(name: str, value: Tensor | float) -> None:
    finite = torch.all(torch.isfinite(value)).item() if isinstance(value, Tensor) else math.isfinite(value)
    if not finite:
        raise FloatingPointError(f"non-finite MADDPG diagnostic: {name}")


class MADDPGLearner:
    def __init__(
        self,
        config: MADDPGConfig | None = None,
        *,
        device: torch.device | str = "cpu",
        initialization_seed: int = 0,
    ) -> None:
        self.config = config or default_maddpg_config()
        self.config.validate()
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            actors = [
                Actor(
                    self.config.observation_dim,
                    self.config.hidden_dims,
                    self.config.action_dim,
                )
                for _ in range(self.config.agent_count)
            ]
            critics = [
                JointCritic(
                    self.config.agent_count,
                    self.config.observation_dim,
                    self.config.action_dim,
                    self.config.hidden_dims,
                )
                for _ in range(self.config.agent_count)
            ]
        self.actors = nn.ModuleList(actors).to(self.device)
        self.critics = nn.ModuleList(critics).to(self.device)
        self.target_actors = deepcopy(self.actors).to(self.device)
        self.target_critics = deepcopy(self.critics).to(self.device)
        for module in (*self.target_actors, *self.target_critics):
            module.requires_grad_(False)
            module.eval()
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=self.config.actor_learning_rate)
            for actor in self.actors
        ]
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.config.critic_learning_rate)
            for critic in self.critics
        ]
        self.update_count = 0

    def select_actions(self, observations: ArrayLike) -> NDArray[np.float32]:
        values = np.asarray(observations, dtype=np.float32)
        expected = (self.config.agent_count, self.config.observation_dim)
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise ValueError(f"observations must be finite with shape {expected}")
        tensor = torch.as_tensor(values, device=self.device)
        with torch.no_grad():
            actions = torch.stack(
                [
                    actor(tensor[index])
                    for index, actor in enumerate(self.actors)
                ]
            )
        return actions.cpu().numpy().astype(np.float32, copy=False)

    def compute_critic_targets(self, batch: JointBatch) -> Tensor:
        with torch.no_grad():
            next_actions = torch.stack(
                [
                    actor(batch.next_observations[:, index, :])
                    for index, actor in enumerate(self.target_actors)
                ],
                dim=1,
            )
            mask = batch.bootstrap_mask
            targets = torch.stack(
                [
                    batch.rewards[:, index]
                    + self.config.gamma
                    * mask
                    * critic(batch.next_observations, next_actions)
                    for index, critic in enumerate(self.target_critics)
                ],
                dim=1,
            )
        _require_finite("critic_targets", targets)
        return targets

    def update(self, batch: JointBatch) -> MADDPGUpdateDiagnostics:
        targets = self.compute_critic_targets(batch)
        critic_records: list[dict[str, float]] = []
        for index, (critic, optimizer) in enumerate(
            zip(self.critics, self.critic_optimizers, strict=True)
        ):
            optimizer.zero_grad(set_to_none=True)
            q_values = critic(batch.observations, batch.actions)
            td_error = targets[:, index] - q_values
            loss = torch.mean(td_error.square())
            _require_finite(f"critic_{index}_loss", loss)
            loss.backward()
            gradient_norm = _gradient_norm(critic.parameters())
            _require_finite(f"critic_{index}_gradient_norm", gradient_norm)
            optimizer.step()
            critic_records.append(
                {
                    "critic_loss": float(loss.detach().item()),
                    "q_mean": float(q_values.detach().mean().item()),
                    "target_q_mean": float(targets[:, index].mean().item()),
                    "td_abs_mean": float(td_error.detach().abs().mean().item()),
                    "td_abs_max": float(td_error.detach().abs().max().item()),
                    "critic_gradient_norm": gradient_norm,
                }
            )
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            base_actions = torch.stack(
                [
                    actor(batch.observations[:, index, :])
                    for index, actor in enumerate(self.actors)
                ],
                dim=1,
            )
        actor_records: list[dict[str, float]] = []
        for index, (actor, critic, optimizer) in enumerate(
            zip(self.actors, self.critics, self.actor_optimizers, strict=True)
        ):
            optimizer.zero_grad(set_to_none=True)
            critic.requires_grad_(False)
            own_action = actor(batch.observations[:, index, :])
            joint_actions = torch.stack(
                [
                    own_action if other == index else base_actions[:, other, :]
                    for other in range(self.config.agent_count)
                ],
                dim=1,
            )
            loss = -critic(batch.observations, joint_actions).mean()
            _require_finite(f"actor_{index}_loss", loss)
            loss.backward()
            gradient_norm = _gradient_norm(actor.parameters())
            _require_finite(f"actor_{index}_gradient_norm", gradient_norm)
            optimizer.step()
            critic.requires_grad_(True)
            actor_records.append(
                {
                    "actor_loss": float(loss.detach().item()),
                    "actor_gradient_norm": gradient_norm,
                    "action_norm_mean": float(
                        torch.linalg.vector_norm(own_action.detach(), dim=-1)
                        .mean()
                        .item()
                    ),
                    "action_norm_max": float(
                        torch.linalg.vector_norm(own_action.detach(), dim=-1)
                        .max()
                        .item()
                    ),
                }
            )

        with torch.no_grad():
            policy_actions = torch.stack(
                [
                    actor(batch.observations[:, index, :])
                    for index, actor in enumerate(self.actors)
                ],
                dim=1,
            )
            shuffled_actions = torch.roll(batch.actions, shifts=1, dims=0)
            for index, critic in enumerate(self.critics):
                policy_q = float(
                    critic(batch.observations, policy_actions).mean().item()
                )
                shuffled_q = float(
                    critic(batch.observations, shuffled_actions).mean().item()
                )
                _require_finite(f"critic_{index}_policy_q", policy_q)
                _require_finite(f"critic_{index}_shuffled_q", shuffled_q)
                actor_records[index].update(
                    {
                        "policy_q_mean": policy_q,
                        "shuffled_action_q_mean": shuffled_q,
                        "policy_minus_replay_q": policy_q
                        - critic_records[index]["q_mean"],
                        "policy_minus_shuffled_q": policy_q - shuffled_q,
                    }
                )

        self.soft_update_targets()
        self.update_count += 1
        diagnostics = []
        for critic_record, actor_record in zip(
            critic_records, actor_records, strict=True
        ):
            diagnostics.append(AgentUpdateDiagnostics(**critic_record, **actor_record))
        return MADDPGUpdateDiagnostics(self.update_count, tuple(diagnostics))

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        tau = self.config.tau
        pairs = (
            (self.actors, self.target_actors),
            (self.critics, self.target_critics),
        )
        for online_modules, target_modules in pairs:
            for online, target in zip(online_modules, target_modules, strict=True):
                for online_parameter, target_parameter in zip(
                    online.parameters(), target.parameters(), strict=True
                ):
                    target_parameter.lerp_(online_parameter, tau)

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "actors": [deepcopy(module.state_dict()) for module in self.actors],
            "critics": [deepcopy(module.state_dict()) for module in self.critics],
            "target_actors": [
                deepcopy(module.state_dict()) for module in self.target_actors
            ],
            "target_critics": [
                deepcopy(module.state_dict()) for module in self.target_critics
            ],
            "actor_optimizers": [
                deepcopy(optimizer.state_dict()) for optimizer in self.actor_optimizers
            ],
            "critic_optimizers": [
                deepcopy(optimizer.state_dict()) for optimizer in self.critic_optimizers
            ],
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["config"] != asdict(self.config):
            raise ValueError("MADDPG checkpoint configuration mismatch")
        groups = (
            (self.actors, state["actors"]),
            (self.critics, state["critics"]),
            (self.target_actors, state["target_actors"]),
            (self.target_critics, state["target_critics"]),
        )
        for modules, values in groups:
            if len(modules) != len(values):
                raise ValueError("MADDPG checkpoint agent count mismatch")
            for module, value in zip(modules, values, strict=True):
                module.load_state_dict(value)
        optimizer_groups = (
            (self.actor_optimizers, state["actor_optimizers"]),
            (self.critic_optimizers, state["critic_optimizers"]),
        )
        for optimizers, values in optimizer_groups:
            if len(optimizers) != len(values):
                raise ValueError("MADDPG optimizer count mismatch")
            for optimizer, value in zip(optimizers, values, strict=True):
                optimizer.load_state_dict(value)
        self.update_count = int(state["update_count"])
