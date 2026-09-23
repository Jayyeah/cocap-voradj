"""Exact categorical SAC preflight for the existing AW9 action contract.

This module is deliberately local to the discrete-SAC preflight.  It reuses
the validated shared-local observation encoder and AW9 categorical heads, but
does not change the environment, reward, observation, or curriculum paths.
"""
from __future__ import annotations

import copy
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cocap_voradj.models.shared_local_ac import (
    SharedLocalACNetworkConfig,
    SharedLocalActor,
    SharedLocalQ,
    hard_copy,
)
from cocap_voradj.training.shared_local_ac import state_hash


SCHEMA = "discrete-sac-aw9-preflight-v1"


@dataclass(frozen=True)
class DiscreteSACConfig:
    """Optimizer and Bellman settings for the categorical SAC learner."""

    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    alpha_init: float = 0.2
    target_entropy: float | None = None
    adam_eps: float = 1e-8
    max_grad_norm: float | None = 0.5


def categorical_target_value(
    policy_logits: torch.Tensor,
    q1_target: torch.Tensor,
    q2_target: torch.Tensor,
    alpha: torch.Tensor | float,
) -> torch.Tensor:
    """Compute the exact discrete-SAC target value for every batch row.

    The expectation is over all nine categorical actions.  ``q1_target`` and
    ``q2_target`` are vector-valued ``Q(o', a)`` tensors, so the twin minimum
    is taken action-wise before the entropy term is applied.
    """

    if policy_logits.ndim != 2 or q1_target.shape != policy_logits.shape or q2_target.shape != policy_logits.shape:
        raise ValueError("policy logits and twin-Q targets must all have shape [batch, action]")
    log_pi = F.log_softmax(policy_logits, dim=-1)
    pi = log_pi.exp()
    q_min = torch.minimum(q1_target, q2_target)
    return (pi * (q_min - torch.as_tensor(alpha, device=pi.device, dtype=pi.dtype) * log_pi)).sum(dim=-1)


class DiscreteSACTrainer:
    """Shared-local categorical SAC with twin vector-valued critics.

    Each critic returns ``Q(o, a_0..a_8)``.  The policy is categorical over
    those same nine indices; no relaxed/Gumbel or continuous action path is
    involved.
    """

    schema_version = SCHEMA

    def __init__(
        self,
        network: SharedLocalACNetworkConfig,
        config: DiscreteSACConfig | None = None,
        device: str = "cpu",
    ) -> None:
        if int(network.action_size) != 9:
            raise ValueError("AW9 discrete SAC requires exactly nine actions")
        self.device = torch.device(device)
        self.network_config = network
        self.config = config or DiscreteSACConfig()
        if self.config.alpha_init <= 0.0:
            raise ValueError("alpha_init must be positive")
        if self.config.target_entropy is None:
            self.target_entropy = float(0.98 * np.log(int(network.action_size)))
        else:
            self.target_entropy = float(self.config.target_entropy)

        self.actor = SharedLocalActor(network).to(self.device)
        self.critic1 = SharedLocalQ(network).to(self.device)
        self.critic2 = SharedLocalQ(network).to(self.device)
        self.target_critic1 = hard_copy(self.critic1).to(self.device)
        self.target_critic2 = hard_copy(self.critic2).to(self.device)
        self.log_alpha = nn.Parameter(
            torch.log(torch.tensor(float(self.config.alpha_init), device=self.device))
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr, eps=self.config.adam_eps
        )
        self.critic_optimizer = torch.optim.Adam(
            [*self.critic1.parameters(), *self.critic2.parameters()],
            lr=self.config.critic_lr,
            eps=self.config.adam_eps,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.config.alpha_lr, eps=self.config.adam_eps
        )
        self.update_count = 0
        self._set_eval_modes()

    def _set_eval_modes(self) -> None:
        """Match the existing categorical AC path's deterministic mode."""

        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.target_critic1.eval()
        self.target_critic2.eval()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp_min(1e-8)

    @staticmethod
    @torch.no_grad()
    def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.mul_(1.0 - float(tau)).add_(source_parameter, alpha=float(tau))
        for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
            target_buffer.copy_(source_buffer)

    def _batch_tensors(self, batch: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        required = {"obs", "next_obs", "actions", "rewards"}
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"discrete SAC batch missing keys: {sorted(missing)}")
        terminal_key = "terminated" if "terminated" in batch else "dones" if "dones" in batch else None
        if terminal_key is None:
            raise ValueError("discrete SAC batch requires terminated or dones")
        obs = {key: value.to(self.device) for key, value in batch["obs"].items()}
        next_obs = {key: value.to(self.device) for key, value in batch["next_obs"].items()}
        actions = batch["actions"].to(self.device, dtype=torch.long).reshape(-1)
        rewards = batch["rewards"].to(self.device, dtype=torch.float32).reshape(-1)
        terminated = batch[terminal_key].to(self.device, dtype=torch.bool).reshape(-1)
        batch_size = int(rewards.shape[0])
        if actions.shape[0] != batch_size or terminated.shape[0] != batch_size:
            raise ValueError("discrete SAC batch tensors have inconsistent batch dimensions")
        if torch.any((actions < 0) | (actions >= int(self.network_config.action_size))):
            raise ValueError("discrete SAC action index is outside AW9")
        return obs, next_obs, actions, rewards, terminated

    @torch.no_grad()
    def target_value(self, next_obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        policy_logits = self.actor.logits(next_obs)
        q1_target = self.target_critic1(next_obs)
        q2_target = self.target_critic2(next_obs)
        return categorical_target_value(policy_logits, q1_target, q2_target, self.alpha.detach())

    def _clip(self, parameters: list[nn.Parameter]) -> float:
        if self.config.max_grad_norm is None:
            values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
            return float(torch.sqrt(torch.stack(values).sum()).detach()) if values else 0.0
        return float(torch.nn.utils.clip_grad_norm_(parameters, float(self.config.max_grad_norm)))

    def update(self, batch: Mapping[str, Any]) -> dict[str, float]:
        obs, next_obs, actions, rewards, terminated = self._batch_tensors(batch)
        with torch.no_grad():
            target_value = self.target_value(next_obs)
            td_target = rewards + float(self.config.gamma) * (~terminated).to(rewards.dtype) * target_value

        q1_all = self.critic1(obs)
        q2_all = self.critic2(obs)
        q1_taken = q1_all.gather(1, actions[:, None]).squeeze(1)
        q2_taken = q2_all.gather(1, actions[:, None]).squeeze(1)
        critic_loss = F.mse_loss(q1_taken, td_target) + F.mse_loss(q2_taken, td_target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = self._clip([*self.critic1.parameters(), *self.critic2.parameters()])
        self.critic_optimizer.step()

        for parameter in [*self.critic1.parameters(), *self.critic2.parameters()]:
            parameter.requires_grad_(False)
        logits = self.actor.logits(obs)
        log_pi = F.log_softmax(logits, dim=-1)
        pi = log_pi.exp()
        q_min = torch.minimum(self.critic1(obs), self.critic2(obs))
        actor_loss = (pi * (self.alpha.detach() * log_pi - q_min)).sum(dim=-1).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = self._clip(list(self.actor.parameters()))
        self.actor_optimizer.step()
        for parameter in [*self.critic1.parameters(), *self.critic2.parameters()]:
            parameter.requires_grad_(True)

        entropy = -(pi * log_pi).sum(dim=-1)
        alpha_loss = (self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_grad_norm = self._clip([self.log_alpha])
        self.alpha_optimizer.step()

        self.soft_update(self.critic1, self.target_critic1, self.config.tau)
        self.soft_update(self.critic2, self.target_critic2, self.config.tau)
        self.update_count += 1

        values = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
            "entropy": entropy.mean(),
            "td_target_mean": td_target.mean(),
            "q1_mean": q1_all.mean(),
            "q2_mean": q2_all.mean(),
        }
        finite = all(bool(torch.isfinite(value).all()) for value in values.values())
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "entropy": float(entropy.mean().detach()),
            "td_target_mean": float(td_target.mean().detach()),
            "q1_mean": float(q1_all.mean().detach()),
            "q2_mean": float(q2_all.mean().detach()),
            "critic_grad_norm": critic_grad_norm,
            "actor_grad_norm": actor_grad_norm,
            "alpha_grad_norm": alpha_grad_norm,
            "finite": float(finite),
            "update_count": float(self.update_count),
        }

    def checkpoint_payload(
        self,
        *,
        contract: Mapping[str, Any],
        global_step: int = 0,
        episode: int = 0,
        include_runtime: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema_version,
            "network_config": asdict(self.network_config),
            "trainer_config": asdict(self.config),
            "target_entropy": self.target_entropy,
            "contract": copy.deepcopy(dict(contract)),
            "global_step": int(global_step),
            "episode": int(episode),
            "update_count": int(self.update_count),
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_hash": state_hash(self.actor),
            "critic1_hash": state_hash(self.critic1),
            "critic2_hash": state_hash(self.critic2),
        }
        if include_runtime:
            payload.update(
                actor_optimizer=self.actor_optimizer.state_dict(),
                critic_optimizer=self.critic_optimizer.state_dict(),
                alpha_optimizer=self.alpha_optimizer.state_dict(),
                python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),
            )
            if torch.cuda.is_available():
                payload["cuda_rng"] = torch.cuda.get_rng_state_all()
        return payload

    def load_payload(
        self,
        payload: Mapping[str, Any],
        expected_contract: Mapping[str, Any],
        *,
        load_optimizer: bool = True,
    ) -> None:
        if payload.get("schema") != self.schema_version:
            raise ValueError("discrete SAC checkpoint schema mismatch")
        if dict(payload.get("contract", {})) != dict(expected_contract):
            raise ValueError("discrete SAC checkpoint contract mismatch")
        if dict(payload.get("network_config", {})) != asdict(self.network_config):
            raise ValueError("discrete SAC checkpoint network configuration mismatch")
        self.actor.load_state_dict(payload["actor"])
        self.critic1.load_state_dict(payload["critic1"])
        self.critic2.load_state_dict(payload["critic2"])
        self.target_critic1.load_state_dict(payload["target_critic1"])
        self.target_critic2.load_state_dict(payload["target_critic2"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
        if load_optimizer and "actor_optimizer" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        self.update_count = int(payload.get("update_count", 0))
        if payload.get("python_rng") is not None:
            random.setstate(payload["python_rng"])
        if payload.get("numpy_rng") is not None:
            np.random.set_state(payload["numpy_rng"])
        if payload.get("torch_rng") is not None:
            torch.set_rng_state(payload["torch_rng"].cpu())
        cuda_rng = payload.get("cuda_rng")
        if cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([item.cpu() for item in cuda_rng])
        self._set_eval_modes()


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    """Atomically write a preflight checkpoint in its destination directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
