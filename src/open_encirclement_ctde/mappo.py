"""Audited bridge to the immutable ``light_mappo`` MAPPO core.

The upstream actor/critic, PPO update, GAE buffer, Huber loss, and ValueNorm
are reused.  The bridge replaces only the unbounded Gaussian action head with
a tanh-squashed head whose samples and log probabilities use the physical
``[-0.04, 0.04]`` action contract.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gym
import numpy as np
import torch
import torch.nn as nn


LIGHT_MAPPO_COMMIT = "c503d89b6f28c9687ce9e45304fe66b57322ce1e"


@dataclass(frozen=True)
class MAPPOConfig:
    obs_dim: int = 26
    num_agents: int = 3
    action_dim: int = 2
    action_limit: float = 0.04
    hidden_size: int = 64
    layer_N: int = 1
    lr: float = 5e-4
    critic_lr: float = 5e-4
    ppo_epoch: int = 15
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 10.0
    huber_delta: float = 10.0
    value_loss_coef: float = 1.0
    num_mini_batch: int = 1
    rollout_length: int = 25
    n_rollout_threads: int = 8
    log_std_init: float = -0.5

    @property
    def joint_obs_dim(self) -> int:
        return self.obs_dim * self.num_agents


class _BoundedNormal:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, scale: float) -> None:
        self.base = torch.distributions.Normal(mean, std)
        self.scale = float(scale)

    def sample(self) -> torch.Tensor:
        return torch.tanh(self.base.sample()) * self.scale

    def mode(self) -> torch.Tensor:
        return torch.tanh(self.base.mean) * self.scale

    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp(actions / self.scale, -1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(normalized)
        correction = torch.log(self.scale * (1.0 - normalized.square()) + 1e-8)
        return (self.base.log_prob(pre_tanh) - correction).sum(dim=-1, keepdim=True)

    def entropy(self) -> torch.Tensor:
        # Deterministic base entropy keeps the upstream entropy interface and
        # avoids adding an extra Monte-Carlo RNG draw during PPO updates.
        return self.base.entropy()


class BoundedDiagGaussian(nn.Module):
    def __init__(self, inputs_dim: int, action_dim: int, action_limit: float, gain: float, log_std_init: float) -> None:
        super().__init__()
        self.mean = nn.Linear(inputs_dim, action_dim)
        nn.init.orthogonal_(self.mean.weight, gain=gain)
        nn.init.constant_(self.mean.bias, 0.0)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.action_limit = float(action_limit)

    def forward(self, features: torch.Tensor) -> _BoundedNormal:
        mean = self.mean(features)
        std = self.log_std.exp().expand_as(mean)
        return _BoundedNormal(mean, std, self.action_limit)


class BoundedACTLayer(nn.Module):
    continuous_action = True
    mixed_action = False
    multi_discrete = False

    def __init__(self, hidden_size: int, action_dim: int, action_limit: float, gain: float, log_std_init: float) -> None:
        super().__init__()
        self.action_out = BoundedDiagGaussian(hidden_size, action_dim, action_limit, gain, log_std_init)

    def forward(self, features, available_actions=None, deterministic=False):
        distribution = self.action_out(features)
        actions = distribution.mode() if deterministic else distribution.sample()
        return actions, distribution.log_probs(actions)

    def evaluate_actions(self, features, action, available_actions=None, active_masks=None):
        distribution = self.action_out(features)
        log_probs = distribution.log_probs(action)
        entropy = distribution.entropy()
        if active_masks is not None:
            entropy = (entropy * active_masks).sum() / active_masks.sum()
        else:
            entropy = entropy.mean()
        return log_probs, entropy


def _args(config: MAPPOConfig) -> SimpleNamespace:
    return SimpleNamespace(
        lr=config.lr,
        critic_lr=config.critic_lr,
        opti_eps=1e-5,
        weight_decay=0.0,
        hidden_size=config.hidden_size,
        gain=0.01,
        use_orthogonal=True,
        use_policy_active_masks=True,
        use_naive_recurrent_policy=False,
        use_recurrent_policy=False,
        recurrent_N=1,
        use_feature_normalization=True,
        use_ReLU=True,
        stacked_frames=1,
        layer_N=config.layer_N,
        use_popart=False,
        clip_param=config.clip_param,
        ppo_epoch=config.ppo_epoch,
        num_mini_batch=config.num_mini_batch,
        data_chunk_length=10,
        value_loss_coef=config.value_loss_coef,
        entropy_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm,
        huber_delta=config.huber_delta,
        use_max_grad_norm=True,
        use_clipped_value_loss=True,
        use_huber_loss=True,
        use_valuenorm=True,
        use_value_active_masks=True,
        episode_length=config.rollout_length,
        n_rollout_threads=config.n_rollout_threads,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        use_gae=True,
        use_proper_time_limits=False,
    )


def _load_upstream(repo_root: Path):
    vendor = repo_root / "vendor" / "upstream" / "light_mappo" / LIGHT_MAPPO_COMMIT
    if not vendor.is_dir():
        raise FileNotFoundError(vendor)
    vendor_text = str(vendor)
    # Keep the exported upstream snapshot byte-for-byte clean at runtime.
    sys.dont_write_bytecode = True
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)
    from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy
    from algorithms.algorithm.r_mappo import RMAPPO
    from utils.shared_buffer import SharedReplayBuffer

    return RMAPPOPolicy, RMAPPO, SharedReplayBuffer


class MAPPOLearner:
    def __init__(self, config: MAPPOConfig, device: torch.device, repo_root: Path) -> None:
        self.config = config
        self.device = device
        self.args = _args(config)
        policy_cls, trainer_cls, buffer_cls = _load_upstream(Path(repo_root))
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(config.obs_dim,), dtype=np.float32)
        joint_space = gym.spaces.Box(-np.inf, np.inf, shape=(config.joint_obs_dim,), dtype=np.float32)
        action_space = gym.spaces.Box(-config.action_limit, config.action_limit, shape=(config.action_dim,), dtype=np.float32)
        self.policy = policy_cls(self.args, observation_space, joint_space, action_space, device=device)
        self.policy.actor.act = BoundedACTLayer(
            config.hidden_size,
            config.action_dim,
            config.action_limit,
            self.args.gain,
            config.log_std_init,
        ).to(device)
        # The upstream optimizer was created before replacing its unbounded
        # action head. Recreate it so the bounded head is the optimized module.
        self.policy.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=config.lr, eps=self.args.opti_eps, weight_decay=self.args.weight_decay
        )
        self.trainer = trainer_cls(self.args, self.policy, device=device)
        self.buffer = buffer_cls(self.args, config.num_agents, observation_space, joint_space, action_space)
        self.update_count = 0
        self.initialized = False

    @staticmethod
    def _joint(observations: np.ndarray) -> np.ndarray:
        joint = observations.reshape(observations.shape[0], -1)
        return np.repeat(joint[:, None, :], observations.shape[1], axis=1)

    def initialize(self, observations: np.ndarray) -> None:
        observations = np.asarray(observations, dtype=np.float32)
        expected = (self.config.n_rollout_threads, self.config.num_agents, self.config.obs_dim)
        if observations.shape != expected:
            raise ValueError(f"MAPPO initial observations {observations.shape}, expected {expected}")
        self.buffer.obs[0] = observations
        self.buffer.share_obs[0] = self._joint(observations)
        self.initialized = True

    @torch.no_grad()
    def collect(self, step: int, *, deterministic: bool = False) -> dict[str, np.ndarray]:
        if not self.initialized:
            raise RuntimeError("MAPPO buffer not initialized")
        self.trainer.prep_rollout()
        values, actions, log_probs, rnn, rnn_critic = self.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            deterministic=deterministic,
        )
        split = self.config.n_rollout_threads
        to_numpy = lambda tensor: np.array(np.split(tensor.detach().cpu().numpy(), split))
        action_array = to_numpy(actions)
        if np.any(action_array < -self.config.action_limit - 1e-7) or np.any(action_array > self.config.action_limit + 1e-7):
            raise RuntimeError("bounded MAPPO actor violated physical action space")
        return {
            "values": to_numpy(values),
            "actions": action_array,
            "log_probs": to_numpy(log_probs),
            "rnn_states": to_numpy(rnn),
            "rnn_states_critic": to_numpy(rnn_critic),
        }

    @torch.no_grad()
    def terminal_bootstrap(self, terminal_observations: np.ndarray) -> np.ndarray:
        terminal_observations = np.asarray(terminal_observations, dtype=np.float32)
        joint = self._joint(terminal_observations)
        batch = terminal_observations.shape[0] * self.config.num_agents
        rnn = np.zeros((batch, 1, self.config.hidden_size), dtype=np.float32)
        masks = np.ones((batch, 1), dtype=np.float32)
        values = self.policy.get_values(np.concatenate(joint), rnn, masks)
        if self.trainer.value_normalizer is not None:
            values = self.trainer.value_normalizer.denormalize(values)
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        return np.asarray(values, dtype=np.float32).reshape(terminal_observations.shape[0], self.config.num_agents, 1)

    def insert(
        self,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        ended: np.ndarray,
        collected: dict[str, np.ndarray],
    ) -> None:
        ended = np.asarray(ended, dtype=bool)
        rnn = collected["rnn_states"].copy()
        rnn_critic = collected["rnn_states_critic"].copy()
        rnn[ended] = 0.0
        rnn_critic[ended] = 0.0
        masks = np.ones((self.config.n_rollout_threads, self.config.num_agents, 1), dtype=np.float32)
        masks[ended] = 0.0
        self.buffer.insert(
            self._joint(next_observations),
            np.asarray(next_observations, dtype=np.float32),
            rnn,
            rnn_critic,
            collected["actions"],
            collected["log_probs"],
            collected["values"],
            np.asarray(rewards, dtype=np.float32),
            masks,
        )

    @torch.no_grad()
    def policy_actions(self, observations: np.ndarray, *, deterministic: bool) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        batch = observations.shape[0] * self.config.num_agents
        rnn = np.zeros((batch, 1, self.config.hidden_size), dtype=np.float32)
        masks = np.ones((batch, 1), dtype=np.float32)
        actions, _ = self.policy.act(np.concatenate(observations), rnn, masks, deterministic=deterministic)
        return actions.detach().cpu().numpy().reshape(observations.shape[0], self.config.num_agents, self.config.action_dim)

    def deterministic_actions(self, observations: np.ndarray) -> np.ndarray:
        return self.policy_actions(observations, deterministic=True)

    def finish_rollout_and_train(self) -> dict[str, float]:
        self.trainer.prep_rollout()
        with torch.no_grad():
            next_values = self.policy.get_values(
                np.concatenate(self.buffer.share_obs[-1]),
                np.concatenate(self.buffer.rnn_states_critic[-1]),
                np.concatenate(self.buffer.masks[-1]),
            )
        next_values = np.array(np.split(next_values.detach().cpu().numpy(), self.config.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
        self.trainer.prep_training()
        raw = self.trainer.train(self.buffer)
        self.buffer.after_update()
        self.update_count += 1
        metrics = {key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value) for key, value in raw.items()}
        metrics["actor_lr"] = float(self.policy.actor_optimizer.param_groups[0]["lr"])
        metrics["critic_lr"] = float(self.policy.critic_optimizer.param_groups[0]["lr"])
        if not np.isfinite(np.asarray(list(metrics.values()), dtype=np.float64)).all():
            raise FloatingPointError(f"non-finite MAPPO metrics: {metrics}")
        return metrics

    def model_state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "actor": self.policy.actor.state_dict(),
            "critic": self.policy.critic.state_dict(),
            "update_count": self.update_count,
            "upstream_light_mappo_commit": LIGHT_MAPPO_COMMIT,
            "bounded_action_head": True,
        }

    def full_state_dict(self) -> dict[str, Any]:
        payload = {
            **self.model_state_dict(),
            "actor_optimizer": self.policy.actor_optimizer.state_dict(),
            "critic_optimizer": self.policy.critic_optimizer.state_dict(),
            "rollout_buffer_position": int(self.buffer.step),
        }
        if self.trainer.value_normalizer is not None:
            payload["value_normalizer"] = self.trainer.value_normalizer.state_dict()
        return payload

    def load_state_dict(self, state: dict[str, Any], *, full: bool) -> None:
        if state["config"] != asdict(self.config):
            raise ValueError("MAPPO config mismatch")
        if state.get("upstream_light_mappo_commit") != LIGHT_MAPPO_COMMIT:
            raise ValueError("light_mappo upstream commit mismatch")
        self.policy.actor.load_state_dict(state["actor"])
        self.policy.critic.load_state_dict(state["critic"])
        self.update_count = int(state["update_count"])
        if full:
            if state.get("rollout_buffer_position") != 0:
                raise ValueError("MAPPO full resume is only supported at rollout boundaries")
            self.policy.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.policy.critic_optimizer.load_state_dict(state["critic_optimizer"])
            if self.trainer.value_normalizer is not None:
                self.trainer.value_normalizer.load_state_dict(state["value_normalizer"])
