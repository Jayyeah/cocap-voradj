"""Frozen standard-MADDPG schedule for the paper-v1 assumption set."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MADDPGConfig:
    agent_count: int = 3
    observation_dim: int = 26
    action_dim: int = 2
    hidden_dims: tuple[int, int] = (128, 128)
    gamma: float = 0.95
    tau: float = 0.01
    actor_learning_rate: float = 5.0e-4
    critic_learning_rate: float = 1.0e-3
    replay_capacity: int = 1_000_000
    batch_size: int = 1_024
    replay_warmup: int = 10_000
    update_every_environment_steps: int = 1
    gradient_steps_per_update: int = 1
    gradient_clipping: float | None = None

    @property
    def joint_critic_input_dim(self) -> int:
        return self.agent_count * (self.observation_dim + self.action_dim)

    def validate(self) -> None:
        if self.agent_count != 3 or self.observation_dim != 26 or self.action_dim != 2:
            raise ValueError("paper-v1 requires agents=3, observations=26, actions=2")
        if self.joint_critic_input_dim != 84:
            raise ValueError("joint critic input must be 84")
        if len(self.hidden_dims) != 2 or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("actor and critic require two positive hidden widths")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("gamma/tau outside valid ranges")
        if self.actor_learning_rate <= 0.0 or self.critic_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.replay_capacity <= 0 or self.batch_size <= 0:
            raise ValueError("replay capacity and batch size must be positive")
        if self.replay_warmup < self.batch_size:
            raise ValueError("replay warmup must cover at least one batch")
        if self.gradient_clipping is not None:
            raise ValueError("paper-v1 freezes gradient clipping as disabled")


def default_maddpg_config() -> MADDPGConfig:
    config = MADDPGConfig()
    config.validate()
    return config
