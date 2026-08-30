"""Small, deterministic contracts for the centralized TD3 update.

These tests intentionally use tiny critics instead of the attention critic.  The
point is to expose update semantics (masks, target smoothing, actor gradient
scope, and delayed target updates) without starting an experiment-sized run.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from cocap_voradj.training.continuous.local_sac import sac_bootstrap_mask
from cocap_voradj.training.small_step_ac import (
    TD3Config,
    TD3Trainer,
    flatten_local,
    tensor_tree,
)


class TinyActor(nn.Module):
    """Shared bounded actor with the same scale convention as DeterministicAWActor."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 2, bias=False)
        self.register_buffer("scale", torch.tensor([0.4, 0.5]))

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.tanh(self.projection(obs["x"])) * self.scale


class TinyActionCritic(nn.Module):
    """Critic whose cross-agent action dependence is explicit and inspectable."""

    def __init__(self, teammate_coefficient: float = 0.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.25))
        self.teammate_coefficient = float(teammate_coefficient)
        self.last_action: torch.Tensor | None = None

    def forward(self, _central: dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        # Keep a copy for target-action inspection, while retaining the graph
        # for the actor-gradient contract.
        if actions.requires_grad:
            actions.retain_grad()
            self.last_action = actions
        else:
            self.last_action = actions.detach().clone()
        first = actions[..., 0]
        cross_agent = first.sum(dim=1, keepdim=True)
        return first + self.teammate_coefficient * cross_agent + self.bias


class TinyTwinCritics(nn.Module):
    def __init__(self, teammate_coefficient: float = 0.0) -> None:
        super().__init__()
        self.critic1 = TinyActionCritic(teammate_coefficient)
        self.critic2 = TinyActionCritic(teammate_coefficient)

    def forward(
        self, central: dict[str, torch.Tensor], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic1(central, actions), self.critic2(central, actions)


def _trainer(*, policy_delay: int = 2, teammate_coefficient: float = 0.0) -> TD3Trainer:
    torch.manual_seed(20260830)
    return TD3Trainer(
        TinyActor(),
        TinyTwinCritics(teammate_coefficient),
        TD3Config(
            gamma=0.9,
            tau=0.2,
            actor_lr=1e-3,
            critic_lr=1e-3,
            policy_noise=0.1,
            noise_clip=0.2,
            policy_delay=policy_delay,
            max_grad_norm=100.0,
        ),
        device="cpu",
    )


def _batch(
    *,
    batch_size: int = 2,
    agents: int = 2,
    rewards: torch.Tensor | None = None,
    terminated: torch.Tensor | None = None,
    truncated: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
) -> dict[str, object]:
    # Nonzero features keep the tiny actor's parameter gradient observable;
    # the production encoder likewise receives nonconstant local features.
    x = torch.full((batch_size, agents, 1), 0.2)
    if rewards is None:
        rewards = torch.zeros(batch_size, agents)
    if terminated is None:
        terminated = torch.zeros(batch_size, agents, dtype=torch.bool)
    if truncated is None:
        truncated = torch.zeros(batch_size, agents, dtype=torch.bool)
    if active is None:
        active = torch.ones(batch_size, agents, dtype=torch.bool)
    return {
        "local_obs": {"x": x},
        "next_local_obs": {"x": x + 0.1},
        "global_state": {"dummy": torch.zeros(batch_size, agents, 1)},
        "next_global_state": {"dummy": torch.ones(batch_size, agents, 1)},
        "actions": torch.zeros(batch_size, agents, 2),
        "rewards": rewards,
        "active_mask": active,
        "terminated": terminated,
        "truncated": truncated,
    }


def _update(trainer: TD3Trainer, batch: dict[str, object]) -> dict[str, float]:
    # Target policy noise is stochastic; reset it so paired semantic checks
    # compare the update itself rather than different smoothing samples.
    torch.manual_seed(20260831)
    return trainer.update(batch)


def test_td3_bootstrap_contract_matches_terminal_truncation_semantics() -> None:
    terminated = torch.tensor([[False, True, False, True]])
    truncated = torch.tensor([[True, False, False, True]])
    expected = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    # This is the shared project contract: a time-limit is not an MDP terminal.
    torch.testing.assert_close(sac_bootstrap_mask(terminated, truncated), expected)

    timeout_batch = _batch(
        batch_size=1,
        agents=2,
        terminated=torch.zeros(1, 2, dtype=torch.bool),
        truncated=torch.tensor([[False, True]]),
    )
    equivalent_batch = copy.deepcopy(timeout_batch)
    equivalent_batch["truncated"] = torch.zeros(1, 2, dtype=torch.bool)
    first = _trainer()
    second = _trainer()
    first_metrics = _update(first, timeout_batch)
    second_metrics = _update(second, equivalent_batch)
    # TD3 uses (~terminated), which is exactly the intended bootstrap mask:
    # toggling only a time-limit flag must not change its target.
    for key in ("critic_loss", "q1_mean", "q2_mean", "twin_q_gap"):
        assert first_metrics[key] == pytest.approx(second_metrics[key], abs=1e-7)

    terminal_batch = copy.deepcopy(timeout_batch)
    terminal_batch["terminated"] = torch.tensor([[False, True]])
    terminal_metrics = _update(_trainer(), terminal_batch)
    assert terminal_metrics["critic_loss"] != pytest.approx(first_metrics["critic_loss"], abs=1e-6)


def test_td3_dead_agent_reward_is_excluded_from_critic_metrics_and_update() -> None:
    active = torch.tensor([[True, False]])
    base = _batch(
        batch_size=1,
        agents=2,
        rewards=torch.zeros(1, 2),
        active=active,
    )
    dead_reward = copy.deepcopy(base)
    dead_reward["rewards"] = torch.tensor([[0.0, 1_000_000.0]])
    dead_reward["terminated"] = torch.tensor([[False, True]])
    dead_reward["truncated"] = torch.tensor([[False, False]])
    first = _trainer()
    second = _trainer()
    first_metrics = _update(first, base)
    second_metrics = _update(second, dead_reward)
    for key in ("critic_loss", "q1_mean", "q2_mean", "twin_q_gap", "critic_grad_norm"):
        assert first_metrics[key] == pytest.approx(second_metrics[key], abs=1e-7)


def test_td3_shared_actor_currently_uses_joint_action_gradient() -> None:
    agents = 3
    batch = _batch(batch_size=2, agents=agents)
    trainer = _trainer(policy_delay=1, teammate_coefficient=5.0)
    metrics = _update(trainer, batch)
    assert metrics["update_count"] == 1.0
    assert metrics["actor_grad_norm"] > 0.0

    action_gradient = trainer.critics.critic1.last_action
    assert action_gradient is not None and action_gradient.grad is not None
    # q_i = a_i + 5 * sum_j(a_j), so a joint policy loss differentiates every
    # action branch through every focal q_i.  A focal/stop-gradient loss would
    # have coefficient 1 + 5 rather than 1 + 5 * agents here.
    expected = -(1.0 + 5.0 * agents) / (2.0 * agents)
    torch.testing.assert_close(
        action_gradient.grad[..., 0],
        torch.full((2, agents), expected),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        action_gradient.grad[..., 1],
        torch.zeros(2, agents),
        atol=1e-7,
        rtol=0.0,
    )


def test_td3_target_noise_is_normalized_then_scaled_and_clipped(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _trainer()
    batch = _batch(batch_size=1, agents=2)
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.full_like(value, 100.0))
    _update(trainer, batch)

    next_local = tensor_tree(batch["next_local_obs"], torch.device("cpu"))
    flat_next, shape = flatten_local(next_local)
    with torch.no_grad():
        raw = trainer.target_actor(flat_next).reshape(*shape, 2)
        scale = trainer.target_actor.scale.view(1, 1, 2)
        expected = (raw + 0.2 * scale).clamp(-scale, scale)
    recorded = trainer.target_critics.critic1.last_action
    assert recorded is not None
    torch.testing.assert_close(recorded, expected, atol=1e-7, rtol=0.0)


def test_td3_actor_and_targets_update_only_on_delayed_policy_step() -> None:
    trainer = _trainer(policy_delay=2)
    batch = _batch()
    initial_target_actor = {key: value.detach().clone() for key, value in trainer.target_actor.state_dict().items()}
    initial_target_critics = {key: value.detach().clone() for key, value in trainer.target_critics.state_dict().items()}

    first = _update(trainer, batch)
    assert first["update_count"] == 1.0
    assert first["actor_loss"] == 0.0
    for key, value in trainer.target_actor.state_dict().items():
        torch.testing.assert_close(value, initial_target_actor[key])
    for key, value in trainer.target_critics.state_dict().items():
        torch.testing.assert_close(value, initial_target_critics[key])

    second = _update(trainer, batch)
    assert second["update_count"] == 2.0
    assert second["actor_loss"] != 0.0
    assert any(
        not torch.equal(value, initial_target_actor[key])
        for key, value in trainer.target_actor.state_dict().items()
    )
    assert any(
        not torch.equal(value, initial_target_critics[key])
        for key, value in trainer.target_critics.state_dict().items()
    )

    third = _update(trainer, batch)
    assert third["actor_updated_this_step"] == 0.0
    assert third["actor_loss"] == 0.0
    assert third["last_actor_update_count"] == 2.0
    assert third["last_actor_loss"] == pytest.approx(second["actor_loss"])
    assert third["last_actor_grad_norm"] == pytest.approx(second["actor_grad_norm"])

    restored = _trainer(policy_delay=2)
    restored.load_state_dict(trainer.state_dict())
    assert restored.last_actor_update == trainer.last_actor_update
