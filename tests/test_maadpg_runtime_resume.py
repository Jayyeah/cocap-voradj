from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from maadpg_reproduction.checkpoint import (
    load_full_checkpoint,
    save_full_checkpoint,
    save_model_checkpoint,
)
from maadpg_reproduction.config import default_environment_config
from maadpg_reproduction.evaluation import evaluate_actor_only, wilson_interval
from maadpg_reproduction.exploration import (
    IndependentOUNoise,
    OUExplorationConfig,
)
from maadpg_reproduction.trainer import MAADPGTrainer, TrainerConfig
from maadpg_reproduction.training_config import default_maddpg_config


def _trainer_config(variant: str) -> TrainerConfig:
    environment = replace(default_environment_config(), horizon_steps=20)
    maddpg = replace(
        default_maddpg_config(),
        hidden_dims=(16, 16),
        replay_capacity=64,
        batch_size=4,
        replay_warmup=4,
    )
    exploration = OUExplorationConfig(decay_steps=20)
    return TrainerConfig(
        variant=variant,
        training_seed=2026082301,
        environment=environment,
        maddpg=maddpg,
        exploration=exploration,
    )


def _step_signature(step):
    update_values = tuple(
        (
            update.update_step,
            tuple(
                (
                    agent.critic_loss,
                    agent.actor_loss,
                    agent.q_mean,
                    agent.target_q_mean,
                    agent.td_abs_mean,
                    agent.td_abs_max,
                    agent.critic_gradient_norm,
                    agent.actor_gradient_norm,
                )
                for agent in update.agents
            ),
        )
        for update in step.updates
    )
    gate = None
    if step.gate_decision is not None:
        gate = (
            step.gate_decision.adopted_pfm.tobytes(),
            step.gate_decision.delta_ratios.tobytes(),
            step.gate_decision.adopted_actions.tobytes(),
        )
    episode = None
    if step.episode is not None:
        episode = (
            step.episode.episode_index,
            step.episode.scenario_seed,
            step.episode.length,
            step.episode.returns,
            step.episode.success,
        )
    return (
        step.environment_step,
        step.gradient_step,
        step.reward,
        step.executed_action.tobytes(),
        step.terminated,
        step.truncated,
        step.replay_size,
        update_values,
        gate,
        episode,
    )


def test_ou_exploration_schedule_and_rng_resume_are_exact():
    config = OUExplorationConfig(decay_steps=100)
    noise = IndependentOUNoise(config, seed=5)
    assert noise.sigma(0) == pytest.approx(0.20)
    assert noise.sigma(50) == pytest.approx(0.125)
    assert noise.sigma(1000) == pytest.approx(0.05)
    noise.sample(0)
    state = noise.state_dict()
    expected = noise.sample(1)
    restored = IndependentOUNoise(config, seed=999)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(restored.sample(1), expected)
    restored.reset_episode()
    np.testing.assert_array_equal(restored.state, np.zeros((3, 2)))


@pytest.mark.parametrize("variant", ["maddpg", "maadpg"])
def test_full_checkpoint_split_run_matches_uninterrupted_next_updates(tmp_path, variant):
    config = _trainer_config(variant)
    uninterrupted = MAADPGTrainer(config)
    uninterrupted_steps = [_step_signature(uninterrupted.step_once()) for _ in range(12)]

    split = MAADPGTrainer(config)
    split_steps = [_step_signature(split.step_once()) for _ in range(5)]
    assert split.runtime.current_episode_length > 0
    checkpoint = tmp_path / f"{variant}-rolling.pt"
    save_full_checkpoint(split, checkpoint, metadata={"run_id": f"test-{variant}"})
    resumed, metadata = load_full_checkpoint(checkpoint)
    assert metadata == {"run_id": f"test-{variant}"}
    split_steps.extend(_step_signature(resumed.step_once()) for _ in range(7))
    assert split_steps == uninterrupted_steps

    assert resumed.runtime.environment_steps == uninterrupted.runtime.environment_steps
    assert resumed.runtime.gradient_steps == uninterrupted.runtime.gradient_steps
    np.testing.assert_array_equal(
        resumed.runtime.current_observation,
        uninterrupted.runtime.current_observation,
    )
    np.testing.assert_array_equal(
        resumed.replay.actions[: resumed.replay.size],
        uninterrupted.replay.actions[: uninterrupted.replay.size],
    )
    for resumed_actor, uninterrupted_actor in zip(
        resumed.learner.actors, uninterrupted.learner.actors, strict=True
    ):
        for resumed_value, uninterrupted_value in zip(
            resumed_actor.parameters(), uninterrupted_actor.parameters(), strict=True
        ):
            torch.testing.assert_close(
                resumed_value, uninterrupted_value, rtol=0.0, atol=0.0
            )


def test_full_and_model_checkpoint_kinds_are_distinct(tmp_path):
    trainer = MAADPGTrainer(_trainer_config("maddpg"))
    trainer.step_once()
    full_path = tmp_path / "rolling.pt"
    model_path = tmp_path / "milestone.pt"
    save_full_checkpoint(trainer, full_path, metadata={"kind": "full"})
    save_model_checkpoint(trainer, model_path, metadata={"kind": "model"})
    full = torch.load(full_path, weights_only=False)
    model = torch.load(model_path, weights_only=False)
    assert "replay" in full and "environment" in full and "global_rng" in full
    assert model["checkpoint_kind"] == "model_only"
    assert "replay" not in model and "environment" not in model


def test_actor_only_evaluation_uses_seed_count_and_wilson_interval():
    config = _trainer_config("maadpg")
    trainer = MAADPGTrainer(config)
    environment = replace(config.environment, horizon_steps=2)
    summary = evaluate_actor_only(trainer.learner, environment, [101, 102, 103])
    assert len(summary.episodes) == 3
    assert [episode.scenario_seed for episode in summary.episodes] == [101, 102, 103]
    assert 0.0 <= summary.success_rate <= 1.0
    assert 0.0 <= summary.wilson_95_low <= summary.wilson_95_high <= 1.0
    assert all(episode.steps <= 2 for episode in summary.episodes)
    low, high = wilson_interval(0, 3)
    assert low == pytest.approx(0.0)
    assert high > 0.0
