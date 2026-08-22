from copy import deepcopy
from dataclasses import replace

import pytest

from maadpg_reproduction.config import default_environment_config
from maadpg_reproduction.exploration import OUExplorationConfig
from maadpg_reproduction.trainer import MAADPGTrainer, TrainerConfig
from maadpg_reproduction.training_config import default_maddpg_config


def _trainer():
    maddpg = replace(
        default_maddpg_config(),
        hidden_dims=(8, 8),
        replay_capacity=16,
        batch_size=4,
        replay_warmup=4,
    )
    config = TrainerConfig(
        variant="maddpg",
        training_seed=88,
        environment=replace(default_environment_config(), horizon_steps=10),
        maddpg=maddpg,
        exploration=OUExplorationConfig(decay_steps=10),
    )
    trainer = MAADPGTrainer(config)
    for _ in range(5):
        trainer.step_once()
    return trainer


def test_full_bundle_declares_replay_and_all_step_counters_agree():
    trainer = _trainer()
    state = trainer.state_dict()
    assert state["checkpoint_kind"] == "full_runtime"
    assert state["contains_replay"] is True
    assert state["runtime"].environment_steps == state["replay"]["total_insertions"]
    assert state["runtime"].gradient_steps == state["learner"]["update_count"]
    assert state["runtime"].current_episode_length == state["environment"].state.step_count


@pytest.mark.parametrize(
    "mutation",
    [
        "runtime_environment_step",
        "runtime_gradient_step",
        "runtime_episode_step",
        "replay_cursor",
        "runtime_observation",
    ],
)
def test_checkpoint_counter_or_observation_mismatch_fails_closed(mutation):
    trainer = _trainer()
    state = trainer.state_dict()
    corrupted = deepcopy(state)
    if mutation == "runtime_environment_step":
        corrupted["runtime"].environment_steps += 1
    elif mutation == "runtime_gradient_step":
        corrupted["runtime"].gradient_steps += 1
    elif mutation == "runtime_episode_step":
        corrupted["runtime"].current_episode_length += 1
    elif mutation == "replay_cursor":
        corrupted["replay"]["cursor"] += 1
    else:
        corrupted["runtime"].current_observation[0, 0] += 1.0
    restored = MAADPGTrainer(trainer.config)
    with pytest.raises((RuntimeError, ValueError), match="consistency|observation|cursor"):
        restored.load_state_dict(corrupted)
