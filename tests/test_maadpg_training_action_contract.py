from dataclasses import replace

import numpy as np

from maadpg_reproduction.config import default_environment_config
from maadpg_reproduction.exploration import OUExplorationConfig
from maadpg_reproduction.pfm import potential_field_actions
from maadpg_reproduction.trainer import MAADPGTrainer, TrainerConfig
from maadpg_reproduction.training_config import default_maddpg_config


def _trainer():
    learner = replace(
        default_maddpg_config(),
        hidden_dims=(16, 16),
        replay_capacity=32,
        batch_size=4,
        replay_warmup=4,
    )
    return MAADPGTrainer(
        TrainerConfig(
            variant="maadpg",
            training_seed=99,
            environment=replace(default_environment_config(), horizon_steps=20),
            maddpg=learner,
            exploration=OUExplorationConfig(decay_steps=20),
        )
    )


def test_gate_candidate_noise_contract():
    trainer = _trainer()
    deterministic_actor = trainer.learner.select_actions(
        trainer.runtime.current_observation
    )
    exploration_state = trainer.exploration.state_dict()
    noisy_actor = trainer._exploratory_actor_actions()
    trainer.exploration.load_state_dict(exploration_state)
    np.testing.assert_array_equal(noisy_actor, trainer._exploratory_actor_actions())
    assert not np.array_equal(noisy_actor, deterministic_actor)
    assert np.all(np.linalg.norm(noisy_actor, axis=1) <= 1.0 + 1e-15)

    state = trainer.env.state
    first = potential_field_actions(
        state.pursuer_positions,
        state.target_position,
        trainer.env.obstacle_pairs,
        trainer.config.potential_field,
    ).actions
    second = potential_field_actions(
        state.pursuer_positions,
        state.target_position,
        trainer.env.obstacle_pairs,
        trainer.config.potential_field,
    ).actions
    np.testing.assert_array_equal(first, second)
