from dataclasses import replace

import pytest
import torch

from maadpg_reproduction.checkpoint import load_full_checkpoint, save_full_checkpoint
from maadpg_reproduction.config import default_environment_config
from maadpg_reproduction.exploration import OUExplorationConfig
from maadpg_reproduction.trainer import MAADPGTrainer, TrainerConfig
from maadpg_reproduction.training_config import default_maddpg_config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_checkpoint_rng_can_be_restored_with_cuda_map_location(tmp_path):
    maddpg = replace(
        default_maddpg_config(),
        hidden_dims=(8, 8),
        replay_capacity=16,
        batch_size=4,
        replay_warmup=4,
    )
    config = TrainerConfig(
        variant="maddpg",
        training_seed=7,
        environment=replace(default_environment_config(), horizon_steps=10),
        maddpg=maddpg,
        exploration=OUExplorationConfig(decay_steps=10),
    )
    trainer = MAADPGTrainer(config, device="cpu")
    for _ in range(4):
        trainer.step_once()
    path = tmp_path / "cpu-full.pt"
    save_full_checkpoint(trainer, path)
    restored, _ = load_full_checkpoint(path, device="cuda:0")
    step = restored.step_once()
    assert step.environment_step == 5
    assert all(
        parameter.is_cuda
        for module in restored.learner.actors
        for parameter in module.parameters()
    )
