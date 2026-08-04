from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cocap_voradj.envs.coverage_ce import normalized_control_cost
from cocap_voradj.training.trainer import (
    CoCapTrainer,
    deep_update,
    load_config,
    parse_scalar_step_schedule,
    scalar_step_schedule_value,
)


ROOT = Path(__file__).resolve().parents[1]
FINAL_CONFIG = (
    ROOT
    / "configs/experiments/cr_ms_support_approach_ce_curriculum_20260802"
    / "stage1_4p1e1obs_scratch2m.yaml"
)


def test_final_delayed_energy_config_has_exact_transition_boundary() -> None:
    config = load_config(str(FINAL_CONFIG))
    schedule = parse_scalar_step_schedule(
        config["reward"]["coverage_ce_speed_weight_schedule"],
        default_value=config["reward"]["coverage_ce_speed_weight"],
        field_name="coverage_ce_speed_weight",
    )

    assert scalar_step_schedule_value(
        schedule, global_step=199_999, default_value=-1.0
    ) == 0.0
    assert scalar_step_schedule_value(
        schedule, global_step=200_000, default_value=-1.0
    ) == 0.0005
    assert config["reward"]["coverage_ce_acceleration_weight"] == 0.0
    assert config["reward"]["coverage_ce_angular_velocity_weight"] == 0.0
    assert config["reward"]["coverage_ce_pbrs_reset_mode"] == "phase_and_all_terminal"
    assert config["iqn"]["checkpoint_freq"] == 100_000
    assert config["iqn"]["learning_rate_schedule"] == [
        {"step": 0, "learning_rate": 0.0001},
        {"step": 200_000, "learning_rate": 0.00003},
    ]


def test_static_speed_weight_remains_backward_compatible() -> None:
    schedule = parse_scalar_step_schedule(
        [], default_value=0.05, field_name="coverage_ce_speed_weight"
    )

    assert schedule == [(0, 0.05)]
    assert scalar_step_schedule_value(
        schedule, global_step=0, default_value=0.05
    ) == 0.05
    assert scalar_step_schedule_value(
        schedule, global_step=999_999, default_value=0.05
    ) == 0.05


def test_invalid_speed_weight_schedule_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate steps"):
        parse_scalar_step_schedule(
            [{"step": 10, "value": 0.0}, {"step": 10, "value": 0.1}],
            default_value=0.0,
            field_name="coverage_ce_speed_weight",
        )


def test_speed_weight_changes_the_normalized_control_cost() -> None:
    no_energy, _ = normalized_control_cost(
        speed=2.0,
        acceleration=0.0,
        angular_velocity=0.0,
        max_speed=2.0,
        max_acceleration=1.0,
        max_angular_velocity=1.0,
        speed_weight=0.0,
        acceleration_weight=0.0,
        angular_velocity_weight=0.0,
    )
    delayed_energy, terms = normalized_control_cost(
        speed=2.0,
        acceleration=0.0,
        angular_velocity=0.0,
        max_speed=2.0,
        max_acceleration=1.0,
        max_angular_velocity=1.0,
        speed_weight=0.0005,
        acceleration_weight=0.0,
        angular_velocity_weight=0.0,
    )

    assert no_energy == 0.0
    assert np.isclose(terms["speed"], 1.0)
    assert np.isclose(delayed_energy, 0.0005)


def test_trainer_applies_schedule_to_environment_and_learning_rate(
    tmp_path: Path,
) -> None:
    config = load_config(str(FINAL_CONFIG))
    config = deep_update(
        config,
        {
            "device": "cpu",
            "output_root": str(tmp_path),
            "run_name": "ce8_schedule_contract",
            "total_timesteps": 1,
            "reward": {"voradj_grid_size": 12},
            "iqn": {
                "hidden_dim": 32,
                "num_heads": 4,
                "num_layers": 1,
                "batch_size": 4,
                "replay_capacity": 32,
                "min_replay_size": 32,
            },
            "voradj": {
                "replay_batch_counts": {
                    "pursuing": 1,
                    "pre_capture_cover": 1,
                    "post_capture_real": 1,
                    "recovery_pure": 1,
                }
            },
        },
    )
    trainer = CoCapTrainer(config)
    try:
        env = trainer.envs["voradj"]
        assert trainer.current_coverage_ce_speed_weight == 0.0
        assert env.reward_cfg["coverage_ce_speed_weight"] == 0.0

        trainer.global_step = 199_999
        trainer._set_coverage_ce_control_weights()
        trainer._set_learning_rate()
        assert trainer.current_coverage_ce_speed_weight == 0.0
        assert trainer.current_learning_rate == 0.0001

        trainer.global_step = 200_000
        trainer._set_coverage_ce_control_weights()
        trainer._set_learning_rate()
        assert trainer.current_coverage_ce_speed_weight == 0.0005
        assert env.reward_cfg["coverage_ce_speed_weight"] == 0.0005
        assert trainer.current_learning_rate == 0.00003
    finally:
        trainer.episode_log.close()
        trainer.metric_log.close()
