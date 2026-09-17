from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cocap_voradj.training.td3_stage1_contract import aw9_grid, discrete_task_config
from cocap_voradj.training.trainer import replay_transition_done
from tools.run_td3_stage1_discrete_baseline_20260917 import (
    BUDGET,
    EVAL_SEED_BASE,
    IQN_HYPERPARAMETERS,
    MILESTONES,
    SEED,
    check_env,
    iqn_config,
    make_env,
)


def test_iqn_terminated_only_bootstraps_truncation_not_true_terminal() -> None:
    assert replay_transition_done(
        True, {"terminated": True, "truncated": False}, "terminated_only"
    )
    assert not replay_transition_done(
        True, {"terminated": False, "truncated": True}, "terminated_only"
    )
    assert not replay_transition_done(
        False, {"terminated": False, "truncated": False}, "terminated_only"
    )
    with pytest.raises(RuntimeError, match="explicit terminated/truncated"):
        replay_transition_done(True, {}, "terminated_only")


def test_legacy_replay_done_modes_remain_unchanged() -> None:
    assert not replay_transition_done(
        True, {"state": "all targets captured"}, "terminal_state"
    )
    assert replay_transition_done(True, {"state": "collision"}, "terminal_state")
    assert replay_transition_done(True, {}, "environment_done")


def test_iqn_baseline_contract_uses_historical_algorithm_and_stage1_budget(tmp_path: Path) -> None:
    config = iqn_config("coverage", tmp_path / "D1", "cpu")
    assert config["seed"] == SEED
    assert config["total_timesteps"] == BUDGET
    assert MILESTONES == (25_000, 50_000, 75_000, 100_000)
    assert EVAL_SEED_BASE == 2026191501
    assert config["train_mode"] == "voradj"
    assert config["iqn"] == IQN_HYPERPARAMETERS
    assert config["iqn"]["replay_done_mode"] == "terminated_only"
    assert config["iqn"]["architecture"] == "voradj_single_head"
    assert config["iqn"]["batch_size"] == 128
    assert config["iqn"]["min_replay_size"] == 3000
    assert config["iqn"]["train_freq"] == 4
    assert config["iqn"]["target_update_freq"] == 10_000
    assert config["iqn"]["epsilon_decay_steps"] == 1_000_000


@pytest.mark.parametrize("task,evaders", [("coverage", 0), ("capture", 1)])
def test_discrete_baseline_env_is_exact_normsense_v2_contract(task: str, evaders: int) -> None:
    env, _ = make_env(task, SEED)
    facts = check_env(env, task)
    assert env.config == discrete_task_config(task)
    assert len(env.evaders) == evaders
    assert facts["normsense_v2"] is True
    assert facts["global_enemy_flag"] is False
    assert facts["capture_terminal"] is (task == "capture")
    np.testing.assert_allclose(env.pursuers[0].action_list, aw9_grid(), atol=1e-12)
