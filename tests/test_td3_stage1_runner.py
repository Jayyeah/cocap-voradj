from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cocap_voradj.training.continuous.joint_replay import ROLE_COVERAGE, ROLE_INACTIVE
from cocap_voradj.training.td3_stage1_contract import make_td3_env
from cocap_voradj.training.trainer import load_config
from tools.train_td3_local_aw_stage1_20260916 import (
    MAX_AGENTS,
    derive_roles,
    pad_local_observations,
    split_termination_flags,
    validate_config,
)


def _config():
    return load_config("configs/experiments/td3_local_aw_stage1_20260916/coverage_scratch.yaml")


def test_formal_config_locks_local_critic_and_stage_boundaries() -> None:
    config = _config()
    validate_config(config)
    assert config["critic"]["formula"] == "Q_i(o_i,a_i)"
    assert config["critic"]["global_state"] is False
    assert config["critic"]["teammate_actions"] is False
    assert all(config["forbidden"].values())


def test_pad_local_observations_marks_inactive_rows_fully_masked() -> None:
    env, observations = make_td3_env("coverage", 11)
    observations = list(observations)
    observations[2] = None
    padded = pad_local_observations(observations, _config()["network"])
    assert padded["self"].shape == (MAX_AGENTS, 9)
    assert not padded["masks"][2].any()
    assert np.allclose(padded["self"][2], 0.0)


def test_truncation_bootstraps_but_true_terminal_does_not() -> None:
    terminated, truncated = split_termination_flags(
        [True, True, False, False],
        [
            {"terminated": True, "truncated": False},
            {"terminated": False, "truncated": True},
            {"terminated": False, "truncated": False},
            {},
        ],
    )
    assert terminated.tolist() == [True, False, False, False]
    assert truncated.tolist() == [False, True, False, False]


def test_coverage_roles_use_only_active_rows() -> None:
    active = np.asarray([True, False, True, False])
    roles = derive_roles("coverage", [{}, {}, {}, {}], active)
    assert roles.tolist() == [ROLE_COVERAGE, ROLE_INACTIVE, ROLE_COVERAGE, ROLE_INACTIVE]
