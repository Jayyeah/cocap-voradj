from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

from cocap_voradj.models.iqn import CoCapIQN
from tools import iqn_token_matched_20260919 as matched


def test_role_z_full_contract_diff_is_allowlisted(tmp_path: Path):
    report = matched.config_diff(
        matched.ROLE_CONFIG,
        tmp_path / "ROLE_vs_Z_resolved_config_diff.json",
    )
    assert report["status"] == "pass"
    assert report["non_token_diff_count"] == 0
    assert {row["path"] for row in report["differences"]} == matched.ALLOWED_TOKEN_DIFFS
    assert report["resolved_config"]["matched_contract_sha256"] == report["resolved_config"]["z_matched_contract_sha256"]
    assert report["trainer_config"]["role"] == report["trainer_config"]["z"]
    assert report["replay_config"]["role"] == report["replay_config"]["z"]
    assert report["evaluator_config"]["role"] == report["evaluator_config"]["z"]


def test_role_z_initial_state_dict_is_bit_identical():
    role = matched.resolved(matched.ROLE_CONFIG)
    z = matched.resolved(matched.Z_CONFIG)
    torch.manual_seed(int(role["seed"]))
    role_model = CoCapIQN(matched.model_config(role))
    torch.manual_seed(int(z["seed"]))
    z_model = CoCapIQN(matched.model_config(z))
    assert dataclasses.asdict(role_model.config) != dataclasses.asdict(z_model.config)
    assert list(role_model.state_dict()) == list(z_model.state_dict())
    for key, value in role_model.state_dict().items():
        assert value.shape == z_model.state_dict()[key].shape
        assert torch.equal(value, z_model.state_dict()[key])
    assert matched.state_hash(role_model) == matched.state_hash(z_model)


def test_role_z_physical_observation_contract_is_exact():
    role = matched.resolved(matched.ROLE_CONFIG)
    z = matched.resolved(matched.Z_CONFIG)
    role_env, role_obs = matched.make_env(role, "mixed", int(role["seed"]))
    z_env, z_obs = matched.make_env(z, "mixed", int(z["seed"]))
    assert matched.runtime_metadata(role_env) == matched.runtime_metadata(z_env)
    for index, (role_row, z_row) in enumerate(zip(role_obs, z_obs)):
        if role_row is None:
            assert z_row is None
            continue
        np.testing.assert_array_equal(role_row["self"][:-1], z_row["self"][:-1])
        np.testing.assert_array_equal(role_row["pursuers"][:, :6], z_row["pursuers"][:, :6])
        np.testing.assert_array_equal(role_row["evaders"], z_row["evaders"])
        np.testing.assert_array_equal(role_row["obstacles"], z_row["obstacles"])
        assert float(role_row["self"][-1]) in (0.0, 1.0)
        assert np.isclose(z_row["self"][-1], z_env.z_state[index])


def test_registered_science_variables_are_frozen():
    role = matched.resolved(matched.ROLE_CONFIG)
    z = matched.resolved(matched.Z_CONFIG)
    assert role["perception"]["friend_ordering_mode"] == "physical_only"
    assert z["perception"]["friend_ordering_mode"] == "physical_only"
    assert role["iqn"]["pursuing_late_fusion"] is False
    assert z["iqn"]["pursuing_late_fusion"] is False
    assert z["z_state"] == {
        "enabled": True,
        "lambda": 0.95,
        "eta": 0.85,
        "update_semantics": "synchronous_previous_decision_state",
    }
    assert matched.MILESTONES == tuple(range(25_000, 200_001, 25_000))
    assert role["total_timesteps"] == z["total_timesteps"] == 200_000
