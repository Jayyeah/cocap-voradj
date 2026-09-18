from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.trainer import set_global_config
from tools import iqn_z_token_scratch_20260918 as run


def test_role_z_configs_differ_only_in_registered_evidence_fields(tmp_path: Path):
    report = run.config_diff(run.ROLE_REFERENCE, tmp_path / "diff.json")
    assert report["status"] == "pass"
    assert {row["path"] for row in report["differences"]} == run.ALLOWED_TOKEN_DIFFS
    z = run.resolved(run.Z_CONFIG)
    role = run.resolved(run.ROLE_REFERENCE)
    for config in (z, role):
        assert config["seed"] == 2026080201
        assert config["total_timesteps"] == 200000
        assert config["iqn"]["checkpoint_freq"] == 25000
        assert config["perception"]["friend_ordering_mode"] == "physical_only"
        assert config["iqn"]["pursuing_late_fusion"] is False
        assert config["voradj"]["onboard_sensing"]["policy"] == run.NORMSENSE_POLICY
        assert "enemy_sensing_radius" not in config["voradj"]
        assert "obstacle_sensing_radius" not in config["voradj"]
        assert not (config.get("pretrained", {}) or {}).get("path")
    assert z["z_state"] == {
        "enabled": True,
        "lambda": 0.95,
        "eta": 0.85,
        "update_semantics": "synchronous_previous_decision_state",
    }
    assert role["z_state"]["enabled"] is False


def test_matched_scratch_networks_have_identical_random_initialization():
    z, role = run.resolved(run.Z_CONFIG), run.resolved(run.ROLE_REFERENCE)
    torch.manual_seed(z["seed"])
    z_model = CoCapIQN(run.model_config(z))
    torch.manual_seed(role["seed"])
    role_model = CoCapIQN(run.model_config(role))
    assert run.state_hash(z_model) == run.state_hash(role_model)
    assert z_model.pursuing_embed is None
    assert role_model.pursuing_embed is None
    assert not any("pursuing_embed" in key for key in z_model.state_dict())


def test_role_token_cannot_change_physical_friend_ordering():
    config = run.resolved(run.ROLE_REFERENCE)
    scene = run.scene_config(config, "mixed")
    set_global_config(scene)
    env = VorAdjEnv(copy.deepcopy(scene), seed=config["seed"])
    env.reset()
    roles = {index: bool(index % 2) for index in range(len(env.pursuers))}
    original = env._effective_is_pursuing
    env._effective_is_pursuing = lambda index, raw: roles[index]
    before = env.get_policy_observations(True, False)
    roles = {index: not value for index, value in roles.items()}
    after = env.get_policy_observations(True, False)
    env._effective_is_pursuing = original
    for left, right in zip(before, after):
        if left is None:
            assert right is None
            continue
        np.testing.assert_array_equal(left["pursuers"][:, :6], right["pursuers"][:, :6])


def test_z_architecture_and_budget_are_frozen_v1():
    config = run.resolved(run.Z_CONFIG)
    model = CoCapIQN(run.model_config(config))
    assert config["perception"]["include_is_pursuing"] is False
    assert config["perception"]["include_z_state"] is True
    assert config["iqn"]["include_is_pursuing"] is False
    assert config["iqn"]["include_z_state"] is True
    assert config["iqn"]["pursuing_late_fusion"] is False
    assert model.pursuing_embed is None
    assert run.MILESTONES == tuple(range(25_000, 200_001, 25_000))
    assert config["formal_evaluation"]["episodes_per_scene"] == 20
    assert config["formal_evaluation"]["seed_base"] == 2026092801
    todo = config["experiment_metadata"]["z_v2_todo"]
    assert todo == ["shorter_half_life_lower_lambda", "hard_floor_epsilon_z_after_z_tilde"]
    assert "epsilon" not in config["z_state"]


def test_supervisor_rechecks_actual_role_even_with_cached_sanity(monkeypatch, tmp_path: Path):
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    (preflight / "startup_sanity.json").write_text('{"status":"pass"}')
    checked = []

    def reject(role_path, output):
        checked.append((role_path, output))
        raise RuntimeError("actual ROLE drift")

    monkeypatch.setattr(run, "config_diff", reject)
    with pytest.raises(RuntimeError, match="actual ROLE drift"):
        run.supervisor(tmp_path, "cpu", run.ROLE_REFERENCE)
    assert checked == [(run.ROLE_REFERENCE, preflight / "config_diff.json")]
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "blocked_prelaunch"
    assert status["phase"] == "prelaunch_contract_gate"
