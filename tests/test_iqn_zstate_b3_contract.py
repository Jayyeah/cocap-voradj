from __future__ import annotations

import copy
import math
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.trainer import deep_update, load_config, set_global_config


ROOT = Path(__file__).resolve().parents[1]
TEACHER = ROOT / "artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage1_4v1_step_2000000.pt"
B1 = ROOT / "configs/experiments/iqn_cleanup_emergent_b1_20260917/no_is_pursuing.yaml"
B3 = ROOT / "configs/experiments/iqn_zstate_b3_20260918/z_state.yaml"


def _config(scene: str = "voradj") -> dict:
    config = load_config(str(B3))
    return deep_update(config, config["tasks"][scene])


def _env(seed: int = 2026091801, scene: str = "voradj") -> VorAdjEnv:
    config = _config(scene)
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=seed)
    env.reset()
    return env


def test_b3_config_changes_observation_only_and_preregisters_lambda_eta():
    b1 = load_config(str(B1))
    b3 = load_config(str(B3))
    assert b3["z_state"] == {
        "enabled": True,
        "lambda": 0.95,
        "eta": 0.85,
        "update_semantics": "synchronous_previous_decision_state",
    }
    assert b3["perception"]["include_is_pursuing"] is False
    assert b3["perception"]["include_z_state"] is True
    assert b3["perception"]["self_feature_dim"] == 9
    assert b3["iqn"]["include_is_pursuing"] is False
    assert b3["iqn"]["include_z_state"] is True
    assert b3["iqn"]["pursuing_embed_dim"] == 0
    assert b3["iqn"]["pursuer_feature_dim"] == 7
    assert b3["reward"] == b1["reward"]
    assert b3["voradj"] == b1["voradj"]
    assert b3["env"] == b1["env"]


def test_final_teacher_loads_and_z_student_has_no_late_role_branch():
    teacher = CoCapIQN.load(str(TEACHER), device="cpu").eval()
    assert teacher.config.include_is_pursuing is True
    assert teacher.config.include_z_state is False
    student, report = CoCapIQN.make_z_state_student(teacher)
    assert student.config.include_is_pursuing is False
    assert student.config.include_z_state is True
    assert student.config.self_feature_dim == 9
    assert student.config.pursuer_feature_dim == 7
    assert not any(key.startswith("pursuing_embed.") for key in student.state_dict())
    assert report["late_fusion_removed"] is True
    assert report["z_special_branch"] is False
    assert {item["student_semantic"] for item in report["semantic_column_remaps"]} == {
        "self.z_i",
        "friend.z_j",
    }
    assert torch.equal(
        student.state_dict()["transformer.layers.0.self_attn.in_proj_weight"],
        teacher.state_dict()["transformer.layers.0.self_attn.in_proj_weight"],
    )


def test_initial_decision_has_current_direct_evidence_and_plain_token_dimensions():
    env = _env()
    observations = env.get_policy_observations(False, True)
    teacher_view = env.get_policy_observations(True, False)
    assert env._z_update_count == 1
    assert env.pursuers[0].dt * env.pursuers[0].N == pytest.approx(0.5)
    assert all((not direct) or value == pytest.approx(1.0) for direct, value in zip(env._z_last_direct, env.z_state))
    for index, observation in enumerate(observations):
        if observation is None:
            continue
        assert observation["self"].shape == (9,)
        assert observation["pursuers"].shape == (8, 7)
        assert observation["self"][-1] == pytest.approx(float(env.z_state[index]))
        assert teacher_view[index]["self"].shape == (9,)
        assert teacher_view[index]["pursuers"].shape == (8, 7)
    with pytest.raises(ValueError):
        env.get_policy_observations(True, True)


def test_z_update_is_synchronous_and_uses_previous_neighbor_state(monkeypatch):
    env = _env()
    env._reset_z_state()
    sources = {0}
    monkeypatch.setattr(
        env,
        "_has_enemy_neighbor",
        lambda _data, key: bool(key[0] == "pursuer" and int(key[1]) in sources),
    )
    adjacency = {
        ("pursuer", 0): {("pursuer", 1)},
        ("pursuer", 1): {("pursuer", 0), ("pursuer", 2)},
        ("pursuer", 2): {("pursuer", 1), ("pursuer", 3)},
        ("pursuer", 3): {("pursuer", 2)},
    }
    data = {"adjacency": adjacency}
    env._advance_z_state(data)
    assert env.z_state.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    sources.clear()
    env._advance_z_state(data)
    assert env.z_state.tolist() == pytest.approx([0.95, 0.85, 0.0, 0.0])
    env._advance_z_state(data)
    assert env.z_state[2] == pytest.approx(0.85 * 0.85)
    assert env._z_lineage_hops[2] == 2


def test_z_decay_half_life_and_exact_snapshot_restore(monkeypatch):
    env = _env()
    env._reset_z_state()
    env.z_state[:] = 1.0
    env._z_lineage_hops[:] = 0
    env._z_source_age_steps[:] = 0
    monkeypatch.setattr(env, "_has_enemy_neighbor", lambda _data, _key: False)
    empty = {"adjacency": {("pursuer", i): set() for i in range(len(env.pursuers))}}
    half_life_steps = math.ceil(math.log(0.5) / math.log(0.95))
    for _ in range(half_life_steps):
        env._advance_z_state(empty)
    assert half_life_steps == 14
    assert np.all(env.z_state < 0.5)
    assert np.all(env.z_state > 0.45)
    state = env.z_state_dict()
    restored = _env(seed=2026091802)
    restored.load_z_state_dict(state)
    assert restored.z_state_dict() == state
    assert copy.deepcopy(restored).z_state_dict() == state
    assert pickle.loads(pickle.dumps(restored)).z_state_dict() == state


def test_replay_transition_records_current_and_next_z():
    env = _env()
    before = env.z_state.copy()
    result = env.step([0] * len(env.pursuers), [None] * len(env.evaders))
    for index, info in enumerate(result.infos):
        metadata = info["replay_metadata"]
        assert metadata["z_state_enabled"] is True
        assert metadata["z"] == pytest.approx(float(before[index]))
        assert metadata["next_z"] == pytest.approx(float(env.z_state[index]))
        if result.observations[index] is not None:
            assert result.observations[index]["self"][-1] == pytest.approx(metadata["next_z"])
