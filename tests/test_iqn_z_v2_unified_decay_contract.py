from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from cocap_voradj.envs.density_sensing import enable_v2
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.trainer import load_config, set_global_config
from tools.iqn_token_matched_20260919 import model_config, scene_config, state_hash


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/experiments/iqn_z_unified_decay_curriculum_20260919"
Z05_STAGE1 = CONFIG_DIR / "z05_stage1_4p1e1obs_2m.yaml"
Z07_STAGE1 = CONFIG_DIR / "z07_stage1_4p1e1obs_2m.yaml"
Z07_STAGE3 = CONFIG_DIR / "z07_stage3_12p3e3obs_700k.yaml"


def resolved(path: Path) -> dict:
    config = load_config(str(path))
    assert config["normsense_v2"]["enabled"] is True
    return enable_v2(config)


def make_env(path: Path, scene: str = "mixed") -> VorAdjEnv:
    config = resolved(path)
    live = scene_config(config, scene)
    set_global_config(live)
    env = VorAdjEnv(copy.deepcopy(live), seed=int(config["seed"]))
    env.reset()
    return env


def set_single_source(env: VorAdjEnv) -> None:
    env.z_state[:] = 0.0
    env.z_state[0] = 1.0
    env._z_lineage_hops[:] = -1
    env._z_lineage_hops[0] = 0
    env._z_source_age_steps[:] = -1
    env._z_source_age_steps[0] = 0
    env._z_last_source = ["zero"] * len(env.pursuers)
    env._z_last_source[0] = "direct"


@pytest.mark.parametrize(
    ("config_path", "expected"),
    [
        (Z05_STAGE1, [0.5, 0.25, 0.125, 0.0]),
        (Z07_STAGE1, [0.7, 0.49, 0.343, 0.2401, 0.16807, 0.117649, 0.0]),
    ],
)
def test_unified_decay_temporal_release_sequence(config_path: Path, expected: list[float]) -> None:
    env = make_env(config_path)
    set_single_source(env)
    env._has_enemy_neighbor = lambda _data, _key: False
    empty = {"adjacency": {}}
    observed = []
    for _ in expected:
        env._advance_z_state(empty)
        observed.append(float(env.z_state[0]))
    assert observed == pytest.approx(expected)
    assert env._z_last_source[0] == "zero"
    assert env._z_lineage_hops[0] == -1
    assert env._z_source_age_steps[0] == -1


@pytest.mark.parametrize(
    ("config_path", "agent_count", "expected"),
    [
        (Z05_STAGE1, 4, [1.0, 0.5, 0.25, 0.125]),
        (Z07_STAGE3, 8, [1.0, 0.7, 0.49, 0.343, 0.2401, 0.16807, 0.117649, 0.0]),
    ],
)
def test_unified_decay_spatial_hop_sequence(
    config_path: Path,
    agent_count: int,
    expected: list[float],
) -> None:
    env = make_env(config_path)
    assert len(env.pursuers) >= agent_count
    adjacency = {("pursuer", index): set() for index in range(len(env.pursuers))}
    for index in range(agent_count - 1):
        adjacency[("pursuer", index)].add(("pursuer", index + 1))
        adjacency[("pursuer", index + 1)].add(("pursuer", index))
    env._reset_z_state()
    env._has_enemy_neighbor = lambda _data, key: key == ("pursuer", 0)
    for _ in range(agent_count):
        env._advance_z_state({"adjacency": adjacency})
    assert env.z_state[:agent_count].tolist() == pytest.approx(expected)
    if expected[-1] == 0.0:
        index = agent_count - 1
        assert env._z_last_source[index] == "zero"
        assert env._z_lineage_hops[index] == -1
        assert env._z_source_age_steps[index] == -1


@pytest.mark.parametrize(("config_path", "alpha"), [(Z05_STAGE1, 0.5), (Z07_STAGE1, 0.7)])
def test_full_env_step_updates_z_once_not_per_physics_substep(
    config_path: Path,
    alpha: float,
) -> None:
    env = make_env(config_path)
    set_single_source(env)
    env._has_enemy_neighbor = lambda _data, _key: False
    before_count = env._z_update_count
    env.step([0] * len(env.pursuers), [None] * len(env.evaders))
    assert env._z_update_count == before_count + 1
    assert float(env.z_state[0]) == pytest.approx(alpha)
    assert float(env.z_state[0]) != pytest.approx(alpha**10)


def test_z_v2_checkpoint_is_exact_and_refuses_old_or_drifted_contracts() -> None:
    env = make_env(Z05_STAGE1)
    set_single_source(env)
    state = env.z_state_dict()
    assert state["schema"] == "cocap-z-state-v2"
    assert state["lambda"] == pytest.approx(0.5)
    assert state["eta"] == pytest.approx(0.5)
    assert state["hard_zero_threshold"] == pytest.approx(0.10)
    restored = make_env(Z05_STAGE1)
    restored.load_z_state_dict(state)
    assert restored.z_state_dict() == state

    old = copy.deepcopy(state)
    old["schema"] = "cocap-z-state-v1"
    with pytest.raises(ValueError, match="unsupported z-state checkpoint schema"):
        restored.load_z_state_dict(old)

    wrong_alpha = copy.deepcopy(state)
    wrong_alpha["lambda"] = 0.7
    with pytest.raises(ValueError, match="lambda mismatch"):
        restored.load_z_state_dict(wrong_alpha)

    wrong_threshold = copy.deepcopy(state)
    wrong_threshold["hard_zero_threshold"] = 0.0
    with pytest.raises(ValueError, match="hard-zero threshold mismatch"):
        restored.load_z_state_dict(wrong_threshold)

    inconsistent_zero = copy.deepcopy(state)
    inconsistent_zero["values"][1] = 0.0
    inconsistent_zero["last_source"][1] = "neighbor"
    with pytest.raises(ValueError, match="zero bookkeeping is inconsistent"):
        restored.load_z_state_dict(inconsistent_zero)


def test_z05_z07_initial_networks_are_bit_exact() -> None:
    z05 = resolved(Z05_STAGE1)
    z07 = resolved(Z07_STAGE1)
    assert z05["seed"] == z07["seed"]
    torch.manual_seed(int(z05["seed"]))
    model05 = CoCapIQN(model_config(z05))
    torch.manual_seed(int(z07["seed"]))
    model07 = CoCapIQN(model_config(z07))
    assert list(model05.state_dict()) == list(model07.state_dict())
    for key, value in model05.state_dict().items():
        assert torch.equal(value, model07.state_dict()[key])
    assert state_hash(model05) == state_hash(model07)
