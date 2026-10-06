from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.trainer import deep_update, load_config, set_global_config
from tools import iqn_token_matched_20260919 as matched
from tools.evaluate_iqn_evidence_checkpoint_20261006 import EvidenceDiagnostics


ROOT = Path(__file__).resolve().parents[1]
Z05 = ROOT / "configs/experiments/iqn_z_unified_decay_curriculum_20260919/z05_stage1_4p1e1obs_2m.yaml"
VARIANTS = {
    "z_state": Z05,
    "local_binary": ROOT / "configs/experiments/iqn_evidence_comparison_20261006/local_binary_stage1_4p1e1obs_2m.yaml",
    "global_oracle": ROOT / "configs/experiments/iqn_evidence_comparison_20261006/global_oracle_stage1_4p1e1obs_2m.yaml",
}


def _env(mode: str, seed: int = 2026100601) -> tuple[VorAdjEnv, list[dict | None]]:
    cfg = matched.resolved(VARIANTS[mode])
    cfg = deep_update(copy.deepcopy(cfg), cfg["tasks"]["voradj"])
    set_global_config(cfg)
    env = VorAdjEnv(copy.deepcopy(cfg), seed=seed)
    return env, list(env.reset())


def _active_entity_count(obs: dict, entity_type: int, capacity: int) -> int:
    mask = np.asarray(obs["masks"], dtype=bool)
    types = np.asarray(obs["types"], dtype=int)
    return int(np.sum(mask[1 + (capacity if entity_type == 2 else 0):] & (types[1 + (capacity if entity_type == 2 else 0):] == entity_type)))


def _friend_index(env: VorAdjEnv, observer_index: int, row: np.ndarray) -> int:
    observer = env.pursuers[observer_index]
    for friend_index, friend in enumerate(env.pursuers):
        if friend_index == observer_index or friend.deactivated:
            continue
        pos = env._robot_frame(observer, env._position(friend), False)
        vel = env._robot_frame(observer, friend.velocity, True)
        dist = float(np.linalg.norm(pos))
        expected = np.asarray([
            pos[0] / env._distance_scale(), pos[1] / env._distance_scale(),
            vel[0], vel[1], dist / env._distance_scale(), np.arctan2(pos[1], pos[0]),
        ])
        if np.allclose(row[:6], expected, atol=1e-6, rtol=0.0):
            return friend_index
    raise AssertionError("friend token did not map to one physical friend")


def test_three_variants_keep_identical_iqn_parameter_shapes() -> None:
    models = []
    for path in VARIANTS.values():
        cfg = matched.resolved(path)
        model = CoCapIQN(matched.model_config(cfg))
        models.append(model)
    signatures = [
        [(key, tuple(value.shape)) for key, value in model.state_dict().items()]
        for model in models
    ]
    assert signatures[0] == signatures[1] == signatures[2]


def test_local_binary_is_current_direct_evidence_without_z_state() -> None:
    env, observations = _env("local_binary")
    active_targets = sum(not target.deactivated for target in env.evaders)
    assert env._z_update_count == 0
    for i, obs in enumerate(observations):
        if obs is None:
            continue
        direct = bool(env._policy_direct_enemy_ids(i))
        assert float(obs["self"][-1]) == float(direct)
        assert _active_entity_count(obs, 2, env.per_cfg["max_pursuer_num"]) == len(env._policy_direct_enemy_ids(i))
        for row in np.asarray(obs["pursuers"]):
            if np.allclose(row, 0.0):
                continue
            friend = _friend_index(env, i, row)
            assert float(row[-1]) == float(bool(env._policy_direct_enemy_ids(friend)))
    assert active_targets == 1


def test_global_oracle_shows_all_active_target_tokens_and_scalar_flags() -> None:
    env, observations = _env("global_oracle")
    active_targets = sum(not target.deactivated for target in env.evaders)
    assert env._z_update_count == 0
    for i, obs in enumerate(observations):
        if obs is None:
            continue
        assert float(obs["self"][-1]) == float(active_targets > 0)
        assert _active_entity_count(obs, 2, env.per_cfg["max_pursuer_num"]) == active_targets
        for row in np.asarray(obs["pursuers"]):
            if not np.allclose(row, 0.0):
                assert float(row[-1]) == float(active_targets > 0)


def test_global_diagnostics_measure_token_occupancy_and_shortfall_per_observation() -> None:
    class DiagnosticEnv:
        per_cfg = {"max_evader_num": 8}
        evaders = [SimpleNamespace(deactivated=False, collision=False) for _ in range(3)]

        @staticmethod
        def _policy_direct_enemy_ids(_index: int) -> list[int]:
            return []

    def observation(enemy_tokens: int) -> dict:
        return {
            "self": np.zeros(9, dtype=np.float32),
            "pursuers": np.zeros((0, 7), dtype=np.float32),
            "types": np.asarray([2] * enemy_tokens + [0] * (8 - enemy_tokens), dtype=int),
            "masks": np.asarray([True] * enemy_tokens + [False] * (8 - enemy_tokens), dtype=bool),
        }

    diagnostics = EvidenceDiagnostics("global_oracle", "mixed")
    diagnostics.transition(
        DiagnosticEnv(),
        [observation(2), observation(3)],
        None,
        None,
        None,
        [0, 1],
        None,
        None,
        1,
        SimpleNamespace(infos=[]),
    )
    result = diagnostics.finish(DiagnosticEnv())
    assert result["enemy_tokens_visible_sum"] == 5
    assert result["enemy_token_capacity_slots"] == 16
    assert result["enemy_token_occupancy"] == 5 / 16
    assert result["target_token_shortfall_events"] == 1
    assert result["active_target_capacity_overflow_events"] == 0
    assert result["entity_truncation_events"] == 1


def test_local_and_global_do_not_change_reward_or_lifecycle_contract() -> None:
    z = matched.resolved(VARIANTS["z_state"])
    for mode in ("local_binary", "global_oracle"):
        cfg = matched.resolved(VARIANTS[mode])
        assert cfg["reward"] == z["reward"]
        assert cfg["voradj"] == z["voradj"]
        assert cfg["env"] == z["env"]
        assert cfg["runtime_semantic_assertions"] == z["runtime_semantic_assertions"]
        assert cfg["perception"]["friend_ordering_mode"] == "physical_only"
        assert cfg["perception"]["max_evader_num"] == z["perception"]["max_evader_num"]
        assert cfg["iqn"]["self_feature_dim"] == z["iqn"]["self_feature_dim"]
        assert cfg["iqn"]["pursuer_feature_dim"] == z["iqn"]["pursuer_feature_dim"]
