from __future__ import annotations

import copy
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cocap_voradj.dynamics.continuous_action import body_to_world
from cocap_voradj.dynamics.robot import Robot
from cocap_voradj.envs.base import CoCapEnv
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.continuous.central_attention_critic import (
    CentralAttentionCritic,
    CentralCriticConfig,
)
from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoderConfig
from cocap_voradj.models.continuous.radial_actor import RadialActorConfig
from cocap_voradj.training.continuous.central_sac import CentralSACConfig, CentralSACTrainer
from cocap_voradj.training.continuous.central_schema import build_central_global_obs
from cocap_voradj.training.continuous.formal_config import (
    resolve_formal_config,
    scene_config,
    validate_formal_config,
)
from cocap_voradj.training.continuous.joint_replay import (
    FOCAL_BUCKETS,
    FocalReplaySampler,
    JointReplayBuffer,
)
from cocap_voradj.training.continuous.local_sac import sac_bootstrap_mask
from cocap_voradj.training.trainer import set_global_config
from tools.run_continuous_ctde_training import (
    _make_trainer,
    _pad_local_obs_tree,
    _reset_pure_recovery,
    _split_termination_flags,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "configs/experiments/continuous_marl_20260804/p6_formal_central_masac_4v1.yaml"


def make_obs_tree(max_agents: int = 12, actor_pursuers: int = 12, fill: float = 1.0) -> dict:
    token_count = 1 + actor_pursuers + 8 + 5
    return {
        "self": np.full((max_agents, 9), fill, dtype=np.float32),
        "pursuers": np.full((max_agents, actor_pursuers, 7), fill, dtype=np.float32),
        "evaders": np.full((max_agents, 8, 7), fill, dtype=np.float32),
        "obstacles": np.full((max_agents, 5, 5), fill, dtype=np.float32),
        "masks": np.ones((max_agents, token_count), dtype=bool),
        "types": np.zeros((max_agents, token_count), dtype=np.int64),
    }


def make_central(max_agents: int = 12) -> dict:
    active = np.zeros((1, max_agents), dtype=bool)
    active[:, :4] = True
    return {
        "self": np.zeros((1, max_agents, 9), dtype=np.float32),
        "pursuers": np.zeros((1, max_agents, max_agents, 7), dtype=np.float32),
        "evaders": np.zeros((1, 8, 7), dtype=np.float32),
        "obstacles": np.zeros((1, 5, 5), dtype=np.float32),
        "pursuer_mask": active[:, :, None] & active[:, None, :],
        "evader_mask": np.zeros((1, 8), dtype=bool),
        "obstacle_mask": np.zeros((1, 5), dtype=bool),
        "active_mask": active,
    }


def add_focal_transition(
    replay: JointReplayBuffer,
    *,
    roles: np.ndarray,
    phase: str = "pre_capture",
    scene: str = "capture",
    origin: str = "map_random",
    seed_value: float = 1.0,
) -> int:
    active = roles > 0
    return replay.add(
        local_obs=make_obs_tree(fill=seed_value),
        next_local_obs=make_obs_tree(fill=seed_value + 1.0),
        global_state={key: value[0] for key, value in make_central().items()},
        next_global_state={key: value[0] for key, value in make_central().items()},
        actions=np.full((replay.max_agents, 2), 0.1, dtype=np.float32),
        rewards=np.full(replay.max_agents, 1.0, dtype=np.float32),
        active_mask=active,
        terminated=np.zeros(replay.max_agents, dtype=bool),
        truncated=np.zeros(replay.max_agents, dtype=bool),
        metadata={
            "phase": phase,
            "scene": scene,
            "origin": origin,
            "regime": "active_target" if phase == "pre_capture" else "coverage_only",
            "coverage_only": phase != "pre_capture",
            "active_target": phase == "pre_capture",
            "event_ids": [],
            "task_label": "",
        },
        agent_role_id=roles,
    )


def test_formal_resolved_config_contract() -> None:
    config = resolve_formal_config(FORMAL)
    assert config["algorithm"] == "masac_ctde"
    assert config["critic_mode"] == "central"
    assert config["env"]["width"] == 120.0
    assert config["env"]["height"] == 120.0
    assert config["env"]["num_obstacles"] == 1
    assert config["env"]["num_pursuers"] == 4
    assert config["training"]["max_agents"] == 12
    assert config["central_critic"]["max_agents"] == 12
    assert config["replay"]["max_agents"] == 12
    assert config["training"]["batch_size"] == 128
    assert config["training"]["grad_clip_norm"] == 0.5
    assert config["training"]["warmup_joint_transitions"] == 5000
    assert config["training"]["update_every_env_steps"] == 4
    assert config["training"]["total_env_steps"] == 2_000_000
    assert config["action"]["mode"] == "acceleration_2d_world"
    assert config["action"]["a_max"] == 0.4
    assert config["dynamics"]["profile"] == "continuous_parity_v1"
    assert abs(config["dynamics"]["linear_drag_coefficient"] - 0.4 / 3.0) < 1e-6
    for scene in ("capture", "pure_ce", "mixed_crms"):
        sc = scene_config(config, scene)
        assert sc["env"]["episode_max_length"] != 128


def test_formal_rejects_local_smoke_and_legacy_action_aliases() -> None:
    base = copy.deepcopy(resolve_formal_config(FORMAL))
    local = copy.deepcopy(base)
    local["critic_mode"] = "local"
    with pytest.raises(ValueError, match="central"):
        validate_formal_config(local)
    smoke = copy.deepcopy(base)
    smoke["trainer_profile"] = "smoke"
    with pytest.raises(ValueError, match="smoke"):
        validate_formal_config(smoke)
    body = copy.deepcopy(base)
    body["action"]["mode"] = "acceleration_2d_body"
    with pytest.raises(ValueError, match="forbidden"):
        validate_formal_config(body)
    velocity = copy.deepcopy(base)
    velocity["action"]["mode"] = "velocity_2d_body"
    with pytest.raises(ValueError, match="forbidden"):
        validate_formal_config(velocity)
    no_map = copy.deepcopy(base)
    no_map["env"].pop("width")
    with pytest.raises(ValueError, match="mandatory"):
        validate_formal_config(no_map)


def test_three_scene_hashes_and_task_fields() -> None:
    config = resolve_formal_config(FORMAL)
    scenes = {name: scene_config(config, name) for name in ("capture", "pure_ce", "mixed_crms")}
    hashes = {name: hash(json_safe(value)) for name, value in scenes.items()}
    assert len(set(hashes.values())) == 3
    assert scenes["capture"]["env"]["num_evaders"] == 1
    assert scenes["capture"]["env"]["pursuer_spawn_mode"] == "map_random"
    assert scenes["capture"]["env"]["pre_capture_max_length"] == 1000
    assert scenes["capture"]["voradj"]["capture_episode_ends_on_capture"] is True
    assert scenes["pure_ce"]["env"]["num_evaders"] == 0
    assert scenes["pure_ce"]["env"]["episode_max_length"] == 1500
    assert scenes["mixed_crms"]["env"]["num_evaders"] == 1
    assert scenes["mixed_crms"]["env"]["pre_capture_max_length"] == 1000
    assert scenes["mixed_crms"]["env"]["episode_max_length"] == 1500
    assert scenes["mixed_crms"]["reward"]["post_capture_coverage_window_steps"] == 500
    assert scenes["mixed_crms"]["voradj"]["capture_episode_ends_on_capture"] is False


def json_safe(value: object) -> str:
    return repr(value)


def test_actor_does_not_leak_critic_global_state() -> None:
    config = resolve_formal_config(FORMAL)
    trainer = _make_trainer(config, "cpu")
    obs = {
        "self": torch.randn(4, 12, 9),
        "pursuers": torch.randn(4, 12, 12, 7),
        "evaders": torch.randn(4, 12, 8, 7),
        "obstacles": torch.randn(4, 12, 5, 5),
        "masks": torch.ones(4, 12, 26, dtype=torch.bool),
        "types": torch.zeros(4, 12, 26, dtype=torch.long),
    }
    central_a = make_central()
    central_b = make_central()
    central_b["self"] = central_b["self"] + 123.0
    central_b["evaders"] = central_b["evaders"] + 456.0
    with torch.no_grad():
        action_a, _, _ = trainer._actor_sample(obs)
    batch = {
        "local_obs": obs,
        "next_local_obs": obs,
        "global_state": central_b,
        "next_global_state": central_b,
        "actions": torch.zeros(4, 12, 2),
        "active_mask": torch.zeros(4, 12, dtype=torch.bool),
    }
    assert all(
        not name.startswith("global")
        for name, _ in trainer.actor.named_parameters()
    )
    # The actor API never receives global state; changing it cannot change output.
    assert "global_state" not in {name for name, _ in trainer.actor.named_parameters()}
    assert action_a.shape == (4, 12, 2)
    del batch


def test_central_critic_sensitive_to_teammate_action_and_padding() -> None:
    critic = CentralAttentionCritic(
        CentralCriticConfig(hidden_dim=32, num_heads=4, num_layers=1, max_agents=12)
    ).eval()
    obs = make_central()
    obs_t = {key: torch.as_tensor(value) for key, value in obs.items()}
    commands = torch.zeros(1, 12, 2)
    with torch.no_grad():
        baseline = critic(obs_t, commands)
        teammate = commands.clone()
        teammate[:, 1] = torch.tensor([2.0, -1.0])
        changed = critic(obs_t, teammate)
        padded = commands.clone()
        padded[:, 8] = torch.tensor([2.0, -1.0])
        padded_q = critic(obs_t, padded)
    assert not torch.allclose(baseline[:, 0], changed[:, 0])
    assert torch.allclose(baseline[:, :4], padded_q[:, :4], atol=1e-6, rtol=1e-6)


def test_central_critic_4_8_12_agent_shapes() -> None:
    for count in (4, 8, 12):
        model = CentralAttentionCritic(
            CentralCriticConfig(hidden_dim=16, num_heads=4, num_layers=1, max_agents=count)
        ).eval()
        active = np.zeros((2, count), dtype=bool)
        active[:, :count] = True
        obs = {
            "self": torch.randn(2, count, 9),
            "pursuers": torch.randn(2, count, count, 7),
            "evaders": torch.randn(2, 8, 7),
            "obstacles": torch.randn(2, 5, 5),
            "pursuer_mask": torch.as_tensor(active[:, :, None] & active[:, None, :]),
            "evader_mask": torch.ones(2, 8, dtype=torch.bool),
            "obstacle_mask": torch.ones(2, 5, dtype=torch.bool),
            "active_mask": torch.as_tensor(active),
        }
        with torch.no_grad():
            q = model(obs, torch.randn(2, count, 2))
        assert q.shape == (2, count)
        assert torch.isfinite(q).all()


def test_terminated_does_not_bootstrap_but_truncated_does() -> None:
    terminated = torch.tensor([True, False])
    truncated = torch.tensor([False, True])
    mask = sac_bootstrap_mask(terminated, truncated)
    assert mask.tolist() == [0.0, 1.0]


def test_world_acceleration_constant_command_analytic() -> None:
    a = np.asarray([0.4, 0.0])
    k = 0.4 / 3.0
    dt = 0.05
    robot = Robot(0)
    robot.x = 0.0
    robot.y = 0.0
    robot.max_speed = 3.0
    robot.velocity = np.zeros(2)
    robot.coefficient_water_resistance = k
    robot.update_state_acceleration_world(a)
    v, x = 0.0, 0.0
    for _ in range(10):
        v_next = v + (0.4 - k * v) * dt
        x += 0.5 * (v + v_next) * dt
        v = v_next
    assert np.isclose(robot.x, x, atol=1e-9)
    assert np.isclose(robot.speed, v, atol=1e-9)


def test_no_drag_keeps_speed_and_parity_drag_decelerates() -> None:
    robot = Robot(0)
    robot.x = 0.0
    robot.y = 0.0
    robot.max_speed = 3.0
    robot.velocity = np.asarray([1.0, 0.5])
    robot.coefficient_water_resistance = 0.0
    robot.update_state_acceleration_world(np.zeros(2))
    assert np.allclose(robot.velocity, [1.0, 0.5], atol=1e-9)
    speed_after_no_drag = robot.speed

    robot2 = Robot(1)
    robot2.x = 0.0
    robot2.y = 0.0
    robot2.max_speed = 3.0
    robot2.velocity = np.asarray([1.0, 0.5])
    robot2.coefficient_water_resistance = 0.4 / 3.0
    robot2.update_state_acceleration_world(np.zeros(2))
    assert robot2.speed < speed_after_no_drag


def test_world_body_equivalent_when_yaw_zero_no_drag() -> None:
    a = np.asarray([0.3, -0.2])
    world = Robot(0)
    world.x = 0.0
    world.y = 0.0
    world.theta = 0.0
    world.max_speed = 3.0
    world.velocity = np.zeros(2)
    world.coefficient_water_resistance = 0.0
    world.update_state_acceleration_world(a)
    body = Robot(1)
    body.x = 0.0
    body.y = 0.0
    body.theta = 0.0
    body.max_speed = 3.0
    body.velocity = np.zeros(2)
    body.coefficient_water_resistance = 0.0
    body.update_state_acceleration_body(body_to_world(a, 0.0))
    assert np.allclose(world.velocity, body.velocity, atol=1e-9)
    assert np.isclose(world.x, body.x, atol=1e-9)


def test_speed_cap_each_substep_and_callback_interrupt() -> None:
    robot = Robot(0)
    robot.x = 0.0
    robot.y = 0.0
    robot.max_speed = 3.0
    robot.velocity = np.asarray([2.8, 0.0])
    robot.coefficient_water_resistance = 0.0
    assert robot.update_state_acceleration_world(np.asarray([0.8, 0.0])) is True
    assert robot.speed <= 3.0 + 1e-9

    interrupted = Robot(1)
    interrupted.x = 0.0
    interrupted.y = 0.0
    interrupted.max_speed = 3.0
    interrupted.velocity = np.zeros(2)
    interrupted.coefficient_water_resistance = 0.0
    calls = []
    interrupted.update_state_acceleration_world(
        np.asarray([0.4, 0.0]),
        substep_callback=lambda: (calls.append(interrupted.x), setattr(interrupted, "deactivated", True))[1],
    )
    assert len(calls) == 1
    assert interrupted.deactivated is True


def test_env_world_action_replay_matches_executed_action() -> None:
    config = scene_config(resolve_formal_config(FORMAL), "capture")
    set_global_config(config)
    env = VorAdjEnv(config, seed=2026080601)
    env.reset()
    observations = list(env.get_observations())
    padded = _pad_local_obs_tree(observations, 12, 12)
    actions = np.asarray([[0.2, -0.1]] * len(env.pursuers), dtype=np.float32)
    outcome = env.step(actions.tolist(), [None] * len(env.evaders))
    replay = JointReplayBuffer(capacity=8, max_agents=12, seed=1)
    active = np.zeros(12, dtype=bool)
    active[: len(env.pursuers)] = True
    terminated, truncated = _split_termination_flags(outcome.dones, outcome.infos)
    replay.add(
        local_obs=padded,
        next_local_obs=padded,
        global_state=build_central_global_obs(env, max_agents=12, max_evaders=8, max_obstacles=5, self_feature_dim=9),
        next_global_state=build_central_global_obs(env, max_agents=12, max_evaders=8, max_obstacles=5, self_feature_dim=9),
        actions=np.vstack([actions, np.zeros((12 - len(actions), 2), dtype=np.float32)]).astype(np.float32),
        rewards=np.concatenate([outcome.rewards, np.zeros(12 - len(outcome.rewards), dtype=np.float32)]).astype(np.float32),
        active_mask=active,
        terminated=np.concatenate([terminated, np.zeros(12 - len(terminated), dtype=bool)]),
        truncated=np.concatenate([truncated, np.zeros(12 - len(truncated), dtype=bool)]),
        metadata={"phase": "pre_capture", "scene": "capture", "origin": "map_random", "event_ids": []},
        agent_role_id=np.concatenate([np.ones(len(env.pursuers), dtype=np.uint8), np.zeros(12 - len(env.pursuers), dtype=np.uint8)]),
    )
    batch = replay.sample(1)
    np.testing.assert_allclose(batch["actions"][0, : len(actions)], actions, atol=1e-6)


def test_spawn_reset_distribution_10000() -> None:
    config = scene_config(resolve_formal_config(FORMAL), "pure_ce")
    set_global_config(config)
    rng = np.random.default_rng(1234)
    pool = deque(maxlen=100)
    pool.append({"positions": [[1.0, 1.0]] * 4, "active_mask": [True] * 4})
    origins = []
    for _ in range(10_000):
        cfg, origin, positions, active = _reset_pure_recovery(copy.deepcopy(config), pool, rng)
        origins.append(origin)
        assert positions is not None or origin != "capture_snapshot"
    counts = {name: origins.count(name) for name in ("capture_snapshot", "map_random", "inner_cluster")}
    assert counts["capture_snapshot"] / 10_000 == pytest.approx(0.75, abs=0.02)
    non_capture = counts["map_random"] + counts["inner_cluster"]
    assert counts["map_random"] / non_capture == pytest.approx(0.5, abs=0.03)


def test_focal_batch_exact_128_quota() -> None:
    replay = JointReplayBuffer(capacity=1000, max_agents=12, seed=7)
    for i in range(200):
        roles = np.zeros(12, dtype=np.uint8)
        roles[0] = 1
        roles[1] = 2
        roles[2] = 3
        roles[3] = 3
        phase = "pre_capture" if i % 2 == 0 else ("post_capture" if i % 3 == 0 else "pure_coverage")
        add_focal_transition(replay, roles=roles, phase=phase, seed_value=float(i))
    quotas = {
        "pre_capture_pursuing": 64,
        "pre_capture_support": 8,
        "pre_capture_coverage": 8,
        "post_capture_coverage": 32,
        "pure_recovery_coverage": 16,
    }
    sampler = FocalReplaySampler(quotas, seed=3)
    batch = replay.sample(128, sampler=sampler)
    stats = batch["sampling_stats"]
    assert batch["focal_agent_id"].shape == (128,)
    assert batch["focal_mask"].shape == (128, 12)
    assert stats["actual_batch_size"] == 128
    assert stats["requested_counts"] == quotas
    assert stats["silent_uniform_fallback"] is False
    assert stats["max_items_per_joint_transition"] <= 4


def test_same_joint_transition_multiple_focal_agents_and_full_critic_context() -> None:
    replay = JointReplayBuffer(capacity=16, max_agents=12, seed=1)
    roles = np.zeros(12, dtype=np.uint8)
    roles[:4] = [1, 2, 3, 3]
    tid = add_focal_transition(replay, roles=roles, phase="pre_capture")
    quotas = {
        "pre_capture_pursuing": 1,
        "pre_capture_support": 1,
        "pre_capture_coverage": 1,
        "post_capture_coverage": 0,
        "pure_recovery_coverage": 0,
    }
    sampler = FocalReplaySampler(quotas, seed=1)
    batch = replay.sample(3, sampler=sampler)
    assert set(batch["transition_ids"].tolist()) == {tid}
    assert sorted(batch["focal_agent_id"].tolist()) == [0, 1, 2]
    assert batch["global_state"]["self"].shape == (3, 12, 9)
    assert batch["actions"].shape == (3, 12, 2)
    assert batch["focal_mask"].sum(dim=1).tolist() == [1, 1, 1]


def test_no_duplicate_pair_and_max_four_per_transition() -> None:
    replay = JointReplayBuffer(capacity=16, max_agents=12, seed=9)
    roles = np.zeros(12, dtype=np.uint8)
    roles[:12] = [1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 3, 3]
    add_focal_transition(replay, roles=roles, phase="pre_capture")
    quotas = {
        "pre_capture_pursuing": 2,
        "pre_capture_support": 1,
        "pre_capture_coverage": 1,
        "post_capture_coverage": 0,
        "pure_recovery_coverage": 0,
    }
    sampler = FocalReplaySampler(
        quotas,
        max_focal_items_per_joint_transition=4,
        seed=2,
    )
    batch = replay.sample(4, sampler=sampler)
    pairs = list(zip(batch["transition_ids"].tolist(), batch["focal_agent_id"].tolist()))
    assert len(pairs) == len(set(pairs))
    assert len(set(batch["transition_ids"].tolist())) == 1
    assert len(pairs) <= 4


def test_role_uniqueness_and_inactive_consistency() -> None:
    replay = JointReplayBuffer(capacity=4, max_agents=12, seed=1)
    roles = np.zeros(12, dtype=np.uint8)
    roles[:4] = [1, 1, 3, 3]
    tid = add_focal_transition(replay, roles=roles, phase="pre_capture")
    assert np.array_equal(replay.get(tid).agent_role_id, roles)
    assert int(np.count_nonzero(replay.get(tid).agent_role_id)) == int(np.count_nonzero(replay.get(tid).active_mask))
    invalid = roles.copy()
    invalid[4] = 1
    active = np.zeros(12, dtype=bool)
    active[:4] = True
    with pytest.raises(ValueError, match="non-zero primary role"):
        replay.add(
            local_obs=make_obs_tree(),
            next_local_obs=make_obs_tree(),
            global_state={key: value[0] for key, value in make_central().items()},
            next_global_state={key: value[0] for key, value in make_central().items()},
            actions=np.zeros((12, 2), dtype=np.float32),
            rewards=np.zeros(12, dtype=np.float32),
            active_mask=active,
            terminated=np.zeros(12, dtype=bool),
            truncated=np.zeros(12, dtype=bool),
            metadata={"phase": "pre_capture", "scene": "capture", "origin": "map_random", "event_ids": []},
            agent_role_id=invalid,
        )
    roles[:4] = [1, 2, 3, 3]
    tid = add_focal_transition(replay, roles=roles, phase="pre_capture")
    assert np.array_equal(replay.get(tid).agent_role_id, roles)


def test_ring_overwrite_invalidates_stale_role_index() -> None:
    replay = JointReplayBuffer(capacity=1, max_agents=12, seed=3)
    roles = np.zeros(12, dtype=np.uint8)
    roles[0] = 1
    first = add_focal_transition(replay, roles=roles, phase="pre_capture", seed_value=0.0)
    second = add_focal_transition(replay, roles=roles, phase="pre_capture", seed_value=1.0)
    assert first != second
    assert replay.focal_index_sizes["pre_capture_pursuing"] == 1
    batch = replay.sample(1, sampler=FocalReplaySampler(
        {"pre_capture_pursuing": 1, "pre_capture_support": 0, "pre_capture_coverage": 0, "post_capture_coverage": 0, "pure_recovery_coverage": 0},
        seed=1,
    ))
    assert batch["transition_ids"].tolist() == [second]


def test_bucket_fallback_and_no_silent_uniform() -> None:
    replay = JointReplayBuffer(capacity=32, max_agents=12, seed=4)
    roles = np.zeros(12, dtype=np.uint8)
    roles[:4] = 3
    for i in range(8):
        add_focal_transition(replay, roles=roles, phase="pre_capture", seed_value=float(i))
    quotas = {
        "pre_capture_pursuing": 0,
        "pre_capture_support": 8,
        "pre_capture_coverage": 0,
        "post_capture_coverage": 0,
        "pure_recovery_coverage": 0,
    }
    sampler = FocalReplaySampler(quotas, seed=5)
    batch = replay.sample(8, sampler=sampler)
    stats = batch["sampling_stats"]
    assert stats["fallback_count"] > 0
    assert stats["actual_counts"]["pre_capture_coverage"] > 0
    assert stats["silent_uniform_fallback"] is False


def test_capture_snapshot_pool_save_restore_and_sample() -> None:
    config = scene_config(resolve_formal_config(FORMAL), "pure_ce")
    pool = deque(maxlen=100)
    snapshot = {
        "step": 42,
        "positions": [[10.0, 10.0], [20.0, 10.0], [10.0, 20.0], [20.0, 20.0]],
        "active_mask": [True, True, True, True],
    }
    pool.append(snapshot)
    rng = np.random.default_rng(6)
    cfg, origin, positions, active = _reset_pure_recovery(copy.deepcopy(config), pool, rng)
    assert origin == "capture_snapshot"
    assert positions == snapshot["positions"]
    restored = deque(maxlen=100)
    for item in list(pool):
        restored.append(item)
    cfg2, origin2, _, _ = _reset_pure_recovery(copy.deepcopy(config), restored, rng)
    assert origin2 in {"capture_snapshot", "map_random", "inner_cluster"}
    assert len(restored) == 1


def test_replay_and_checkpoint_manifest_mismatch_refuses_resume(tmp_path) -> None:
    replay = JointReplayBuffer(capacity=4, max_agents=12, seed=1)
    roles = np.zeros(12, dtype=np.uint8)
    roles[0] = 1
    add_focal_transition(replay, roles=roles, phase="pre_capture")
    path = tmp_path / "replay.pkl"
    manifest = {"action_mode": "acceleration_2d_world", "max_agents": 12}
    replay.save(path, manifest)
    with pytest.raises(ValueError, match="manifest mismatch"):
        JointReplayBuffer.load(path, {"action_mode": "acceleration_2d_world", "max_agents": 4})

    config = resolve_formal_config(FORMAL)
    trainer = _make_trainer(config, "cpu")
    ckpt = tmp_path / "trainer.pt"
    trainer.save_checkpoint(ckpt, manifest)
    with pytest.raises(ValueError, match="contract mismatch"):
        trainer.load_checkpoint(ckpt, {"action_mode": "acceleration_2d_world", "max_agents": 4})


def test_env_pre_capture_timeout_is_truncated() -> None:
    config = scene_config(resolve_formal_config(FORMAL), "mixed_crms")
    config["env"]["pre_capture_max_length"] = 2
    config["env"]["episode_max_length"] = 2
    set_global_config(config)
    env = VorAdjEnv(config, seed=2026080602)
    env.reset()
    result = None
    for _ in range(2):
        result = env.step(
            [np.zeros(2, dtype=np.float32) for _ in env.pursuers],
            [None] * len(env.evaders),
        )
        if all(result.dones):
            break
    assert result is not None
    terminated, truncated = _split_termination_flags(result.dones, result.infos)
    assert not bool(terminated.any())
    assert bool(truncated.all())


def test_formal_world_env_preserves_one_obstacle_and_120_map() -> None:
    config = scene_config(resolve_formal_config(FORMAL), "capture")
    set_global_config(config)
    env = VorAdjEnv(config, seed=2026080603)
    env.reset()
    assert env.width == 120.0
    assert env.height == 120.0
    assert len(env.obstacles) == 1
    assert env.action_mode == "acceleration_2d_world"
    assert env.continuous_world_action is True


def test_central_trainer_focal_update_is_finite() -> None:
    replay = JointReplayBuffer(capacity=128, max_agents=12, seed=11)
    for i in range(64):
        roles = np.zeros(12, dtype=np.uint8)
        roles[:4] = [1, 2, 3, 3]
        phase = "pre_capture" if i % 2 == 0 else ("post_capture" if i % 3 == 0 else "pure_coverage")
        add_focal_transition(replay, roles=roles, phase=phase, seed_value=float(i))
    quotas = {
        "pre_capture_pursuing": 16,
        "pre_capture_support": 2,
        "pre_capture_coverage": 2,
        "post_capture_coverage": 8,
        "pure_recovery_coverage": 4,
    }
    batch = replay.sample(32, sampler=FocalReplaySampler(quotas, seed=12))
    trainer = CentralSACTrainer(
        encoder_config=LocalEntityTokenEncoderConfig(hidden_dim=16, num_heads=4, num_layers=1, max_pursuers=12),
        actor_config=RadialActorConfig(hidden_dim=16, a_max=0.4, decision_dt=0.5),
        critic_config=CentralCriticConfig(hidden_dim=16, num_heads=4, num_layers=1, max_agents=12),
        config=CentralSACConfig(hidden_dim=16, grad_clip_norm=0.5),
        device="cpu",
    )
    metrics = trainer.update(batch)
    assert metrics["finite"] == 1.0
    assert metrics["focal_count"] == 32.0
