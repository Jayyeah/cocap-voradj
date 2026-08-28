from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from cocap_voradj.dynamics.continuous_action import ActionContractError, DesiredBodyVelocityActionAdapter
from cocap_voradj.dynamics.robot import Robot
from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoder, LocalEntityTokenEncoderConfig
from cocap_voradj.models.small_step_ac import CategoricalGridActor, CentralValueNetwork, DeterministicAWActor
from cocap_voradj.training.continuous.formal_config import resolve_ladder_config
from cocap_voradj.training.continuous.joint_replay import JointReplayBuffer
from cocap_voradj.training.replay import ReplayBuffer
from tools.run_small_step_ac_migration import configure_environment, empty_rollout, finalize_resume_checkpoint, stack_rollout


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/experiments/small_step_ac_migration_20260828"


def local_obs(batch: int = 4):
    config = LocalEntityTokenEncoderConfig(hidden_dim=32, num_heads=4, num_layers=1, self_feature_dim=9,
                                           max_pursuers=8, max_evaders=8, max_obstacles=5, dropout=0.0)
    tokens = 1 + 8 + 8 + 5
    masks = torch.zeros(batch, tokens, dtype=torch.bool); masks[:, 0] = True; masks[:, 1:4] = True
    types = torch.tensor([0] + [1] * 8 + [2] * 8 + [3] * 5).repeat(batch, 1)
    obs = {"self": torch.zeros(batch, 9), "pursuers": torch.zeros(batch, 8, 7),
           "evaders": torch.zeros(batch, 8, 7), "obstacles": torch.zeros(batch, 5, 5),
           "masks": masks, "types": types}
    return config, obs


def global_obs(batch: int = 2):
    active = torch.zeros(batch, 12, dtype=torch.bool); active[:, :4] = True
    return {"self": torch.zeros(batch, 12, 9), "pursuers": torch.zeros(batch, 12, 12, 7),
            "evaders": torch.zeros(batch, 8, 7), "obstacles": torch.zeros(batch, 5, 5),
            "pursuer_mask": active[:, None, :].expand(-1, 12, -1) & active[:, :, None],
            "evader_mask": torch.ones(batch, 8, dtype=torch.bool), "obstacle_mask": torch.ones(batch, 5, dtype=torch.bool),
            "active_mask": active}


def test_all_formal_configs_resolve_to_same_golden_contract():
    fingerprints = []
    for path in sorted(CONFIG_DIR.glob("*_seed*.yaml")):
        config = resolve_ladder_config(path)
        fingerprints.append((config["env"]["collision_semantics"], config["env"]["episode_max_length"],
                             config["reward"]["k_required"], config["reward"]["min_active_pursuers"],
                             config["voradj"]["is_pursuing_release_delay_steps"], config["perception"]["global_evader_visibility"],
                             config["actor"]["hidden_dim"], config["actor"]["num_heads"], config["actor"]["num_layers"]))
    assert len(fingerprints) == 12
    assert len(set(fingerprints)) == 1
    assert fingerprints[0] == ("synchronized_swept_v1", 1000, 3, 2, 10, False, 256, 8, 4)
    assert all(resolve_ladder_config(path)["actor"]["max_pursuers"] == 8 for path in CONFIG_DIR.glob("*_seed*.yaml"))


def test_iqn_runtime_retains_historical_padding_and_replay_contract():
    root = resolve_ladder_config(CONFIG_DIR / "iqn_vxy9_seed1.yaml")
    config = configure_environment(root, "iqn_vxy9")
    assert config["actor"]["max_pursuers"] == 8
    assert config["iqn"]["replay_capacity"] == 1_000_000
    assert config["iqn"]["min_replay_size"] == 3_000
    assert config["iqn"]["target_update_freq"] == 10_000


def test_aw_grid_is_exact_legacy_cartesian_order():
    grid = np.asarray([(a, w) for a in (-0.4, 0.0, 0.4) for w in (-math.pi / 6, 0.0, math.pi / 6)], dtype=np.float32)
    config, obs = local_obs()
    actor = CategoricalGridActor(LocalEntityTokenEncoder(config), grid)
    action, log_prob, index = actor.sample(obs, deterministic=True)
    assert action.shape == (4, 2); assert log_prob.shape == (4,); assert index.shape == (4,)
    assert torch.allclose(action, actor.action_grid[index])
    sampled = {int(actor.sample(obs, deterministic=False)[2][0]) for _ in range(50)}
    assert len(sampled) > 1


def test_continuous_actor_bounds_and_deterministic_contract():
    config, obs = local_obs()
    actor = DeterministicAWActor(LocalEntityTokenEncoder(config), 0.4, math.pi / 6)
    first = actor.sample(obs, deterministic=True)[0]; second = actor.sample(obs, deterministic=False)[0]
    assert torch.equal(first, second)
    assert torch.all(first[:, 0].abs() <= 0.4 + 1e-6)
    assert torch.all(first[:, 1].abs() <= math.pi / 6 + 1e-6)


def test_central_value_is_action_free_and_masks_inactive_rows():
    value = CentralValueNetwork(hidden_dim=32, num_heads=4, num_layers=1)
    obs = global_obs()
    output = value(obs)
    assert output.shape == (2, 12)
    assert torch.count_nonzero(output[:, 4:]) == 0
    with pytest.raises(TypeError):
        value(obs, torch.zeros(2, 12, 2))


def test_vxy9_grid_and_adapter_are_dimensionally_audited():
    vmax = 3.0; component = vmax / math.sqrt(2.0)
    grid = np.asarray([(x, y) for x in (-component, 0.0, component) for y in (-component, 0.0, component)])
    adapter = DesiredBodyVelocityActionAdapter(vmax, 0.5)
    for action in grid:
        assert np.linalg.norm(adapter.validate(action)) <= vmax + 1e-6
    with pytest.raises(ActionContractError): adapter.validate([vmax, vmax])


def test_vxy_servo_retains_acceleration_and_speed_limits():
    robot = Robot(0); robot.max_speed = 3.0; robot.coefficient_water_resistance = 0.4 / 3.0
    robot.start = np.asarray([10.0, 10.0]); robot.init_theta = 0.0; robot.init_speed = 0.0; robot.reset_state()
    before = robot.velocity.copy()
    robot.update_state_desired_velocity_body(np.asarray([3.0 / math.sqrt(2.0)] * 2), acceleration_limit=0.4)
    delta = np.linalg.norm(robot.velocity - before)
    assert delta <= 0.4 * robot.N * robot.dt + 1e-6
    for _ in range(100): robot.update_state_desired_velocity_body(np.asarray([3.0 / math.sqrt(2.0)] * 2), acceleration_limit=0.4)
    assert robot.speed <= 3.0 + 1e-6


def test_ppo_rollout_and_offpolicy_replay_contracts():
    _, obs = local_obs(batch=4)
    local = {key: value.numpy() for key, value in obs.items()}
    rollout = empty_rollout()
    global_state = {key: value.numpy()[0] for key, value in global_obs(batch=1).items()}
    values = {"local_obs": local, "global_obs": global_state, "actions": np.zeros((4, 2), np.float32),
              "latent": np.zeros(4, np.int64), "log_prob": np.zeros(4, np.float32), "values": np.zeros(12, np.float32),
              "next_values": np.zeros(12, np.float32), "rewards": np.zeros(12, np.float32),
              "active_mask": np.asarray([True] * 4 + [False] * 8), "terminated": np.zeros(12, bool), "episode_end": np.zeros(12, bool)}
    for key, value in values.items(): rollout[key].append(value)
    stacked = stack_rollout(rollout)
    assert stacked["actions"].shape == (1, 4, 2)
    assert stacked["global_obs"]["self"].shape == (1, 12, 9)

    joint = JointReplayBuffer(capacity=8, max_agents=4, seed=7)
    joint.add(local_obs=local, next_local_obs=local, global_state={"x": np.zeros(3)}, next_global_state={"x": np.zeros(3)},
              actions=np.zeros((4, 2)), rewards=np.zeros(4), active_mask=np.ones(4, bool), terminated=np.zeros(4, bool),
              truncated=np.zeros(4, bool), agent_role_id=np.ones(4, np.uint8),
              metadata={"regime": "active_target", "event_ids": [], "phase": "pre_capture", "scene": "capture", "origin": "test"})
    assert joint.sample(1)["actions"].shape == (1, 4, 2)
    item = ReplayBuffer(8); one = {key: value[0] for key, value in local.items()}
    item.add(one, 0, 1.0, one, False); assert item.sample(1, "cpu")["actions"].shape == (1,)


def test_checkpoint_roundtrip_preserves_actor_outputs(tmp_path):
    config, obs = local_obs()
    actor = DeterministicAWActor(LocalEntityTokenEncoder(config), 0.4, math.pi / 6)
    before = actor(obs).detach().clone(); path = tmp_path / "actor.pt"
    torch.save({"actor": actor.state_dict(), "rng": torch.get_rng_state()}, path)
    restored = DeterministicAWActor(LocalEntityTokenEncoder(config), 0.4, math.pi / 6)
    payload = torch.load(path, weights_only=False); restored.load_state_dict(payload["actor"]); torch.set_rng_state(payload["rng"])
    assert torch.equal(before, restored(obs).detach())

def test_final_full_replay_retention_contract(tmp_path):
    retained = tmp_path / "retained.pt"; retained.write_bytes(b"full-resume")
    finalize_resume_checkpoint(retained, retain=True)
    assert retained.read_bytes() == b"full-resume"

    released = tmp_path / "released.pt"; released.write_bytes(b"full-resume")
    finalize_resume_checkpoint(released, retain=False)
    assert not released.exists()

    for path in sorted(CONFIG_DIR.glob("*_seed*.yaml")):
        config = resolve_ladder_config(path)
        algorithm = config["small_step_ac"]["algorithm"]
        seed_index = int(path.stem.rsplit("seed", 1)[1])
        retained_expected = algorithm not in {"iqn_vxy9", "td3_aw"} or seed_index == 1
        assert bool(config["small_step_ac"]["retain_final_full_resume"]) is retained_expected
