from __future__ import annotations

from dataclasses import replace

import numpy as np

from open_encirclement_ctde.environment import CorrectedRoundupEnv, RoundupSpec
from open_encirclement_ctde.geometry import point_in_triangle
from open_encirclement_ctde.oracle import run_oracle_episode


def test_triangle_inside_edge_outside_and_degenerate() -> None:
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert point_in_triangle(np.array([0.2, 0.2]), triangle)
    assert point_in_triangle(np.array([0.5, 0.5]), triangle)
    assert not point_in_triangle(np.array([0.8, 0.8]), triangle)
    assert not point_in_triangle(np.array([0.5, 0.0]), np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]))


def capture_env() -> CorrectedRoundupEnv:
    target = np.array([1.0, 1.0])
    angles = np.deg2rad([90.0, 210.0, 330.0])
    hunters = target + 0.2 * np.column_stack((np.cos(angles), np.sin(angles)))
    return CorrectedRoundupEnv(
        seed=3,
        num_obstacles=0,
        stationary_target=True,
        fixed_hunter_positions=hunters,
        fixed_target_position=target,
    )


def test_all_three_hunters_receive_terminal_reward() -> None:
    env = capture_env()
    env.reset()
    _, rewards, terminated, truncated, info = env.step(np.zeros((3, 2)))
    assert terminated and not truncated and info["success"]
    assert rewards.shape == (3,)
    assert np.all(rewards > 45.0)


def test_per_agent_speed_limit_is_independent() -> None:
    env = CorrectedRoundupEnv(seed=4, num_obstacles=0, stationary_target=True)
    env.reset()
    env.velocities[:3] = np.array([[0.2, 0.0], [0.02, 0.0], [0.0, 0.0]])
    env.step(np.zeros((3, 2)))
    speeds = np.linalg.norm(env.velocities[:3], axis=1)
    assert np.isclose(speeds[0], env.spec.hunter_v_max)
    assert np.isclose(speeds[1], 0.02)
    assert np.isclose(speeds[2], 0.0)


def test_declared_action_equals_executed_action_elementwise() -> None:
    env = CorrectedRoundupEnv(seed=5, num_obstacles=0, stationary_target=True)
    env.reset()
    action = np.array([[0.04, -0.04], [0.01, -0.02], [-0.03, 0.035]])
    _, _, _, _, info = env.step(action)
    assert np.array_equal(np.asarray(info["executed_actions"]), action)
    for idx in range(3):
        assert env.action_space[idx].contains(action[idx].astype(np.float32))


def test_out_of_contract_action_is_clipped_and_reported() -> None:
    env = CorrectedRoundupEnv(seed=6, num_obstacles=0, stationary_target=True)
    env.reset()
    action = np.full((3, 2), 0.4)
    _, _, _, _, info = env.step(action)
    assert np.allclose(info["executed_actions"], 0.04)
    assert np.isclose(info["action_saturation_fraction"], 1.0)


def test_scripted_target_flees_and_wall_repels() -> None:
    env = CorrectedRoundupEnv(
        seed=7,
        num_obstacles=0,
        fixed_hunter_positions=[[0.5, 1.0], [0.2, 0.2], [1.8, 0.2]],
        fixed_target_position=[1.0, 1.0],
    )
    env.reset()
    action = env.scripted_target_action()
    assert action[0] > 0.0
    assert np.isclose(np.linalg.norm(action), env.spec.target_a_max)

    wall_env = CorrectedRoundupEnv(
        seed=8,
        num_obstacles=0,
        fixed_hunter_positions=[[0.5, 1.0], [1.0, 1.5], [1.0, 0.5]],
        fixed_target_position=[0.05, 1.0],
    )
    wall_env.reset()
    assert wall_env.scripted_target_action()[0] > 0.0


def test_fixed_seed_reset_and_rollout_reproduce() -> None:
    first = CorrectedRoundupEnv(seed=99)
    second = CorrectedRoundupEnv(seed=99)
    obs_a, _ = first.reset()
    obs_b, _ = second.reset()
    assert np.array_equal(first.obstacle_positions, second.obstacle_positions)
    assert np.array_equal(first.obstacle_radii, second.obstacle_radii)
    assert np.array_equal(obs_a, obs_b)
    actions = np.array([[0.01, 0.0], [0.0, 0.02], [-0.01, 0.01]])
    for _ in range(5):
        out_a = first.step(actions)
        out_b = second.step(actions)
        assert np.array_equal(out_a[0], out_b[0])
        assert np.array_equal(out_a[1], out_b[1])
        assert out_a[2:4] == out_b[2:4]


def test_capture_terminates_and_timeout_only_truncates() -> None:
    cap = capture_env()
    cap.reset()
    _, _, terminated, truncated, _ = cap.step(np.zeros((3, 2)))
    assert terminated and not truncated

    timeout = CorrectedRoundupEnv(
        seed=10,
        spec=replace(RoundupSpec(), max_steps=1),
        num_obstacles=0,
        stationary_target=True,
    )
    timeout.reset()
    _, _, terminated, truncated, _ = timeout.step(np.zeros((3, 2)))
    assert not terminated and truncated


def test_environment_state_roundtrip_is_exact() -> None:
    env = CorrectedRoundupEnv(seed=11)
    env.reset()
    env.step(np.full((3, 2), 0.01))
    state = env.state_dict()
    expected = env.step(np.full((3, 2), -0.015))
    restored = CorrectedRoundupEnv(seed=11)
    restored.load_state_dict(state)
    actual = restored.step(np.full((3, 2), -0.015))
    assert np.array_equal(expected[0], actual[0])
    assert np.array_equal(expected[1], actual[1])
    assert expected[2:4] == actual[2:4]
    assert expected[4]["executed_actions"] == actual[4]["executed_actions"]


def test_moving_target_oracle_is_stable_without_obstacles() -> None:
    results = [run_oracle_episode(CorrectedRoundupEnv(seed=seed, num_obstacles=0)) for seed in range(10)]
    assert all(result["success"] for result in results)
    assert max(result["episode_length"] for result in results) < 100


def test_random_policy_does_not_fake_high_success() -> None:
    successes = 0
    for seed in range(20):
        env = CorrectedRoundupEnv(seed=1000 + seed, num_obstacles=0)
        env.reset()
        rng = np.random.default_rng(2000 + seed)
        for _ in range(env.spec.max_steps):
            _, _, terminated, truncated, _ = env.step(rng.uniform(-0.04, 0.04, size=(3, 2)))
            if terminated:
                successes += 1
                break
            if truncated:
                break
    assert successes <= 2
