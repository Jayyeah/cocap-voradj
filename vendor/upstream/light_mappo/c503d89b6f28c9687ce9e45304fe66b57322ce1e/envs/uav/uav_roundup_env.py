"""Adapter that exposes the inner UAVEnv to the MAPPO training framework.

Only the 3 hunter agents are trained; the target follows a hand-coded
"flee from the nearest hunter" policy so the rendered video shows a
purely learned encirclement behaviour.
"""
from __future__ import annotations

import numpy as np
from gym import spaces

from envs.uav.uav_env import UAVEnv


class UAVRoundupEnv:
    """Wrap UAVEnv into the (list-of-per-agent) interface used by DummyVecEnv."""

    NUM_HUNTERS = 3
    HUNTER_OBS_DIM = 26
    ACT_DIM = 2

    def __init__(self, length=2, num_obstacle=3, max_steps=100):
        self._env = UAVEnv(length=length, num_obstacle=num_obstacle, num_agents=4)
        self._env.MAX_STEPS = max_steps
        self.num_agent = self.NUM_HUNTERS

        # Per-hunter homogeneous spaces, so the shared-policy runner is happy.
        self.action_space = [
            spaces.Box(low=-self._env.a_max, high=self._env.a_max,
                       shape=(self.ACT_DIM,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf,
                       shape=(self.HUNTER_OBS_DIM,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]
        share_dim = self.HUNTER_OBS_DIM * self.num_agent
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(share_dim,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]

    def _scripted_target_action(self):
        """Flee from the nearest hunter while staying inside the arena.

        Force = (unit vector away from nearest hunter) + (wall-repulsion)
        Wall repulsion grows as the target approaches a boundary; it
        keeps the scripted target from running blindly into the wall.
        """
        target_pos = self._env.multi_current_pos[-1]
        hunter_positions = self._env.multi_current_pos[:-1]
        nearest = min(hunter_positions, key=lambda p: np.linalg.norm(p - target_pos))
        flee = target_pos - nearest
        norm = np.linalg.norm(flee)
        if norm < 1e-6:
            flee = np.random.uniform(-1, 1, size=2)
            norm = np.linalg.norm(flee) + 1e-6
        flee = flee / norm

        # Wall repulsion: kicks in within `margin` of any wall, capped at 1.
        L = self._env.length
        margin = 0.25
        wall_force = np.array([
            max(0.0, 1.0 - target_pos[0] / margin) - max(0.0, 1.0 - (L - target_pos[0]) / margin),
            max(0.0, 1.0 - target_pos[1] / margin) - max(0.0, 1.0 - (L - target_pos[1]) / margin),
        ])

        combined = flee + 2.0 * wall_force
        cnorm = np.linalg.norm(combined)
        if cnorm > 1e-6:
            combined = combined / cnorm
        return combined * self._env.a_max_e

    def reset(self):
        obs = self._env.reset()
        return self._hunter_obs(obs)

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        # actions shape: (num_hunters, 2). Append target's scripted action.
        target_act = self._scripted_target_action()
        full_actions = np.vstack([actions, target_act[None, :]])

        next_obs, rewards, dones = self._env.step(full_actions)

        hunter_obs = self._hunter_obs(next_obs)
        hunter_rew = np.asarray(rewards[:self.NUM_HUNTERS], dtype=np.float32).reshape(self.NUM_HUNTERS, 1)
        hunter_done = np.asarray(dones[:self.NUM_HUNTERS], dtype=bool)
        infos = [{} for _ in range(self.NUM_HUNTERS)]
        return hunter_obs, hunter_rew, hunter_done, infos

    def _hunter_obs(self, full_obs):
        return np.stack([np.asarray(full_obs[i], dtype=np.float32) for i in range(self.NUM_HUNTERS)])

    def render(self, mode="rgb_array"):
        return self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed):
        np.random.seed(seed)
