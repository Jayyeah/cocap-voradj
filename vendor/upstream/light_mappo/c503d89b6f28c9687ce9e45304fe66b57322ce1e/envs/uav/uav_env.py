"""UAVEnv — multi-agent UAV round-up environment.

3 hunter agents try to encircle and capture 1 evading target, in a 2-D
arena with randomly-placed obstacles. Each agent has 16 lidar lasers.
Action space is per-agent acceleration `[a_x, a_y]`; observation is a
flat per-agent feature vector mixing self state, teammate positions,
target bearing, and lidar readings.

Adapted from `marl/src/envs/uav_env.py` (originally from a MAPPO
training repo). Two adjustments for use as a benchmark env:

  - the optional `UAV.png` icon used by `render()` is detected at
    runtime; if absent we fall back to plain triangle markers, so the
    env runs anywhere without an asset bundle.
  - imports rewritten to live alongside the other `envs/*.py` modules.

NOTE: this is a non-standard multi-agent env. `action_space` and
`observation_space` are dicts keyed by agent name (`agent_0` ..
`agent_2`, `target`), not single `gymnasium.spaces.Space` objects.
"""
from __future__ import annotations

import copy
import itertools
import os
import random

import matplotlib.backends.backend_agg as agg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from gymnasium import spaces

from envs.uav.uav_utils import cal_triangle_S, update_lasers


_UAV_ICON_PATH = os.path.join(os.path.dirname(__file__), "UAV.png")


def _load_uav_icon():
    if os.path.exists(_UAV_ICON_PATH):
        try:
            return mpimg.imread(_UAV_ICON_PATH)
        except Exception:
            return None
    return None


class UAVEnv:
    def __init__(self, length=2, num_obstacle=3, num_agents=4):
        self.length = length
        self.num_obstacle = num_obstacle
        self.num_agents = num_agents
        self.time_step = 0.5
        self.v_max = 0.1
        self.v_max_e = 0.12
        self.a_max = 0.04
        self.a_max_e = 0.05
        self.L_sensor = 0.2
        self.num_lasers = 16
        self.multi_current_lasers = [
            [self.L_sensor for _ in range(self.num_lasers)]
            for _ in range(self.num_agents)
        ]
        self.agents = ["agent_0", "agent_1", "agent_2", "target"]
        self.info = np.random.get_state()
        self.obstacles = [Obstacle(length=length) for _ in range(self.num_obstacle)]
        self.history_positions = [[] for _ in range(num_agents)]

        self.action_space = {
            "agent_0": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            "agent_1": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            "agent_2": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            "target": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
        }
        self.observation_space = {
            "agent_0": spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            "agent_1": spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            "agent_2": spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            "target": spaces.Box(low=-np.inf, high=np.inf, shape=(23,)),
        }
        self.step_cnt = 0
        self.MAX_STEPS = 100

    def reset(self):
        self.step_cnt = 0
        random.seed(random.randint(1, 1000))
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            if i != self.num_agents - 1:
                self.multi_current_pos.append(np.random.uniform(low=0.1, high=0.4, size=(2,)))
            else:
                self.multi_current_pos.append(np.array([0.5, 1.75]))
            self.multi_current_vel.append(np.zeros(2))
        self.update_lasers_isCollied_wrapper()
        return self.get_multi_obs()

    def step(self, actions):
        last_d2target = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            if i != self.num_agents - 1:
                pos_target = self.multi_current_pos[-1]
                last_d2target.append(np.linalg.norm(pos - pos_target))
            self.multi_current_vel[i][0] += actions[i][0] * self.time_step
            self.multi_current_vel[i][1] += actions[i][1] * self.time_step
            vel_magnitude = np.linalg.norm(self.multi_current_vel)
            if i != self.num_agents - 1:
                if vel_magnitude >= self.v_max:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max
            else:
                if vel_magnitude >= self.v_max_e:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max_e
            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step

        for obs in self.obstacles:
            obs.position += obs.velocity * self.time_step
            for dim in (0, 1):
                if obs.position[dim] - obs.radius < 0:
                    obs.position[dim] = obs.radius
                    obs.velocity[dim] *= -1
                elif obs.position[dim] + obs.radius > self.length:
                    obs.position[dim] = self.length - obs.radius
                    obs.velocity[dim] *= -1

        Collided = self.update_lasers_isCollied_wrapper()
        rewards, dones = self.cal_rewards_dones(Collided, last_d2target)
        multi_next_obs = self.get_multi_obs()

        if self.step_cnt >= self.MAX_STEPS:
            dones = [True] * self.num_agents
        self.step_cnt += 1
        return multi_next_obs, rewards, dones

    def get_multi_obs(self):
        total_obs = []
        S_evade_d = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [pos[0] / self.length, pos[1] / self.length, vel[0] / self.v_max, vel[1] / self.v_max]
            S_team = []
            S_target = []
            for j in range(self.num_agents):
                if j != i and j != self.num_agents - 1:
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0] / self.length, pos_other[1] / self.length])
                elif j == self.num_agents - 1:
                    pos_target = self.multi_current_pos[j]
                    d = np.linalg.norm(pos - pos_target)
                    theta = np.arctan2(pos_target[1] - pos[1], pos_target[0] - pos[0])
                    S_target.extend([d / np.linalg.norm(2 * self.length), theta])
                    if i != self.num_agents - 1:
                        S_evade_d.append(d / np.linalg.norm(2 * self.length))
            S_obser = self.multi_current_lasers[i]
            if i != self.num_agents - 1:
                single_obs = [S_uavi, S_team, S_obser, S_target]
            else:
                single_obs = [S_uavi, S_obser, S_evade_d]
            total_obs.append(list(itertools.chain(*single_obs)))
        return total_obs

    def cal_rewards_dones(self, IsCollied, last_d):
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)
        mu1, mu2, mu3, mu4 = 0.7, 0.4, 0.01, 5
        d_capture = 0.3
        d_limit = 0.75

        for i in range(3):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            pos_target = self.multi_current_pos[-1]
            v_i = np.linalg.norm(vel)
            dire_vec = pos_target - pos
            d = np.linalg.norm(dire_vec)
            cos_v_d = np.dot(vel, dire_vec) / (v_i * d + 1e-3)
            r_near = abs(2 * v_i / self.v_max) * cos_v_d
            rewards[i] += mu1 * r_near

        for i in range(self.num_agents):
            if IsCollied[i]:
                r_safe = -10
            else:
                lasers = self.multi_current_lasers[i]
                r_safe = (min(lasers) - self.L_sensor - 0.1) / self.L_sensor
            rewards[i] += mu2 * r_safe

        p0, p1, p2, pe = self.multi_current_pos[0], self.multi_current_pos[1], self.multi_current_pos[2], self.multi_current_pos[-1]
        S1 = cal_triangle_S(p0, p1, pe)
        S2 = cal_triangle_S(p1, p2, pe)
        S3 = cal_triangle_S(p2, p0, pe)
        S4 = cal_triangle_S(p0, p1, p2)
        d1 = np.linalg.norm(p0 - pe)
        d2 = np.linalg.norm(p1 - pe)
        d3 = np.linalg.norm(p2 - pe)
        Sum_S = S1 + S2 + S3
        Sum_d = d1 + d2 + d3
        Sum_last_d = sum(last_d)
        rewards[-1] += np.clip(10 * (Sum_d - Sum_last_d), -2, 2)
        if Sum_S > S4 and Sum_d >= d_limit and all(d >= d_capture for d in [d1, d2, d3]):
            r_track = -Sum_d / max([d1, d2, d3])
            rewards[0:2] += mu3 * r_track
        elif Sum_S > S4 and (Sum_d < d_limit or any(d >= d_capture for d in [d1, d2, d3])):
            r_encircle = -1 / 3 * np.log(Sum_S - S4 + 1)
            rewards[0:2] += mu3 * r_encircle
        elif Sum_S == S4 and any(d > d_capture for d in [d1, d2, d3]):
            r_capture = np.exp((Sum_last_d - Sum_d) / (3 * self.v_max))
            rewards[0:2] += mu3 * r_capture

        if Sum_S == S4 and all(d <= d_capture for d in [d1, d2, d3]):
            rewards[0:2] += mu4 * 10
            dones = [True] * self.num_agents
        return rewards, dones

    def update_lasers_isCollied_wrapper(self):
        self.multi_current_lasers = []
        dones = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []
            for obs in self.obstacles:
                _current_lasers, done = update_lasers(
                    pos, obs.position, obs.radius, self.L_sensor, self.num_lasers, self.length
                )
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)
            done = any(done_obs)
            if done:
                self.multi_current_vel[i] = np.zeros(2)
            self.multi_current_lasers.append(current_lasers)
            dones.append(done)
        return dones

    # Per-hunter color so the three agents are visually distinct in the video.
    _HUNTER_COLORS = ("#1f77b4", "#2ca02c", "#9467bd")
    _TARGET_COLOR = "#d62728"
    _BG_COLOR = "#f7f7f9"
    _OBSTACLE_COLOR = "#5a5f66"

    def render(self):
        fig = plt.gcf()
        fig.set_size_inches(6, 6)
        fig.set_facecolor(self._BG_COLOR)
        plt.clf()
        ax = plt.gca()
        ax.set_facecolor(self._BG_COLOR)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.1, self.length + 0.1)
        ax.set_ylim(-0.1, self.length + 0.1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Arena boundary — kept subtle so it doesn't compete with the agents.
        ax.add_patch(plt.Rectangle(
            (0, 0), self.length, self.length,
            fill=False, edgecolor="#cdd0d4", linewidth=0.7,
            linestyle=(0, (3, 3)),
        ))

        # Capture zone around target (visualises the win condition radius)
        d_capture = 0.3
        target_pos = self.multi_current_pos[-1]
        ax.add_patch(plt.Circle(
            target_pos, d_capture,
            fill=False, edgecolor=self._TARGET_COLOR, linewidth=1.0,
            linestyle=":", alpha=0.6,
        ))

        # Obstacles
        for obstacle in self.obstacles:
            ax.add_patch(plt.Circle(
                obstacle.position, obstacle.radius,
                facecolor=self._OBSTACLE_COLOR, edgecolor="#2c3036",
                alpha=0.55, linewidth=1.0, zorder=2,
            ))

        uav_icon = _load_uav_icon()

        # Hunters
        for i in range(self.num_agents - 1):
            color = self._HUNTER_COLORS[i % len(self._HUNTER_COLORS)]
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])

            # Fading trajectory: older points more transparent
            if len(trajectory) >= 2:
                segs = np.stack([trajectory[:-1], trajectory[1:]], axis=1)
                n_segs = len(segs)
                for k, seg in enumerate(segs):
                    alpha = 0.15 + 0.6 * (k / max(n_segs - 1, 1))
                    ax.plot(seg[:, 0], seg[:, 1], color=color, alpha=alpha, linewidth=1.8, zorder=3)

            angle = np.arctan2(vel[1], vel[0])
            if uav_icon is not None:
                t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
                icon_size = 0.1
                ax.imshow(
                    uav_icon,
                    transform=t + ax.transData,
                    extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2),
                    zorder=5,
                )
            else:
                # Body (always label so legend stays put every frame)
                ax.scatter(pos[0], pos[1], c=color, s=140, edgecolors="white",
                           linewidths=1.5, zorder=5, label=f"Hunter {i}")
                # Heading arrow (only when velocity is non-trivial)
                speed = np.linalg.norm(vel)
                if speed > 1e-3:
                    arrow_len = 0.08
                    ax.annotate(
                        "", xy=(pos[0] + arrow_len * np.cos(angle), pos[1] + arrow_len * np.sin(angle)),
                        xytext=(pos[0], pos[1]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.8, shrinkA=4, shrinkB=0),
                        zorder=6,
                    )

        # Target trajectory + marker
        self.history_positions[-1].append(copy.deepcopy(target_pos))
        target_traj = np.array(self.history_positions[-1])
        if len(target_traj) >= 2:
            segs = np.stack([target_traj[:-1], target_traj[1:]], axis=1)
            n_segs = len(segs)
            for k, seg in enumerate(segs):
                alpha = 0.15 + 0.6 * (k / max(n_segs - 1, 1))
                ax.plot(seg[:, 0], seg[:, 1], color=self._TARGET_COLOR,
                        alpha=alpha, linewidth=1.8, zorder=3)
        ax.scatter(target_pos[0], target_pos[1], c=self._TARGET_COLOR,
                   marker="*", s=240, edgecolors="white", linewidths=1.5,
                   zorder=6, label="Target")

        # Step counter — pinned top-left, compact.
        ax.text(
            0.02, 0.98, f"step {self.step_cnt:3d}/{self.MAX_STEPS}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="#2c3036",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="#d0d3d8", alpha=0.85),
        )

        # Legend lifted above the plot so a video player's bottom scrubber
        # never overlaps it. Single horizontal row.
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.01),
            ncol=4, frameon=False, fontsize=9,
            handletextpad=0.4, columnspacing=1.0,
        )

        fig.tight_layout(pad=0.5)

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        return np.asarray(buf)

    def close(self):
        plt.close()


class Obstacle:
    def __init__(self, length=2):
        self.position = np.random.uniform(low=0.45, high=length - 0.55, size=(2,))
        angle = np.random.uniform(0, 2 * np.pi)
        speed = 0.0  # static obstacles
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        self.radius = np.random.uniform(0.1, 0.15)
