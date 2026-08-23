"""Load a trained UAV roundup actor and render an inference episode to mp4/gif.

Example:
    python scripts/render_uav.py \
        --model_dir results/uav/MyEnv/mappo/uav_demo/run1/models \
        --episode_length 100 \
        --video_path videos/uav_demo.mp4

The script loads the shared actor checkpoint produced by training with
``--env_name uav --share_policy`` and rolls out a single deterministic
episode, writing a video of `UAVEnv.render()` frames.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

sys.path.append(os.getcwd())

from config import get_config
from envs.uav.uav_roundup_env import UAVRoundupEnv


def parse_args(argv):
    parser = get_config()  # already declares --model_dir
    parser.add_argument("--video_path", type=str, default="videos/uav.mp4",
                        help="Output video path; suffix decides format (.mp4 / .gif)")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--render_deterministic", action="store_true", default=True)
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--scenario_name", type=str, default="MyEnv")
    parser.add_argument("--num_landmarks", type=int, default=3)
    return parser.parse_known_args(argv)[0]


@torch.no_grad()
def rollout_one_episode(env, policy, all_args, deterministic):
    obs = env.reset()
    rnn_states = np.zeros(
        (all_args.num_agents, all_args.recurrent_N, all_args.hidden_size),
        dtype=np.float32,
    )
    masks = np.ones((all_args.num_agents, 1), dtype=np.float32)

    frames = [env.render()]
    total_reward = np.zeros(all_args.num_agents, dtype=np.float32)
    steps = 0
    while True:
        actions, rnn_states = policy.act(
            obs,
            rnn_states,
            masks,
            deterministic=deterministic,
        )
        actions = actions.detach().cpu().numpy()
        rnn_states = rnn_states.detach().cpu().numpy()

        obs, rewards, dones, _ = env.step(actions)
        frames.append(env.render())
        total_reward += rewards.squeeze(-1)
        steps += 1
        if np.all(dones) or steps >= all_args.episode_length:
            break
    return frames, total_reward, steps


def main(argv):
    all_args = parse_args(argv)
    # Match training-time defaults for the UAV env.
    if all_args.env_name not in ("uav",):
        all_args.env_name = "uav"

    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")

    env = UAVRoundupEnv(max_steps=all_args.episode_length)

    # Build the policy with the same shapes the trainer used.
    from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy

    share_obs_space = env.share_observation_space[0] if all_args.use_centralized_V else env.observation_space[0]
    policy = RMAPPOPolicy(
        all_args,
        env.observation_space[0],
        share_obs_space,
        env.action_space[0],
        device=device,
    )

    actor_path = Path(all_args.model_dir) / "actor.pt"
    state_dict = torch.load(str(actor_path), map_location=device)
    policy.actor.load_state_dict(state_dict)
    policy.actor.eval()
    print(f"Loaded actor weights from {actor_path}")

    all_frames = []
    for ep in range(all_args.num_episodes):
        frames, ep_reward, steps = rollout_one_episode(env, policy, all_args, all_args.render_deterministic)
        print(f"  episode {ep + 1}/{all_args.num_episodes}: {steps} steps, reward per hunter = {ep_reward}")
        all_frames.extend(frames)
    env.close()

    video_path = Path(all_args.video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # imageio expects 3-channel uint8. UAVEnv.render() returns RGBA — drop alpha.
    rgb_frames = [np.asarray(f, dtype=np.uint8)[..., :3] for f in all_frames]
    if video_path.suffix.lower() == ".gif":
        imageio.mimsave(str(video_path), rgb_frames, fps=all_args.fps)
    else:
        with imageio.get_writer(str(video_path), fps=all_args.fps, macro_block_size=1) as writer:
            for frame in rgb_frames:
                writer.append_data(frame)
    print(f"Wrote {len(rgb_frames)} frames to {video_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
