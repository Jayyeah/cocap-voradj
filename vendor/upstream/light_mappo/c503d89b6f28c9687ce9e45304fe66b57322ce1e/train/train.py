"""
# @Time    : 2021/6/30 10:07 下午
# @Author  : hezhiqiang
# @Email   : tinyzqh@163.com
# @File    : train.py
"""

#!/usr/bin/env python
import os
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import setproctitle
except ImportError:  # optional: only used to set the process title
    setproctitle = None

# Append the project root to sys.path so the imports below resolve when running directly.
sys.path.append(os.getcwd())

from config import get_config
from envs.env_wrappers import DummyVecEnv

"""Train script for MPEs."""


def _build_single_env(all_args):
    """Pick the inner env based on --env_name."""
    if all_args.env_name == "uav":
        from envs.uav.uav_roundup_env import UAVRoundupEnv

        return UAVRoundupEnv(max_steps=all_args.episode_length)
    # TODO 选择连续/离散动作空间，注释/启用对应两行即可。
    # TODO Switch between continuous/discrete action spaces by toggling the two lines.
    from envs.custom_env.env_continuous import ContinuousActionEnv

    return ContinuousActionEnv()
    # from envs.custom_env.env_discrete import DiscreteActionEnv
    # return DiscreteActionEnv()


def make_env(all_args, n_threads):
    def get_env_fn(rank):
        def init_env():
            env = _build_single_env(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(n_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="MyEnv", help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=2, help="number of players")
    all_args = parser.parse_known_args(args)[0]
    if all_args.env_name == "uav":
        # UAV env trains 3 hunters (target is scripted); force num_agents accordingly.
        all_args.num_agents = 3
    return all_args


def setup_device(all_args):
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)
    return device


def prepare_run_dir(all_args):
    project_root = Path(__file__).resolve().parent.parent
    base = project_root / "results" / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    base.mkdir(parents=True, exist_ok=True)

    existing = [int(p.name[3:]) for p in base.iterdir() if p.name.startswith("run") and p.name[3:].isdigit()]
    curr_run = f"run{max(existing) + 1 if existing else 1}"
    run_dir = base / curr_run
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert not all_args.use_recurrent_policy and not all_args.use_naive_recurrent_policy, "check recurrent policy!"
    else:
        raise NotImplementedError

    assert not (all_args.share_policy and all_args.scenario_name == "simple_speaker_listener"), (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py."
    )

    device = setup_device(all_args)
    run_dir = prepare_run_dir(all_args)

    if setproctitle is not None:
        setproctitle.setproctitle(
            f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}"
        )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_env(all_args, all_args.n_rollout_threads)
    eval_envs = make_env(all_args, all_args.n_eval_rollout_threads) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    if all_args.share_policy:
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(Path(runner.log_dir) / "summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
