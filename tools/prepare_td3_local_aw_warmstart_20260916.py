#!/usr/bin/env python3
"""Collect matched successful IQN trajectories and fit TD3 actor heads."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch
import yaml

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LegacyVorAdjFeatureBackbone,
)
from cocap_voradj.models.continuous.local_td3 import LocalTD3Actor
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.td3_stage1_contract import (
    aw9_grid,
    check_td3_env,
    discrete_task_config,
    make_td3_env,
)
from cocap_voradj.training.td3_warmstart import (
    DATASET_SCHEMA,
    OBS_KEYS,
    WARMSTART_SCHEMA,
    backbone_config_from_iqn,
    behavior_clone_head,
    file_sha256,
    load_iqn_teacher,
    tensor_state_sha256,
    transfer_iqn_backbone,
)
from cocap_voradj.training.trainer import load_config, set_global_config
from tools import preflight_forward_final_scratch_20260914 as preflight
from tools.forward_final_single_task_20260915 import Telemetry, summarize


CONFIG = ROOT / "configs/experiments/td3_local_aw_stage1_20260916/common.yaml"


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _evader_actions(env: VorAdjEnv, agents: list[Any]) -> list[int | None]:
    if not env.evaders:
        return []
    env.configure_evader_apf_agents(agents)
    observations = env.get_evader_observations_for_apf()
    return [
        None if observation is None else int(agents[index].act(observation))
        for index, observation in enumerate(observations)
    ]


def _episode_success(task: str, row: Mapping[str, Any]) -> bool:
    # Capture teacher data is restricted to ordinary cooperative capture;
    # stationary fallback never enters the normal-capture label.
    return bool(row["ce_success"] if task == "coverage" else row["normal_capture"])


def collect_successful_teacher_dataset(
    *,
    teacher,
    teacher_metadata: Mapping[str, Any],
    task: str,
    device: str,
    train_successes: int,
    heldout_successes: int,
    seed_base: int,
    output: Path,
    max_attempts: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    target_successes = int(train_successes) + int(heldout_successes)
    episode_payloads: list[dict[str, Any]] = []
    episode_metrics: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    teacher.eval()
    for attempt in range(int(max_attempts)):
        if len(episode_payloads) >= target_successes:
            break
        seed = int(seed_base) + attempt
        cfg = discrete_task_config(task)
        set_global_config(cfg)
        env = VorAdjEnv(copy.deepcopy(cfg), seed=seed)
        observations = env.reset()
        env.total_steps = 2_000_000
        telemetry = Telemetry(env, task)
        apf_agents = [preflight.ApfAgent(evader.a, evader.w) for evader in env.evaders]
        rows = {key: [] for key in OBS_KEYS}
        indices: list[int] = []
        for _ in range(env.episode_max_length):
            active = [index for index, observation in enumerate(observations) if observation is not None]
            batch = stack_obs([observations[index] for index in active], device)
            with torch.no_grad():
                selected = teacher.act(
                    batch, mode="voradj", epsilon=0.0, deterministic_quantiles=True
                ).cpu().numpy()
            commands: list[int | None] = [None] * len(observations)
            for agent_id, action_index in zip(active, selected):
                commands[agent_id] = int(action_index)
                for key in OBS_KEYS:
                    rows[key].append(np.asarray(observations[agent_id][key]).copy())
                indices.append(int(action_index))
            outcome = env.step(commands, _evader_actions(env, apf_agents))
            telemetry.observe(env, outcome, active)
            observations = outcome.observations
            if all(outcome.dones):
                break
        metric = dict(seed=seed, attempt=attempt, **telemetry.finish(env))
        success = _episode_success(task, metric)
        attempts.append(
            {
                "seed": seed,
                "success": success,
                "length": metric["length"],
                "normal_capture": metric["normal_capture"],
                "stationary_capture": metric["stationary_capture"],
                "ce_success": metric["ce_success"],
                "collision": metric["collision"],
            }
        )
        print(json.dumps(attempts[-1], sort_keys=True), flush=True)
        if success:
            episode_payloads.append(
                {
                    "obs": {key: np.stack(rows[key]) for key in OBS_KEYS},
                    "action_index": np.asarray(indices, dtype=np.int64),
                    "seed": seed,
                }
            )
            episode_metrics.append(metric)
    if len(episode_payloads) != target_successes:
        raise RuntimeError(
            f"qualified {len(episode_payloads)}/{target_successes} successful {task} episodes "
            f"after {len(attempts)} attempts"
        )

    arrays: dict[str, np.ndarray] = {
        f"obs_{key}": np.concatenate([episode["obs"][key] for episode in episode_payloads])
        for key in OBS_KEYS
    }
    arrays["action_index"] = np.concatenate(
        [episode["action_index"] for episode in episode_payloads]
    ).astype(np.int64)
    arrays["action_aw"] = aw9_grid()[arrays["action_index"]].astype(np.float32)
    episode_ids = np.concatenate(
        [np.full(len(episode["action_index"]), index, dtype=np.int64) for index, episode in enumerate(episode_payloads)]
    )
    arrays["episode_id"] = episode_ids
    arrays["episode_seed"] = np.concatenate(
        [np.full(len(episode["action_index"]), episode["seed"], dtype=np.int64) for episode in episode_payloads]
    )
    arrays["split"] = (episode_ids >= int(train_successes)).astype(np.uint8)
    _atomic_npz(output, arrays)
    manifest = {
        "schema": DATASET_SCHEMA,
        "task": task,
        "contract": "current NormSense-V2 pure single-task + discrete AW9 teacher",
        "teacher": dict(teacher_metadata),
        "dataset": str(output.resolve()),
        "dataset_sha256": file_sha256(output),
        "rows": int(len(arrays["action_index"])),
        "successful_episodes": target_successes,
        "train_episodes": int(train_successes),
        "heldout_episodes": int(heldout_successes),
        "train_rows": int(np.count_nonzero(arrays["split"] == 0)),
        "heldout_rows": int(np.count_nonzero(arrays["split"] == 1)),
        "episode_level_split": True,
        "normal_capture_only": task == "capture",
        "attempts": attempts,
        "successful_episode_summary": summarize(episode_metrics),
        "action_index_histogram": np.bincount(arrays["action_index"], minlength=9).tolist(),
        "action_index_to_aw": aw9_grid().tolist(),
        "config_sha256": _canonical_hash(discrete_task_config(task)),
    }
    _atomic_json(output.with_suffix(".manifest.json"), manifest)
    return arrays, manifest


@torch.no_grad()
def evaluate_continuous_actor(
    actor: LocalTD3Actor,
    *,
    task: str,
    device: str,
    seed_base: int,
    episodes: int = 8,
    max_steps: int = 600,
) -> dict[str, Any]:
    actor.eval()
    records = []
    for episode in range(int(episodes)):
        seed = int(seed_base) + episode
        env, observations = make_td3_env(task, seed)
        check_td3_env(env, task)
        env.total_steps = 2_000_000
        telemetry = Telemetry(env, task)
        apf_agents = [preflight.ApfAgent(evader.a, evader.w) for evader in env.evaders]
        for _ in range(min(int(max_steps), env.episode_max_length)):
            active = [index for index, observation in enumerate(observations) if observation is not None]
            batch = stack_obs([observations[index] for index in active], device)
            selected = actor(batch).cpu().numpy()
            commands: list[Any] = [None] * len(observations)
            for agent_id, action in zip(active, selected):
                commands[agent_id] = action
            outcome = env.step(commands, _evader_actions(env, apf_agents))
            telemetry.observe(env, outcome, active)
            observations = outcome.observations
            if all(outcome.dones):
                break
        row = dict(seed=seed, native_done=all(outcome.dones), **telemetry.finish(env))
        records.append(row)
    return {
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "seed_base": int(seed_base),
        "summary": summarize(records),
        "records": records,
    }


def prepare_task(args: argparse.Namespace, config: Mapping[str, Any], task: str) -> None:
    checkpoint = ROOT / config["teacher"]["checkpoint"]
    teacher, teacher_metadata = load_iqn_teacher(checkpoint, args.device)
    if teacher_metadata["checkpoint_sha256"] != config["teacher"]["checkpoint_sha256"]:
        raise ValueError("IQN teacher checkpoint hash mismatch")
    output_dir = Path(args.output) / task
    dataset_path = output_dir / "teacher_success.npz"
    if dataset_path.exists() and not args.force_collect:
        with np.load(dataset_path, allow_pickle=False) as loaded:
            dataset = {key: loaded[key] for key in loaded.files}
        manifest = json.loads(dataset_path.with_suffix(".manifest.json").read_text())
        if manifest["dataset_sha256"] != file_sha256(dataset_path):
            raise ValueError("existing teacher dataset hash mismatch")
    else:
        dataset, manifest = collect_successful_teacher_dataset(
            teacher=teacher,
            teacher_metadata=teacher_metadata,
            task=task,
            device=args.device,
            train_successes=int(config["teacher"]["successful_train_episodes"]),
            heldout_successes=int(config["teacher"]["successful_heldout_episodes"]),
            seed_base=int(config["teacher"]["collection_seed_base"]) + (10_000 if task == "capture" else 0),
            output=dataset_path,
            max_attempts=args.max_attempts,
        )

    torch.manual_seed(int(config["teacher"]["bc_seed"]) + (1 if task == "capture" else 0))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["teacher"]["bc_seed"]) + (1 if task == "capture" else 0))
    actor = LocalTD3Actor(
        LegacyVorAdjFeatureBackbone(backbone_config_from_iqn(teacher.config))
    ).to(args.device)
    transfer = transfer_iqn_backbone(actor, teacher)
    rollout_seed = int(config["evaluation"][f"{task}_seed_base"])

    def rollout_callback(model: LocalTD3Actor, epoch: int) -> Mapping[str, Any]:
        print(f"{task}: deterministic BC rollout epoch={epoch}", flush=True)
        return evaluate_continuous_actor(
            model,
            task=task,
            device=args.device,
            seed_base=rollout_seed,
            episodes=args.rollout_episodes,
            max_steps=args.rollout_max_steps,
        )

    report = behavior_clone_head(
        actor,
        dataset,
        device=args.device,
        epochs=int(config["teacher"]["bc_epochs"]),
        batch_size=int(config["teacher"]["bc_batch_size"]),
        learning_rate=float(config["teacher"]["bc_learning_rate"]),
        seed=int(config["teacher"]["bc_seed"]) + (1 if task == "capture" else 0),
        rollout_callback=rollout_callback,
    )
    payload = {
        "schema": WARMSTART_SCHEMA,
        "task": task,
        "actor_state_dict": actor.state_dict(),
        "actor_state_sha256": tensor_state_sha256(actor.state_dict()),
        "backbone_config": vars(actor.backbone.config),
        "teacher": teacher_metadata,
        "teacher_dataset_manifest": manifest,
        "transfer": transfer,
        "behavior_cloning": report,
        "critics_initialized": False,
        "critic_transfer": "none",
    }
    actor_path = output_dir / "actor_bc.pt"
    _atomic_torch(actor_path, payload)
    summary = {key: value for key, value in payload.items() if key != "actor_state_dict"}
    summary["actor_checkpoint"] = str(actor_path.resolve())
    summary["actor_checkpoint_sha256"] = file_sha256(actor_path)
    _atomic_json(output_dir / "warmstart_report.json", summary)
    print(json.dumps({"task": task, "actor": str(actor_path), "final": report["final"]}, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("coverage", "capture", "all"), default="all")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts/2026-09-16_td3_local_aw_stage1/warmstart"),
    )
    parser.add_argument("--max-attempts", type=int, default=120)
    parser.add_argument("--rollout-episodes", type=int, default=8)
    parser.add_argument("--rollout-max-steps", type=int, default=600)
    parser.add_argument("--force-collect", action="store_true")
    args = parser.parse_args()
    config = load_config(str(CONFIG))
    if config.get("schema") != "td3-local-aw-stage1-v1":
        raise ValueError("TD3 Stage-1 config schema mismatch")
    for task in (("coverage", "capture") if args.task == "all" else (args.task,)):
        prepare_task(args, config, task)


if __name__ == "__main__":
    main()
