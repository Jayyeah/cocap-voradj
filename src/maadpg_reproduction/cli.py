"""Command line runner for matched MAADPG/MADDPG training and evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import torch

from .checkpoint import save_full_checkpoint, save_model_checkpoint
from .config import default_environment_config
from .evaluation import evaluate_actor_only
from .exploration import OUExplorationConfig
from .logging_utils import append_jsonl, atomic_write_json, json_safe, utc_now
from .maddpg import MADDPGLearner
from .trainer import MAADPGTrainer, TrainerConfig, TrainerStep
from .training_config import default_maddpg_config


PAPER_SHA256 = "956fb59036dcba020b75bda0bc12d75c50204aaf6cfe49bf53c1eaf48d28091f"


class GracefulStop(Exception):
    pass


def _git(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *command], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _provenance() -> dict[str, Any]:
    status = _git(["status", "--porcelain"])
    return {
        "created_at_utc": utc_now(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_dirty": bool(status and status != "unknown"),
        "paper_sha256": PAPER_SHA256,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def _small_test_config(variant: str, seed: int) -> TrainerConfig:
    environment = replace(default_environment_config(), horizon_steps=20)
    maddpg = replace(
        default_maddpg_config(),
        hidden_dims=(32, 32),
        replay_capacity=512,
        batch_size=8,
        replay_warmup=8,
    )
    return TrainerConfig(
        variant=variant,
        training_seed=seed,
        environment=environment,
        maddpg=maddpg,
        exploration=OUExplorationConfig(decay_steps=80),
    )


def _default_config(variant: str, seed: int) -> TrainerConfig:
    return TrainerConfig(variant=variant, training_seed=seed)


def _load_payload(path: Path, device: str) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def _load_trainer(path: Path, device: str) -> tuple[MAADPGTrainer, dict[str, Any]]:
    payload = _load_payload(path, device)
    trainer = MAADPGTrainer.from_state_dict(payload, device=device)
    return trainer, payload.get("metadata", {})


def _metadata(run_id: str, trainer: MAADPGTrainer, provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "variant": trainer.config.variant,
        "training_seed": trainer.config.training_seed,
        "environment_steps": trainer.runtime.environment_steps,
        "gradient_steps": trainer.runtime.gradient_steps,
        "episodes_completed": trainer.runtime.episodes_completed,
        "spec_version": trainer.config.environment.spec_version,
        "environment_config_sha256": trainer.config.environment.canonical_sha256(),
        **provenance,
    }


def _diagnostic_record(step: TrainerStep, trainer: MAADPGTrainer) -> dict[str, Any]:
    record: dict[str, Any] = {
        "event": "diagnostic",
        "time_utc": utc_now(),
        "variant": trainer.config.variant,
        "environment_step": step.environment_step,
        "gradient_step": step.gradient_step,
        "replay_size": step.replay_size,
        "episode_index": trainer.runtime.episodes_completed,
        "episode_length": trainer.runtime.current_episode_length,
        "reward_mean": float(np.mean(step.reward)),
        "action_norm_mean": float(np.mean(np.linalg.norm(step.executed_action, axis=1))),
        "action_norm_max": float(np.max(np.linalg.norm(step.executed_action, axis=1))),
        "exploration_sigma": trainer.exploration.sigma(step.environment_step),
        "gate_adoptions_total": trainer.runtime.gate_adoptions_total.copy(),
        "gate_comparisons_total": trainer.runtime.gate_comparisons_total,
    }
    if step.gate_decision is not None:
        record["gate_adopted"] = step.gate_decision.adopted_pfm
        record["gate_delta_ratio"] = step.gate_decision.delta_ratios
        record["gate_actor_reward"] = step.gate_decision.actor_rewards
        record["gate_pfm_reward"] = step.gate_decision.pfm_rewards
    if step.updates:
        record["learner_update"] = step.updates[-1]
    return record


def _install_signal_handlers() -> None:
    def stop(signum, frame):
        del frame
        raise GracefulStop(f"received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


def train_command(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "rolling-full.pt"
    run_id = args.run_id or run_dir.name
    provenance = _provenance()
    if args.resume:
        trainer, previous_metadata = _load_trainer(Path(args.resume), args.device)
        if trainer.config.variant != args.variant or trainer.config.training_seed != args.seed:
            raise ValueError("resume checkpoint variant/seed differs from command")
    else:
        if checkpoint_path.exists():
            raise FileExistsError(
                f"{checkpoint_path} already exists; use --resume or a new run directory"
            )
        config = (
            _small_test_config(args.variant, args.seed)
            if args.small_test
            else _default_config(args.variant, args.seed)
        )
        trainer = MAADPGTrainer(config, device=args.device)
        previous_metadata = {}
    manifest = {
        "run_id": run_id,
        "status": "running",
        "step_budget": args.step_budget,
        "device": args.device,
        "trainer_config": trainer.config,
        "provenance": provenance,
        "resume_metadata": previous_metadata,
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    if args.ledger:
        append_jsonl(
            args.ledger,
            {"event": "run_started", "time_utc": utc_now(), **_metadata(run_id, trainer, provenance)},
            sync=True,
        )
    _install_signal_handlers()
    latest_step: TrainerStep | None = None
    try:
        while trainer.runtime.environment_steps < args.step_budget:
            latest_step = trainer.step_once()
            if latest_step.episode is not None:
                append_jsonl(
                    run_dir / "episodes.jsonl",
                    {"event": "episode", "time_utc": utc_now(), **asdict(latest_step.episode)},
                )
            if latest_step.environment_step % args.diagnostic_every == 0:
                diagnostic = _diagnostic_record(latest_step, trainer)
                append_jsonl(run_dir / "diagnostics.jsonl", diagnostic)
                print(json.dumps(json_safe(diagnostic), sort_keys=True), flush=True)
            if latest_step.environment_step % args.checkpoint_every == 0:
                save_full_checkpoint(
                    trainer,
                    checkpoint_path,
                    metadata=_metadata(run_id, trainer, provenance),
                )
            if (
                args.model_every > 0
                and latest_step.environment_step % args.model_every == 0
            ):
                save_model_checkpoint(
                    trainer,
                    run_dir
                    / "milestones"
                    / f"model-step-{latest_step.environment_step:09d}.pt",
                    metadata=_metadata(run_id, trainer, provenance),
                )
        save_full_checkpoint(
            trainer,
            checkpoint_path,
            metadata=_metadata(run_id, trainer, provenance),
        )
        save_model_checkpoint(
            trainer,
            run_dir / "final-model.pt",
            metadata=_metadata(run_id, trainer, provenance),
        )
        manifest["status"] = "budget_complete"
        manifest["completed_at_utc"] = utc_now()
        manifest["final"] = _metadata(run_id, trainer, provenance)
        atomic_write_json(run_dir / "manifest.json", manifest)
        if args.ledger:
            append_jsonl(
                args.ledger,
                {"event": "run_budget_complete", "time_utc": utc_now(), **_metadata(run_id, trainer, provenance)},
                sync=True,
            )
        return 0
    except (GracefulStop, KeyboardInterrupt) as error:
        save_full_checkpoint(
            trainer,
            checkpoint_path,
            metadata={"stop_reason": str(error), **_metadata(run_id, trainer, provenance)},
        )
        manifest["status"] = "stopped_with_checkpoint"
        manifest["stop_reason"] = str(error)
        manifest["final"] = _metadata(run_id, trainer, provenance)
        atomic_write_json(run_dir / "manifest.json", manifest)
        return 130
    except Exception as error:
        crash_path = run_dir / "crash-full.pt"
        save_full_checkpoint(
            trainer,
            crash_path,
            metadata={"error": repr(error), **_metadata(run_id, trainer, provenance)},
        )
        manifest["status"] = "failed_with_checkpoint"
        manifest["error"] = repr(error)
        atomic_write_json(run_dir / "manifest.json", manifest)
        raise


def _evaluation_seeds(partition: str) -> list[int]:
    ranges = {
        "tuning": (410000, 20),
        "validation": (420000, 100),
        "test": (430000, 100),
    }
    start, count = ranges[partition]
    return list(range(start, start + count))


def evaluate_command(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation artifact {output}")
    payload = _load_payload(Path(args.checkpoint), args.device)
    config: TrainerConfig = payload["trainer_config"]
    learner = MADDPGLearner(config.maddpg, device=args.device)
    learner.load_state_dict(payload["learner"])
    seeds = (
        list(range(args.seed_start, args.seed_start + args.seed_count))
        if args.seed_start is not None
        else _evaluation_seeds(args.partition)
    )
    summary = evaluate_actor_only(learner, config.environment, seeds)
    record = {
        "event": "actor_only_evaluation",
        "time_utc": utc_now(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_metadata": payload.get("metadata", {}),
        "partition": args.partition if args.seed_start is None else "custom",
        "guidance_enabled": False,
        "exploration_enabled": False,
        "summary": summary,
    }
    atomic_write_json(output, record)
    print(json.dumps(json_safe(record), sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maadpg-reproduction")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--variant", choices=("maddpg", "maadpg"), required=True)
    train.add_argument("--run-dir", required=True)
    train.add_argument("--run-id")
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--step-budget", type=int, required=True)
    train.add_argument("--checkpoint-every", type=int, default=25_000)
    train.add_argument("--model-every", type=int, default=50_000)
    train.add_argument("--diagnostic-every", type=int, default=1_000)
    train.add_argument("--resume")
    train.add_argument("--ledger")
    train.add_argument("--small-test", action="store_true")
    train.set_defaults(function=train_command)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument(
        "--partition", choices=("tuning", "validation", "test"), default="validation"
    )
    evaluate.add_argument("--seed-start", type=int)
    evaluate.add_argument("--seed-count", type=int, default=1)
    evaluate.set_defaults(function=evaluate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("step_budget", "checkpoint_every", "diagnostic_every"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.command == "evaluate" and args.seed_start is None and args.seed_count != 1:
        parser.error("--seed-count requires --seed-start")
    if args.command == "evaluate" and args.seed_count <= 0:
        parser.error("--seed-count must be positive")
    return int(args.function(args))
