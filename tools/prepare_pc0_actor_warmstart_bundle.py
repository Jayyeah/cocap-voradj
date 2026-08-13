#!/usr/bin/env python3
"""Build a strict step-0 PC0 bundle with CF3 Actor and all other state fresh."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from cocap_voradj.training.continuous.formal_config import (
    resolve_ladder_config,
    scene_config,
    training_mode_contract,
)
from cocap_voradj.training.continuous.joint_replay import JointReplayBuffer
from tools.run_continuous_ctde_training import (
    _make_trainer,
    _manifest,
    _runtime_state,
    _set_seed,
    _stable_hash,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / (
    "configs/experiments/parallel_ce_legacy_voradj_20260809/"
    "legacy_voradj_pc0_cf3_actor_warmstart_postcapture300_4p1e1obs_100k_aw.yaml"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def transition_semantics_audit(source_config: Path, target_config: Path) -> dict[str, Any]:
    source = scene_config(resolve_ladder_config(source_config), "capture")
    target = scene_config(resolve_ladder_config(target_config), "capture")
    source_terminal = bool(source.get("voradj", {}).get("capture_episode_ends_on_capture", False))
    target_terminal = bool(target.get("voradj", {}).get("capture_episode_ends_on_capture", False))
    source_window = int(source.get("reward", {}).get("post_capture_coverage_window_steps", 0))
    target_window = int(target.get("reward", {}).get("post_capture_coverage_window_steps", 0))
    terminal_changed = source_terminal != target_terminal
    return {
        "schema_version": 1,
        "source_capture_transition_terminal": source_terminal,
        "target_capture_transition_terminal": target_terminal,
        "source_post_capture_window_steps": source_window,
        "target_post_capture_window_steps": target_window,
        "terminal_semantics_changed": terminal_changed,
        "bootstrap_target_semantics_changed": terminal_changed,
        "old_replay_inheritance_allowed": not terminal_changed,
        "required_initialization": (
            "actor_only_warmstart_fresh_replay_critics_targets_optimizers_alpha_rng_runtime"
            if terminal_changed else "full_resume_may_be_considered_after_separate_contract_audit"
        ),
    }


def build_bundle(
    *, source_checkpoint: Path, source_config: Path, target_config: Path,
    output_dir: Path, seed: int, tag: str,
) -> dict[str, Any]:
    semantics = transition_semantics_audit(source_config, target_config)
    if semantics["old_replay_inheritance_allowed"]:
        raise ValueError("PC0 builder expects an incompatible terminal transition and must fail closed otherwise")
    config = resolve_ladder_config(target_config)
    _set_seed(seed)
    trainer = _make_trainer(config, "cpu")
    fresh_critic1_hash = state_sha256(trainer.critic1.state_dict())
    fresh_critic2_hash = state_sha256(trainer.critic2.state_dict())
    fresh_target1_hash = state_sha256(trainer.target_critic1.state_dict())
    fresh_target2_hash = state_sha256(trainer.target_critic2.state_dict())
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 2 or "actor" not in payload:
        raise ValueError("source CF3 trainer checkpoint schema mismatch")
    incompatible = trainer.actor.load_state_dict(payload["actor"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Actor architecture mismatch: {incompatible}")
    actor_hash = state_sha256(trainer.actor.state_dict())
    source_actor_hash = state_sha256(payload["actor"])
    if actor_hash != source_actor_hash:
        raise ValueError("Actor state changed while loading PC0 warm start")
    if state_sha256(trainer.critic1.state_dict()) != fresh_critic1_hash:
        raise ValueError("critic1 was unexpectedly inherited")
    if state_sha256(trainer.critic2.state_dict()) != fresh_critic2_hash:
        raise ValueError("critic2 was unexpectedly inherited")
    if state_sha256(trainer.target_critic1.state_dict()) != fresh_target1_hash:
        raise ValueError("target critic1 was unexpectedly inherited")
    if state_sha256(trainer.target_critic2.state_dict()) != fresh_target2_hash:
        raise ValueError("target critic2 was unexpectedly inherited")
    optimizer_state_sizes = {
        "actor": len(trainer.actor_optimizer.state_dict()["state"]),
        "critic": len(trainer.critic_optimizer.state_dict()["state"]),
        "alpha": len(trainer.alpha_optimizer.state_dict()["state"]),
    }
    if any(optimizer_state_sizes.values()):
        raise ValueError(f"PC0 optimizer state is not fresh: {optimizer_state_sizes}")
    scenes = tuple(str(item) for item in config["training"]["scene_cycle"])
    scene_hashes = {scene: _stable_hash(scene_config(config, scene)) for scene in scenes}
    manifest = _manifest(config, seed, tag, trainer, scene_hashes, config_path=str(target_config))
    mode = training_mode_contract(config)
    rng = np.random.default_rng(seed)
    runtime = _runtime_state(
        transition_count=0, update_count=0, scene_index=0, current_scene=scenes[0],
        recovery_pool=deque(maxlen=int(config.get("recovery", {}).get("capture_state_pool_capacity", 1000))),
        metrics_history=[], runner_rng=rng, focal_sampler=None, training_mode=mode,
        scene_counts={}, origin_counts={}, sampling_stats={}, all_finite=True, update_metrics_tail=[],
    )
    replay = JointReplayBuffer(
        capacity=int(config["replay"]["capacity_joint"]),
        max_agents=int(config["training"]["max_agents"]), seed=seed,
    )
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    trainer.save_checkpoint(temporary / "trainer.pt", manifest, runtime_state=runtime)
    replay.save(temporary / "replay.pkl", manifest, runtime_state=runtime)
    (temporary / "runtime_state.pkl").write_bytes(pickle.dumps(runtime))
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (temporary / "effective_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (temporary / "metrics.jsonl").write_text("", encoding="utf-8")
    (temporary / "diagnostic_eval.json").write_text("{}\n", encoding="utf-8")
    (temporary / "checkpoint_storage.json").write_text(
        json.dumps({
            "schema_version": 1, "kind": "pc0_actor_only_step0_initialization",
            "step": 0, "contains_replay": True, "replay_size": 0,
        }, indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": 1,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "target_config": str(target_config),
        "seed": seed,
        "tag": tag,
        "transition_semantics": semantics,
        "actor_state_sha256": actor_hash,
        "actor_inherited": True,
        "replay_size": 0,
        "critic_inherited": False,
        "target_critic_inherited": False,
        "optimizer_inherited": False,
        "optimizer_state_entries": optimizer_state_sizes,
        "alpha_inherited": False,
        "alpha_value": float(trainer.alpha.detach()),
        "rng_inherited": False,
        "runtime_inherited": False,
        "target_critics_equal_fresh_critics": bool(
            fresh_target1_hash == fresh_critic1_hash and fresh_target2_hash == fresh_critic2_hash
        ),
        "manifest_sha256": file_sha256(temporary / "manifest.json"),
    }
    (temporary / "actor_only_initialization_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing PC0 init bundle: {output_dir}")
    os.replace(temporary, output_dir)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--target-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026081306)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    audit = build_bundle(
        source_checkpoint=Path(args.source_checkpoint), source_config=Path(args.source_config),
        target_config=Path(args.target_config), output_dir=Path(args.output_dir),
        seed=args.seed, tag=args.tag,
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
