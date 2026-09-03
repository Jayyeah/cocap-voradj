#!/usr/bin/env python3
"""Supervise a MAPPO-9-v2 actor from fixed-midpoint IQN Q targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.small_step_ac import tensor_tree
from cocap_voradj.training.continuous.formal_config import resolve_ladder_config
from tools.run_small_step_ac_migration import configure_environment, make_components

DATASET_SCHEMA = "iqn-aw-teacher-dataset-v1"
CHECKPOINT_SCHEMA = "mappo-iqn-distilled-actor-v1"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_dataset(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != DATASET_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("teacher dataset is not a complete supported artifact")
    shards = list(manifest.get("shards", []))
    if not shards:
        raise ValueError("teacher dataset has no shards")
    storage: dict[str, list[np.ndarray]] = {}
    for shard in shards:
        path = root / str(shard["path"])
        if sha256_file(path) != str(shard["sha256"]):
            raise ValueError(f"dataset shard SHA mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            for key in payload.files:
                storage.setdefault(key, []).append(np.asarray(payload[key]))
    arrays = {key: np.concatenate(values, axis=0) for key, values in storage.items()}
    if len(arrays["teacher_q"]) != int(manifest["row_count"]):
        raise ValueError("dataset row count mismatch")
    if arrays["teacher_q"].shape[1:] != (9,):
        raise ValueError("teacher_q must have shape [rows,9]")
    if not np.array_equal(arrays["teacher_q"].argmax(axis=1), arrays["greedy_action"]):
        raise ValueError("greedy_action does not match stored teacher_q")
    return arrays, manifest


def local_batch(arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    tree = {
        key.removeprefix("local_obs."): value[indices]
        for key, value in arrays.items()
        if key.startswith("local_obs.")
    }
    return tensor_tree(tree, torch.device(device))


@torch.no_grad()
def evaluate_split(actor, arrays, indices: np.ndarray, device: str, temperature: float, batch_size: int) -> dict[str, Any]:
    actor.eval()
    totals = {"kl": 0.0, "hard_ce": 0.0, "agreement": 0.0, "rows": 0}
    group_hits: dict[str, list[int]] = {}
    for start in range(0, len(indices), int(batch_size)):
        selected = indices[start:start + int(batch_size)]
        obs = local_batch(arrays, selected, device)
        q = torch.as_tensor(arrays["teacher_q"][selected], device=device, dtype=torch.float32)
        target = torch.softmax((q - q.max(dim=-1, keepdim=True).values) / float(temperature), dim=-1)
        teacher_action = torch.as_tensor(arrays["greedy_action"][selected], device=device, dtype=torch.long)
        log_prob = actor.distribution(obs).logits
        kl_rows = torch.sum(target * (torch.log(target.clamp_min(1e-12)) - log_prob), dim=-1)
        ce_rows = F.nll_loss(log_prob, teacher_action, reduction="none")
        prediction = log_prob.argmax(dim=-1)
        hits = prediction.eq(teacher_action)
        count = len(selected)
        totals["kl"] += float(kl_rows.sum())
        totals["hard_ce"] += float(ce_rows.sum())
        totals["agreement"] += float(hits.float().sum())
        totals["rows"] += count
        for name, mask in {
            "support": arrays["support"][selected].astype(bool),
            "direct": arrays["direct"][selected].astype(bool),
            "pursuing": arrays["pursuing"][selected].astype(bool),
            "near_capture": arrays["near_capture"][selected].astype(bool),
        }.items():
            group_hits.setdefault(name, [0, 0])
            group_hits[name][0] += int(hits.cpu().numpy()[mask].sum())
            group_hits[name][1] += int(mask.sum())
    rows = max(int(totals["rows"]), 1)
    return {
        "rows": int(totals["rows"]),
        "kl": totals["kl"] / rows,
        "hard_bc_ce": totals["hard_ce"] / rows,
        "action_agreement": totals["agreement"] / rows,
        "group_action_agreement": {
            name: (hits / count if count else None)
            for name, (hits, count) in group_hits.items()
        },
        "group_rows": {name: count for name, (_, count) in group_hits.items()},
    }


def save_actor_checkpoint(path: Path, actor, payload: Mapping[str, Any]) -> None:
    value = {
        "schema": CHECKPOINT_SCHEMA,
        "actor_state_dict": {key: tensor.detach().cpu() for key, tensor in actor.state_dict().items()},
        **dict(payload),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    verified = torch.load(temporary, map_location="cpu", weights_only=True)
    if verified.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("distilled actor checkpoint verification failed")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--validation-modulus", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026091401)
    args = parser.parse_args()
    if args.temperature <= 0 or min(args.epochs, args.batch_size, args.validation_modulus) <= 0:
        parser.error("temperature, epochs, batch-size and validation-modulus must be positive")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    arrays, dataset_manifest = load_dataset(dataset_root)
    teacher_path = Path(args.teacher_checkpoint).resolve()
    teacher_sha = sha256_file(teacher_path)
    if teacher_sha != str(args.expected_teacher_sha256).lower():
        raise ValueError("teacher checkpoint SHA mismatch")
    if teacher_sha != str(dataset_manifest["teacher_checkpoint_sha256"]):
        raise ValueError("distillation teacher differs from dataset teacher")

    root_config = resolve_ladder_config(Path(args.config).resolve())
    config = configure_environment(root_config, "mappo9_v2")
    trainer, _ = make_components(config, "mappo9_v2", str(args.device))
    actor = trainer.actor
    teacher = CoCapIQN.load(str(teacher_path), device=str(args.device)).eval()
    incompatible = actor.encoder.load_legacy_iqn_state_dict(teacher.state_dict(), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("IQN to Legacy backbone mapping was not strict")
    mapped_count = len(actor.encoder.map_legacy_iqn_decision_keys(teacher.state_dict()))
    for parameter in actor.encoder.parameters():
        parameter.requires_grad_(False)
    actor.encoder.eval()
    trainable = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=float(args.learning_rate), eps=1e-5)

    episode_ids = arrays["episode"].astype(np.int64)
    validation_mask = episode_ids % int(args.validation_modulus) == 0
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("episode-level train/validation split is empty")

    history: list[dict[str, Any]] = []
    best_key = (-1.0, float("-inf"))
    best_path = output_root / "distilled_actor.pt"
    for epoch in range(1, int(args.epochs) + 1):
        actor.train()
        actor.encoder.eval()
        permutation = np.random.permutation(train_indices)
        loss_sum = agreement_sum = rows = 0.0
        for start in range(0, len(permutation), int(args.batch_size)):
            selected = permutation[start:start + int(args.batch_size)]
            obs = local_batch(arrays, selected, str(args.device))
            q = torch.as_tensor(arrays["teacher_q"][selected], device=args.device, dtype=torch.float32)
            target = torch.softmax((q - q.max(dim=-1, keepdim=True).values) / float(args.temperature), dim=-1)
            teacher_action = torch.as_tensor(arrays["greedy_action"][selected], device=args.device, dtype=torch.long)
            log_prob = actor.distribution(obs).logits
            loss = F.kl_div(log_prob, target, reduction="batchmean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            optimizer.step()
            count = len(selected)
            loss_sum += float(loss.detach()) * count
            agreement_sum += float(log_prob.argmax(dim=-1).eq(teacher_action).float().sum())
            rows += count
        validation = evaluate_split(
            actor, arrays, validation_indices, str(args.device), float(args.temperature), int(args.batch_size)
        )
        row = {
            "epoch": epoch,
            "train_kl": loss_sum / max(rows, 1),
            "train_action_agreement": agreement_sum / max(rows, 1),
            "last_grad_norm": grad_norm,
            "validation": validation,
        }
        history.append(row)
        key = (float(validation["action_agreement"]), -float(validation["kl"]))
        if key > best_key:
            best_key = key
            save_actor_checkpoint(best_path, actor, {
                "config": str(Path(args.config).resolve()),
                "teacher_checkpoint": str(teacher_path),
                "teacher_checkpoint_sha256": teacher_sha,
                "dataset_manifest": str(dataset_root / "manifest.json"),
                "dataset_manifest_sha256": sha256_file(dataset_root / "manifest.json"),
                "temperature": float(args.temperature),
                "epoch": epoch,
                "validation": validation,
                "legacy_backbone_mapped_key_count": mapped_count,
                "backbone_frozen": True,
                "seed": int(args.seed),
            })
        atomic_json(output_root / "training_history.json", {
            "schema": "mappo-iqn-distillation-history-v1",
            "status": "running",
            "history": history,
            "best_checkpoint": str(best_path),
        })

    best_payload = torch.load(best_path, map_location=str(args.device), weights_only=True)
    actor.load_state_dict(best_payload["actor_state_dict"], strict=True)
    final_train = evaluate_split(actor, arrays, train_indices, str(args.device), float(args.temperature), int(args.batch_size))
    final_validation = evaluate_split(actor, arrays, validation_indices, str(args.device), float(args.temperature), int(args.batch_size))
    report = {
        "schema": "mappo-iqn-distillation-report-v1",
        "status": "complete",
        "completed_at": now(),
        "dataset_rows": int(len(arrays["teacher_q"])),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "teacher_checkpoint_sha256": teacher_sha,
        "temperature": float(args.temperature),
        "backbone_copy": {
            "strict": True,
            "mapped_key_count": mapped_count,
            "frozen_during_supervision": True,
        },
        "best_epoch": int(best_payload["epoch"]),
        "train": final_train,
        "validation": final_validation,
        "checkpoint": str(best_path),
        "checkpoint_sha256": sha256_file(best_path),
        "hard_bc_sanity": "hard-label CE is reported on the same soft-KL student; no hyperparameter sweep",
    }
    atomic_json(output_root / "report.json", report)
    (output_root / "DISTILLATION_DONE").write_text(now() + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
