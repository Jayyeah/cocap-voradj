"""Auditable IQN-teacher transfer and behavior cloning for local TD3."""
from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LegacyVorAdjFeatureBackbone,
    LegacyVorAdjFeatureBackboneConfig,
)
from cocap_voradj.models.continuous.local_td3 import LocalTD3Actor
from cocap_voradj.models.iqn import CoCapIQN, CoCapNetConfig
from cocap_voradj.training.td3_stage1_contract import aw9_grid


DATASET_SCHEMA = "td3-local-aw-iqn-teacher-v1"
WARMSTART_SCHEMA = "td3-local-aw-iqn-warmstart-v1"
OBS_KEYS = ("self", "pursuers", "evaders", "obstacles", "masks", "types")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_iqn_teacher(
    checkpoint: str | Path, device: str = "cpu"
) -> tuple[CoCapIQN, Dict[str, Any]]:
    """Safely load and validate the successful legacy IQN teacher."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(payload) != {"state_dict", "config", "extra"}:
        raise ValueError(f"unexpected IQN checkpoint keys: {sorted(payload)}")
    config = CoCapNetConfig(**payload["config"])
    expected = {
        "architecture": "voradj_single_head",
        "action_size": 9,
        "self_feature_dim": 9,
        "max_pursuers": 8,
        "max_evaders": 8,
        "max_obstacles": 5,
    }
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError(
                f"IQN teacher {name} mismatch: expected {value!r}, got {getattr(config, name)!r}"
            )
    model = CoCapIQN(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    metadata = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "state_sha256": tensor_state_sha256(payload["state_dict"]),
        "config": asdict(config),
        "extra": copy.deepcopy(payload["extra"]),
    }
    return model, metadata


def backbone_config_from_iqn(config: CoCapNetConfig) -> LegacyVorAdjFeatureBackboneConfig:
    names = {field.name for field in fields(LegacyVorAdjFeatureBackboneConfig)}
    values = {name: getattr(config, name) for name in names if hasattr(config, name)}
    # CoCapIQN hard-codes this dropout and eval-mode is used by deterministic TD3.
    values["dropout"] = 0.1
    return LegacyVorAdjFeatureBackboneConfig(**values)


def transfer_iqn_backbone(
    actor: LocalTD3Actor, teacher: CoCapIQN
) -> Dict[str, Any]:
    """Strictly copy decision representation; never copy an IQN output head."""
    expected_config = backbone_config_from_iqn(teacher.config)
    if actor.backbone.config != expected_config:
        raise ValueError(
            f"actor/IQN backbone config mismatch: {actor.backbone.config!r} != {expected_config!r}"
        )
    source = teacher.state_dict()
    mapped = LegacyVorAdjFeatureBackbone.map_legacy_iqn_decision_keys(source)
    actor_keys = set(actor.backbone.state_dict())
    if set(mapped) != actor_keys:
        missing = sorted(actor_keys.difference(mapped))
        unexpected = sorted(set(mapped).difference(actor_keys))
        raise ValueError(f"IQN mapping boundary mismatch: missing={missing}, unexpected={unexpected}")
    forbidden = (
        "single_head.",
        "cos_embedding.",
        "coverage_head.",
        "encirclement_head.",
        "gate_head.",
    )
    if any(key.startswith(forbidden) for key in mapped):
        raise AssertionError("IQN quantile/action head crossed the warm-start boundary")
    actor.backbone.load_state_dict(mapped, strict=True)
    actor.eval()
    return {
        "schema": WARMSTART_SCHEMA,
        "mapped_key_count": len(mapped),
        "mapped_keys": sorted(mapped),
        "mapped_state_sha256": tensor_state_sha256(mapped),
        "excluded_prefixes": list(forbidden),
        "critic_transfer": "none",
    }


def _obs_batch(
    dataset: Mapping[str, np.ndarray], indices: Sequence[int] | np.ndarray, device: str
) -> Dict[str, torch.Tensor]:
    index = np.asarray(indices, dtype=np.int64)
    return {
        key: torch.as_tensor(dataset[f"obs_{key}"][index], dtype=torch.float32, device=device)
        for key in OBS_KEYS
    }


def nearest_aw9_index(actions: torch.Tensor) -> torch.Tensor:
    """Map physical AW to the nearest legacy center in normalized coordinates."""
    grid = torch.as_tensor(aw9_grid(), dtype=actions.dtype, device=actions.device)
    scale = torch.as_tensor([0.4, np.pi / 6.0], dtype=actions.dtype, device=actions.device)
    distance = ((actions[:, None, :] - grid[None, :, :]) / scale).square().sum(-1)
    return distance.argmin(-1)


@torch.no_grad()
def behavior_clone_metrics(
    actor: LocalTD3Actor,
    dataset: Mapping[str, np.ndarray],
    indices: Sequence[int] | np.ndarray,
    device: str,
    batch_size: int = 1024,
) -> Dict[str, float]:
    actor.eval()
    ids = np.asarray(indices, dtype=np.int64)
    squared: list[np.ndarray] = []
    absolute: list[np.ndarray] = []
    agreement: list[np.ndarray] = []
    for start in range(0, len(ids), int(batch_size)):
        selection = ids[start : start + int(batch_size)]
        predicted = actor(_obs_batch(dataset, selection, device))
        target = torch.as_tensor(
            dataset["action_aw"][selection], dtype=torch.float32, device=device
        )
        error = predicted - target
        squared.append(error.square().cpu().numpy())
        absolute.append(error.abs().cpu().numpy())
        target_index = torch.as_tensor(
            dataset["action_index"][selection], dtype=torch.long, device=device
        )
        agreement.append((nearest_aw9_index(predicted) == target_index).cpu().numpy())
    if not squared:
        raise ValueError("behavior-cloning split is empty")
    sq = np.concatenate(squared)
    ab = np.concatenate(absolute)
    agree = np.concatenate(agreement)
    return {
        "mse": float(sq.mean()),
        "a_mse": float(sq[:, 0].mean()),
        "w_mse": float(sq[:, 1].mean()),
        "a_mae": float(ab[:, 0].mean()),
        "w_mae": float(ab[:, 1].mean()),
        "action_index_agreement": float(agree.mean()),
        "rows": int(len(ids)),
    }


def behavior_clone_head(
    actor: LocalTD3Actor,
    dataset: Mapping[str, np.ndarray],
    *,
    device: str,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    seed: int = 20260916,
    rollout_callback: Callable[[LocalTD3Actor, int], Mapping[str, Any]] | None = None,
    rollout_epochs: Sequence[int] = (0, 5, 10, 20, 30),
) -> Dict[str, Any]:
    """Train only the continuous actor head against physical IQN AW actions."""
    split = np.asarray(dataset["split"], dtype=np.uint8)
    train_ids = np.flatnonzero(split == 0)
    heldout_ids = np.flatnonzero(split == 1)
    if not len(train_ids) or not len(heldout_ids):
        raise ValueError("dataset must contain non-empty episode-level train and heldout splits")
    actor.to(device).eval()
    for parameter in actor.backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in actor.head.parameters():
        parameter.requires_grad_(True)
    backbone_before = tensor_state_sha256(actor.backbone.state_dict())
    generator = np.random.default_rng(int(seed))
    optimizer = torch.optim.Adam(actor.head.parameters(), lr=float(learning_rate))
    initial = {
            "epoch": 0,
            "train": behavior_clone_metrics(actor, dataset, train_ids, device),
            "heldout": behavior_clone_metrics(actor, dataset, heldout_ids, device),
        }
    if rollout_callback is not None and 0 in set(int(value) for value in rollout_epochs):
        initial["deterministic_rollout"] = dict(rollout_callback(actor, 0))
    history = [initial]
    for epoch in range(1, int(epochs) + 1):
        order = generator.permutation(train_ids)
        losses: list[float] = []
        for start in range(0, len(order), int(batch_size)):
            selection = order[start : start + int(batch_size)]
            predicted = actor(_obs_batch(dataset, selection, device))
            target = torch.as_tensor(
                dataset["action_aw"][selection], dtype=torch.float32, device=device
            )
            loss = F.mse_loss(predicted, target)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("behavior-cloning loss is NaN or Inf")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {
                "epoch": epoch,
                "batch_loss_mean": float(np.mean(losses)),
                "train": behavior_clone_metrics(actor, dataset, train_ids, device),
                "heldout": behavior_clone_metrics(actor, dataset, heldout_ids, device),
            }
        if rollout_callback is not None and epoch in set(int(value) for value in rollout_epochs):
            row["deterministic_rollout"] = dict(rollout_callback(actor, epoch))
        history.append(row)
    backbone_after = tensor_state_sha256(actor.backbone.state_dict())
    if backbone_before != backbone_after:
        raise AssertionError("frozen IQN backbone changed during behavior cloning")
    return {
        "schema": WARMSTART_SCHEMA,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "backbone_frozen": True,
        "backbone_sha256": backbone_after,
        "history": history,
        "final": history[-1],
    }
