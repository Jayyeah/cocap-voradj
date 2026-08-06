"""Generate a strict compatibility report for legacy IQN encoder transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoderConfig, LocalEntityTokenEncoder
from cocap_voradj.training.continuous.formal_config import resolve_formal_config


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "configs/experiments/continuous_marl_20260804/p6_formal_central_masac_4v1.yaml"
DEFAULT_LEGACY = ROOT / "artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage1_4v1_step_2000000.pt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-checkpoint", default=str(DEFAULT_LEGACY))
    parser.add_argument("--config", default=str(FORMAL))
    parser.add_argument("--out-md", default=str(ROOT / "artifacts/2026-08-06_ctde_contract/legacy_iqn_compatibility_report.md"))
    parser.add_argument("--out-json", default=str(ROOT / "artifacts/2026-08-06_ctde_contract/legacy_iqn_compatibility_report.json"))
    args = parser.parse_args()

    config = resolve_formal_config(args.config)
    actor_cfg = config["actor"]
    encoder = LocalEntityTokenEncoder(
        LocalEntityTokenEncoderConfig(
            hidden_dim=int(actor_cfg["hidden_dim"]),
            num_heads=int(actor_cfg["num_heads"]),
            num_layers=int(actor_cfg["num_layers"]),
            self_feature_dim=int(actor_cfg.get("self_feature_dim", 9)),
            max_pursuers=int(actor_cfg.get("max_pursuers", 12)),
            max_evaders=int(actor_cfg.get("max_evaders", 8)),
            max_obstacles=int(actor_cfg.get("max_obstacles", 5)),
            dropout=float(actor_cfg.get("dropout", 0.0)),
        )
    )
    payload = torch.load(args.legacy_checkpoint, map_location="cpu", weights_only=False)
    source = payload.get("state_dict", payload)
    legacy_cfg = payload.get("config", {}) or {}
    target = encoder.state_dict()
    prefixes = ("encoders.", "type_embedding.", "transformer.")
    source_candidate = {k: v for k, v in source.items() if k.startswith(prefixes)}
    target_candidate = {k: v for k, v in target.items() if k.startswith(prefixes)}
    missing = sorted(set(target_candidate) - set(source_candidate))
    unexpected = sorted(set(source_candidate) - set(target_candidate))
    shape_mismatch = sorted(
        key for key in set(target_candidate) & set(source_candidate)
        if tuple(target_candidate[key].shape) != tuple(source_candidate[key].shape)
    )
    loaded_keys = sorted(set(target_candidate) & set(source_candidate) - set(shape_mismatch))
    loaded_count = sum(int(target_candidate[key].numel()) for key in loaded_keys)
    target_count = sum(int(value.numel()) for value in target_candidate.values())
    ratio = loaded_count / max(target_count, 1)
    checkpoint_sha256 = hashlib.sha256(Path(args.legacy_checkpoint).read_bytes()).hexdigest()
    success = not missing and not unexpected and not shape_mismatch
    if success:
        encoder.load_legacy_iqn_state_dict(source, strict=True)
    report = {
        "legacy_checkpoint": str(args.legacy_checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "legacy_config": {k: v for k, v in legacy_cfg.items()},
        "target_encoder_config": encoder.config.__dict__,
        "loaded_keys": loaded_keys,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch_keys": shape_mismatch,
        "loaded_parameter_count": loaded_count,
        "target_encoder_parameter_count": target_count,
        "loaded_ratio": ratio,
        "exact_shape_transfer_possible": success,
        "action_head_loaded": any("pis" in key or "head" in key or "quantile" in key for key in loaded_keys),
        "critic_loaded": False,
    }
    md = [
        "# Legacy IQN Encoder Compatibility Report",
        "",
        f"- checkpoint: `{args.legacy_checkpoint}`",
        f"- SHA256: `{checkpoint_sha256}`",
        f"- exact_shape_transfer_possible: **{success}**",
        f"- loaded keys: {len(loaded_keys)}",
        f"- missing: {missing}",
        f"- unexpected: {unexpected}",
        f"- shape mismatch: {shape_mismatch}",
        f"- loaded parameter ratio: {ratio:.6f}",
        "",
        "Only `encoders.*`, `type_embedding.*`, `transformer.*` are eligible. "
        "IQN quantile/action/gate heads are not loaded.",
        "",
    ]
    Path(args.out_md).write_text("\n".join(md), encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
