"""Check local-SAC numerical trend independently on each P4 smoke scene."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoderConfig
from cocap_voradj.models.continuous.radial_actor import RadialActorConfig
from cocap_voradj.training.continuous.joint_replay import JointReplayBuffer
from cocap_voradj.training.continuous.local_sac import LocalSACConfig, LocalSACTrainer


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/2026-08-04_continuous_marl_refactor"
REPORT = ARTIFACT_ROOT / "p4_local_sac_per_scene_trend.json"
REPLAY = ARTIFACT_ROOT / "p4_local_sac_smoke_joint_replay_25k.pkl"
REPLAY_REPORT = ARTIFACT_ROOT / "p4_local_sac_smoke_report_25k.json"


def main() -> int:
    replay_report = json.loads(REPLAY_REPORT.read_text(encoding="utf-8"))
    replay = JointReplayBuffer.load(REPLAY, replay_report["manifest"])
    result = {}
    for scene in ("pure_ce", "capture", "mixed_crms"):
        ids = [idx for idx in replay.transition_ids if replay.get(idx).metadata.get("scene") == scene]
        trainer = LocalSACTrainer(
            encoder_config=LocalEntityTokenEncoderConfig(hidden_dim=16, num_heads=4, num_layers=1),
            actor_config=RadialActorConfig(hidden_dim=16, a_max=0.8, decision_dt=0.5),
            config=LocalSACConfig(hidden_dim=16),
        )
        losses = []
        for _ in range(20):
            sampled = replay._draw(ids, min(64, len(ids)))
            metrics = trainer.update(replay.batch_from_ids(sampled))
            losses.append(metrics)
        result[scene] = {
            "transition_count": len(ids),
            "updates": len(losses),
            "critic_loss_first": losses[0]["critic_loss"],
            "critic_loss_last": losses[-1]["critic_loss"],
            "actor_loss_first": losses[0]["actor_loss"],
            "actor_loss_last": losses[-1]["actor_loss"],
            "all_finite": bool(all(item["finite"] == 1.0 for item in losses)),
            "critic_loss_decreased": bool(losses[-1]["critic_loss"] < losses[0]["critic_loss"]),
        }
    payload = {
        "schema_version": 1,
        "kind": "continuous_local_sac_per_scene_trend",
        "replay": str(REPLAY),
        "scenes": result,
        "gate": bool(all(item["all_finite"] and item["critic_loss_decreased"] for item in result.values())),
    }
    REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"report={REPORT}")
    return 0 if payload["gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
