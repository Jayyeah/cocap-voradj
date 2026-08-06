"""Measure local-SAC batch32/64 updates per second and peak CUDA memory."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoderConfig
from cocap_voradj.models.continuous.radial_actor import RadialActorConfig
from cocap_voradj.training.continuous.joint_replay import JointReplayBuffer
from cocap_voradj.training.continuous.local_sac import LocalSACConfig, LocalSACTrainer


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/2026-08-04_continuous_marl_refactor"
REPLAY = ARTIFACT_ROOT / "p4_local_sac_smoke_joint_replay_25k.pkl"
REPLAY_REPORT = ARTIFACT_ROOT / "p4_local_sac_smoke_report_25k.json"
REPORT = ARTIFACT_ROOT / "p4_local_sac_gpu_batch_report.json"


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this resource gate")
    device = "cuda:0"
    replay_report = json.loads(REPLAY_REPORT.read_text(encoding="utf-8"))
    replay = JointReplayBuffer.load(REPLAY, replay_report["manifest"])
    result = {}
    for batch_size in (32, 64):
        trainer = LocalSACTrainer(
            encoder_config=LocalEntityTokenEncoderConfig(hidden_dim=16, num_heads=4, num_layers=1),
            actor_config=RadialActorConfig(hidden_dim=16, a_max=0.8, decision_dt=0.5),
            config=LocalSACConfig(hidden_dim=16),
            device=device,
        )
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        metrics = []
        for _ in range(10):
            ids = replay._draw(replay.transition_ids, batch_size)
            metrics.append(trainer.update(replay.batch_from_ids(ids, device=device)))
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        result[str(batch_size)] = {
            "updates": len(metrics),
            "elapsed_s": elapsed,
            "updates_per_s": len(metrics) / max(elapsed, 1e-9),
            "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
            "all_finite": bool(all(item["finite"] == 1.0 for item in metrics)),
        }
        del trainer
        torch.cuda.empty_cache()
    payload = {"schema_version": 1, "kind": "continuous_local_sac_gpu_batch_gate", "device": torch.cuda.get_device_name(0), "result": result, "gate": bool(all(item["all_finite"] for item in result.values()))}
    REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"report={REPORT}")
    return 0 if payload["gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
