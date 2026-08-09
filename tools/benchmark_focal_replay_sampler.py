#!/usr/bin/env python3
"""Benchmark focal replay sampling against a saved replay bundle.

The benchmark is intentionally read-only: it loads the checkpoint replay and
times pool materialization plus quota sampling without constructing trainer
batches or touching the source artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from cocap_voradj.training.continuous.joint_replay import (
    FOCAL_BUCKETS,
    FocalReplaySampler,
    JointReplayBuffer,
)


DEFAULT_QUOTAS = {
    "pre_capture_pursuing": 64,
    "pre_capture_support": 8,
    "pre_capture_coverage": 8,
    "post_capture_coverage": 32,
    "pure_recovery_coverage": 16,
}


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "count": float(len(ordered)),
        "min_seconds": ordered[0],
        "median_seconds": statistics.median(ordered),
        "mean_seconds": statistics.fmean(ordered),
        "max_seconds": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026080901)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    load_started = time.perf_counter()
    replay = JointReplayBuffer.load(args.replay, manifest)
    load_seconds = time.perf_counter() - load_started

    pool_samples: list[float] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        sizes = {bucket: len(replay._role_pool(bucket)) for bucket in FOCAL_BUCKETS}
        pool_samples.append(time.perf_counter() - started)

    sampler = FocalReplaySampler(DEFAULT_QUOTAS, seed=args.seed)
    sampling_samples: list[float] = []
    last_items = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        last_items = sampler.sample_items(replay)
        sampling_samples.append(time.perf_counter() - started)
    item_keys = [
        (item.slot_id, item.generation_id, item.agent_id, item.bucket_id)
        for item in last_items
    ]
    item_digest = hashlib.sha256(
        json.dumps(item_keys, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    report = {
        "replay": str(args.replay.resolve()),
        "manifest": str(args.manifest.resolve()),
        "replay_size": len(replay),
        "pool_sizes": sizes,
        "load_seconds": load_seconds,
        "role_pool_all_buckets": _summary(pool_samples),
        "sample_items": _summary(sampling_samples),
        "last_batch_size": len(last_items),
        "last_item_digest": item_digest,
        "last_sample_stats": replay.last_sample_stats,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
