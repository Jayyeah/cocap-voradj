#!/usr/bin/env python3
"""Small non-training audit for Final IQN recovery cross-initialization."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.shared_local_ac import RecoveryInitPool
from cocap_voradj.training.trainer import deep_update, load_config, set_global_config


def make(config, scene, seed):
    cfg = deep_update(config, (config.get("tasks", {}) or {}).get(scene, {}) or {})
    cfg.setdefault("env", {})["num_evaders"] = 0 if scene == "voradj_coverage" else 1
    set_global_config(cfg)
    return VorAdjEnv(cfg, seed=seed)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(str(root / "configs/experiments/ac_capability_20260920/ac_capability_mix_stage1.yaml"))
    pool = RecoveryInitPool(capacity=1000, captured_ratio=0.75, map_random_ratio=0.5)
    mixed = make(config, "voradj", 2026092050)
    mixed.reset()
    snapshot = {
        "step": 17,
        "positions": [[float(p.x), float(p.y)] for p in mixed.pursuers],
        "active_mask": [True] * len(mixed.pursuers),
    }
    pool.add(snapshot)
    coverage = make(config, "voradj_coverage", 2026092051)
    results = {}
    for source, positions, active, spawn_mode in (
        ("capture_snapshot", snapshot["positions"], snapshot["active_mask"], None),
        ("ordinary_map_random", None, None, "map_random"),
        ("synthetic_cluster", None, None, "inner_random_cluster"),
    ):
        if spawn_mode is not None:
            coverage.env_cfg["pursuer_spawn_mode"] = spawn_mode
        coverage.reset(initial_pursuer_positions=positions, initial_pursuer_active=active)
        active_mask = [not p.deactivated for p in coverage.pursuers]
        positions_finite = all(np.isfinite([p.x, p.y]).all() for p in coverage.pursuers)
        results[source] = {
            "reset_ok": coverage.episode_step == 0,
            "active_mask_reset": active_mask == [True] * len(coverage.pursuers),
            "positions_finite": bool(positions_finite),
            "capture_snapshot_cleared": coverage.capture_snapshot is None,
            "post_capture_state_cleared": not coverage.post_capture_started and coverage.post_capture_step == 0,
            "collision_free_init": not bool(coverage.last_collision_events),
            "source": source,
        }
    payload = {
        "pool_capacity": pool.data.maxlen,
        "pool_size": len(pool),
        "snapshot_storage": "in_memory_only",
        "sources": results,
        "pass": all(all(value for key, value in row.items() if key not in {"source"}) for row in results.values()),
        "fixture_only": True,
        "training_started": False,
    }
    output = root / "artifacts/2026-09-20_ac_capability_p0/cross_init_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
