#!/usr/bin/env python3
"""CPU-only contract and reset-distribution preflight for A3 I0-I3."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.trainer import load_config, set_global_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/experiments/a3_capture_initialization_20260925"
CANONICAL = ROOT / "configs/experiments/forward_final_mappo_20260908/stage1_4v1.yaml"
STAGES = ("I0", "I1", "I2", "I3")


def contract_projection(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: copy.deepcopy(config.get(key))
        for key in (
            "env",
            "pursuer",
            "evader",
            "perception",
            "apf",
            "reward",
            "voradj",
            "tasks",
            "action_mode",
            "dynamics",
        )
    }


def check_stage(stage: str, canonical: Dict[str, Any]) -> Dict[str, Any]:
    config = load_config(str(CONFIG_ROOT / f"stage_{stage.lower()}.yaml"))
    if contract_projection(config) != contract_projection(canonical):
        raise AssertionError(f"{stage}: canonical Final contract projection changed")
    seeds = list(range(2026092503, 2026092511))
    records = []
    for seed in seeds:
        set_global_config(config)
        env = VorAdjEnv(copy.deepcopy(config), seed=seed)
        env.reset()
        metadata = env.initialization_metadata()
        distances = metadata["pursuer_target_distances"]
        if metadata["initial_capture_events"] != 0 or metadata["initial_collision_events"] != 0:
            raise AssertionError(f"{stage}/{seed}: initial terminal event")
        if not np.isclose(float(metadata["target_speed"]), 0.0):
            raise AssertionError(f"{stage}/{seed}: target speed is not zero")
        if min(distances) <= 8.0:
            raise AssertionError(f"{stage}/{seed}: reset capture distance violation")
        if metadata["pursuer_pairwise_min_center_distance"] < 15.0:
            raise AssertionError(f"{stage}/{seed}: pursuer separation violation")
        if stage == "I0" and not all(12.0 <= value <= 13.0 for value in distances):
            raise AssertionError(f"{stage}/{seed}: distance range violation")
        if stage == "I1" and not all(13.0 <= value <= 16.0 for value in distances):
            raise AssertionError(f"{stage}/{seed}: distance range violation")
        if stage == "I2":
            if metadata["direct_visible_count_by_center_range"] != 2:
                raise AssertionError(f"{stage}/{seed}: direct/hidden count violation")
            if not sum(value >= 23.0 for value in distances) == 2:
                raise AssertionError(f"{stage}/{seed}: hidden range violation")
        if stage == "I3" and min(distances) < 15.0:
            raise AssertionError(f"{stage}/{seed}: Final map-random lower bound violation")
        records.append(
            {
                "seed": seed,
                "source": metadata["source"],
                "direct_visible_count": metadata["direct_visible_count_by_center_range"],
                "min_distance": min(distances),
                "max_distance": max(distances),
                "min_pursuer_pair_distance": metadata["pursuer_pairwise_min_center_distance"],
                "obstacle_min_surface_clearance": metadata["obstacle_min_surface_clearance"],
            }
        )
    coverage = copy.deepcopy(config)
    coverage["env"] = copy.deepcopy(coverage["env"])
    coverage["env"]["num_evaders"] = 0
    set_global_config(coverage)
    coverage_env = VorAdjEnv(coverage, seed=2026092503)
    coverage_env.reset()
    if coverage_env.initialization_metadata().get("source") != "skipped_no_evader_task":
        raise AssertionError(f"{stage}: mixed coverage scene was altered by A3")
    return {
        "stage": stage,
        "contract_projection_match": True,
        "reset_samples": records,
        "coverage_scene_isolated": True,
        "budget": config["a3_initialization_curriculum"]["stages"][stage]["budget"],
    }


def main() -> None:
    canonical = load_config(str(CANONICAL))
    report = {
        "schema": "a3-initialization-only-preflight-v1",
        "status": "PASS",
        "training_steps": 0,
        "gpu_jobs": 0,
        "stages": {stage: check_stage(stage, canonical) for stage in STAGES},
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

