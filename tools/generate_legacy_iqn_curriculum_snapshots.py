"""Generate geometry curriculum snapshots from a legacy IQN checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.continuous.curriculum_snapshots import build_snapshot
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.trainer import set_global_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/crms_supportapproach_ce_curr_stage1_4p1e1obs_scratch2m_20260802_run1"


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    return yaml.safe_load((run_dir / "effective_config.yaml").read_text(encoding="utf-8"))


def _snapshot_key(snapshot: Dict[str, Any]) -> str:
    positions = snapshot["pursuers"]["position"]
    rounded = [tuple(round(float(v), 1) for v in pos) for pos in positions]
    return json.dumps(rounded, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--checkpoint", default=str(DEFAULT_RUN / "checkpoints/final_step_2000000.pt"))
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seeds", default="2026070901,2026070902,2026070903,2026070904")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts/2026-08-06_ctde_contract/curriculum_snapshots"))
    parser.add_argument("--source-branch", default="main")
    parser.add_argument("--source-commit", default="unknown")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    base_cfg = _load_run_config(run_dir)
    model = CoCapIQN.load(args.checkpoint, device=args.device)
    model.eval()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "snapshots.jsonl"
    seen: set[str] = set()
    snapshots: List[Dict[str, Any]] = []

    for scene in ("capture", "pure_ce", "mixed_crms"):
        for seed in seeds[: max(1, args.episodes)]:
            cfg = dict(base_cfg)
            cfg["device"] = "cpu"
            cfg.setdefault("env", {})
            cfg["env"]["num_evaders"] = 0 if scene == "pure_ce" else 1
            cfg["env"]["episode_max_length"] = args.max_steps
            cfg["env"]["num_pursuers"] = 4
            set_global_config(cfg)
            env = VorAdjEnv(cfg, seed=seed)
            obs_list = env.reset()
            apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
            for step in range(args.max_steps):
                active = [idx for idx, obs in enumerate(obs_list) if obs is not None]
                actions: List[Optional[int]] = [None] * len(obs_list)
                if active:
                    batch = stack_obs([obs_list[idx] for idx in active], args.device)
                    selected = model.act(
                        batch,
                        mode="voradj",
                        epsilon=0.0,
                        deterministic_quantiles=True,
                    ).detach().cpu().tolist()
                    for idx, action in zip(active, selected):
                        actions[idx] = int(action)
                evader_actions: List[Optional[int]] = []
                if env.evaders:
                    if hasattr(env, "configure_evader_apf_agents"):
                        env.configure_evader_apf_agents(apf_agents)
                    for idx, evader_obs in enumerate(env.get_evader_observations_for_apf()):
                        evader_actions.append(None if evader_obs is None else int(apf_agents[idx].act(evader_obs)))
                labels = list(getattr(env, "last_task_labels", []))
                phase = "pure_recovery" if scene == "pure_ce" else (
                    "post_capture" if env.evaders and all(e.deactivated for e in env.evaders) else "pre_capture"
                )
                snapshot = build_snapshot(
                    env,
                    scene=scene,
                    phase=phase,
                    step=step,
                    source={
                        "branch": args.source_branch,
                        "commit": args.source_commit,
                        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                        "config_hash": hashlib.sha256(json.dumps(base_cfg, sort_keys=True).encode()).hexdigest(),
                        "seed": seed,
                    },
                    task_labels=labels,
                    post_capture_elapsed=getattr(env, "post_capture_step", 0),
                )
                key = _snapshot_key(snapshot)
                if key not in seen:
                    seen.add(key)
                    snapshots.append(snapshot)
                result = env.step(actions, evader_actions)
                obs_list = result.observations
                if all(result.dones):
                    break

    with dataset_path.open("w", encoding="utf-8") as handle:
        for snapshot in snapshots:
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    counts: Dict[str, int] = {}
    for snapshot in snapshots:
        counts[f"{snapshot['scene']}/{snapshot['phase']}"] = counts.get(f"{snapshot['scene']}/{snapshot['phase']}", 0) + 1
    manifest = {
        "schema_version": 1,
        "dataset_path": str(dataset_path),
        "snapshot_count": len(snapshots),
        "bucket_counts": counts,
        "seeds": seeds,
        "source": {
            "branch": args.source_branch,
            "commit": args.source_commit,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
