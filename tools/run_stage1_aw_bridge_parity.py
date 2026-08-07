"""Stage 1: legacy IQN discrete (a,w) -> continuous (a,w) bridge parity.

The legacy IQN still selects an integer action; this tool maps the integer
through the exact legacy ``action_list`` and executes the same command through
the continuous ``acceleration_angular_velocity_body`` API.  It then compares
fixed-seed pursuer trajectories, rewards, and episode events between the old
discrete path and the new bridge path.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.trainer import load_config, set_global_config


DEFAULT_CONFIG = ROOT / "configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml"
DEFAULT_CHECKPOINT = ROOT / "artifacts/2026-08-02_three_line_stage_best_20rollout10gif/crms_supportapproach_4v1_s1_step_2000000/step_2000000.pt"
DEFAULT_OUT = ROOT / "artifacts/2026-08-07_positive_feedback_ladder/stage1_aw_bridge_parity"
SCENARIOS = ("capture", "coverage", "mix")


def scenario_config(base: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    env = cfg.setdefault("env", {})
    if scenario == "coverage":
        env["num_evaders"] = 0
        env["pursuer_spawn_mode"] = "inner_random_cluster"
        env["pursuer_spawn_min_sep"] = 7.0
    else:
        env["num_evaders"] = 1
        env["pursuer_spawn_mode"] = "map_random"
        env["evader_spawn_mode"] = "map_random"
    return cfg


def continuous_aw_config(base: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["action_mode"] = "acceleration_angular_velocity_body"
    cfg["a_max"] = 0.4
    cfg["w_max"] = float(np.pi / 6.0)
    cfg["v_max"] = 3.0
    cfg["decision_dt"] = 0.5
    cfg["yaw"] = {"init": "legacy_random"}
    pursuer = cfg.setdefault("pursuer", {})
    pursuer["action_mode"] = "acceleration_angular_velocity_body"
    pursuer["a_max"] = 0.4
    pursuer["w_max"] = float(np.pi / 6.0)
    return cfg


def evader_actions(env: VorAdjEnv, apf_agents: List[ApfAgent]) -> List[Optional[int]]:
    if not env.evaders:
        return []
    if hasattr(env, "configure_evader_apf_agents"):
        env.configure_evader_apf_agents(apf_agents)
    observations = list(env.get_evader_observations_for_apf())
    return [
        None if obs is None else int(apf_agents[idx].act(obs))
        for idx, obs in enumerate(observations)
    ]


def pursuer_state(env: VorAdjEnv) -> Dict[int, List[float]]:
    return {
        idx: [
            float(p.x),
            float(p.y),
            float(p.theta),
            float(p.speed),
            float(p.velocity[0]),
            float(p.velocity[1]),
        ]
        for idx, p in enumerate(env.pursuers)
        if not p.deactivated
    }


def state_diffs(old: Dict[int, List[float]], new: Dict[int, List[float]]) -> Dict[str, float]:
    keys = sorted(set(old).intersection(new))
    arrays = {
        key: np.abs(np.asarray(old[key], dtype=float) - np.asarray(new[key], dtype=float))
        for key in keys
    }
    if not arrays:
        return {}
    stacked = np.stack(list(arrays.values()), axis=0)
    return {
        "n_aligned_agents": len(keys),
        "mean": float(stacked.mean()),
        "p50": float(np.percentile(stacked, 50)),
        "p95": float(np.percentile(stacked, 95)),
        "max": float(stacked.max()),
        "x_max": float(max(v[0] for v in arrays.values())),
        "y_max": float(max(v[1] for v in arrays.values())),
        "theta_max": float(max(v[2] for v in arrays.values())),
        "speed_max": float(max(v[3] for v in arrays.values())),
        "vx_max": float(max(v[4] for v in arrays.values())),
        "vy_max": float(max(v[5] for v in arrays.values())),
    }


def run_pair(
    model: CoCapIQN,
    base_cfg: Dict[str, Any],
    scenario: str,
    seed: int,
    device: str,
    max_steps: int,
) -> Dict[str, Any]:
    old_cfg = scenario_config(base_cfg, scenario)
    new_cfg = continuous_aw_config(old_cfg)
    set_global_config(old_cfg)
    old_env = VorAdjEnv(old_cfg, seed=seed)
    old_env.reset()
    old_apf = [ApfAgent(e.a, e.w) for e in old_env.evaders]
    set_global_config(new_cfg)
    new_env = VorAdjEnv(new_cfg, seed=seed)
    new_env.reset()
    new_apf = [ApfAgent(e.a, e.w) for e in new_env.evaders]

    old_states: List[Dict[int, List[float]]] = []
    new_states: List[Dict[int, List[float]]] = []
    old_rewards: List[List[float]] = []
    new_rewards: List[List[float]] = []
    diff_rows: List[Dict[str, Any]] = []
    action_mapping_errors = 0
    rejected_actions = 0

    for _ in range(int(max_steps)):
        old_obs = list(old_env.get_observations())
        new_obs = list(new_env.get_observations())
        if any(item is None for item in old_obs) or any(item is None for item in new_obs):
            break
        active = [idx for idx, obs in enumerate(old_obs) if obs is not None]
        batch = stack_obs([old_obs[idx] for idx in active], device)
        selected = model.act(
            batch,
            mode="voradj",
            epsilon=0.0,
            deterministic_quantiles=True,
        ).detach().cpu().tolist()
        action_map = old_env.pursuers[0].action_list
        old_actions: List[Optional[int]] = [None] * len(old_obs)
        new_actions: List[Optional[Any]] = [None] * len(new_obs)
        for position, idx in enumerate(active):
            action_index = int(selected[position])
            old_actions[idx] = action_index
            try:
                new_actions[idx] = list(action_map[action_index])
            except (IndexError, TypeError):
                action_mapping_errors += 1
                new_actions[idx] = None
            if new_actions[idx] is not None:
                try:
                    new_env.action_adapter.validate(new_actions[idx])
                except Exception:
                    rejected_actions += 1

        old_evader = evader_actions(old_env, old_apf)
        new_evader = evader_actions(new_env, new_apf)
        old_outcome = old_env.step(old_actions, old_evader)
        new_outcome = new_env.step(new_actions, new_evader)
        old_state = pursuer_state(old_env)
        new_state = pursuer_state(new_env)
        old_states.append(old_state)
        new_states.append(new_state)
        old_rewards.append([float(value) for value in old_outcome.rewards])
        new_rewards.append([float(value) for value in new_outcome.rewards])
        diff = state_diffs(old_state, new_state)
        diff_rows.append({"step": len(old_states), **diff})
        if all(old_outcome.dones) or all(new_outcome.dones):
            break

    old_record = old_env.episode_record(task=scenario)
    new_record = new_env.episode_record(task=scenario)
    reward_diff_abs = [
        abs(float(old_outcome_reward) - float(new_outcome_reward))
        for old_outcome_reward, new_outcome_reward in zip(
            [item for row in old_rewards for item in row],
            [item for row in new_rewards for item in row],
        )
    ]
    reward_diff_rows: List[Dict[str, Any]] = []
    for step_idx, (old_row, new_row) in enumerate(zip(old_rewards, new_rewards)):
        diffs = [abs(float(old_value) - float(new_value)) for old_value, new_value in zip(old_row, new_row)]
        if diffs and max(diffs) >= 1.0:
            reward_diff_rows.append(
                {
                    "step": step_idx + 1,
                    "max_abs_diff": float(max(diffs)),
                    "old_rewards": list(old_row),
                    "new_rewards": list(new_row),
                    "old_states": old_states[step_idx],
                    "new_states": new_states[step_idx],
                }
            )
    reward_diff_rows = sorted(reward_diff_rows, key=lambda item: item["max_abs_diff"], reverse=True)[:5]
    state_key_errors: List[float] = []
    for row in diff_rows:
        if row:
            state_key_errors.append(float(row.get("max", 0.0)))
    return {
        "scenario": scenario,
        "seed": seed,
        "steps": len(old_states),
        "action_mapping_errors": action_mapping_errors,
        "rejected_actions": rejected_actions,
        "state_error": {
            "mean": float(np.mean(state_key_errors)) if state_key_errors else 0.0,
            "p50": float(np.percentile(state_key_errors, 50)) if state_key_errors else 0.0,
            "p95": float(np.percentile(state_key_errors, 95)) if state_key_errors else 0.0,
            "max": float(max(state_key_errors)) if state_key_errors else 0.0,
        },
        "reward_abs_diff": {
            "mean": float(np.mean(reward_diff_abs)) if reward_diff_abs else 0.0,
            "p95": float(np.percentile(reward_diff_abs, 95)) if reward_diff_abs else 0.0,
            "max": float(max(reward_diff_abs)) if reward_diff_abs else 0.0,
        },
        "reward_diff_top": reward_diff_rows,
        "old": {
            "episode_success": bool(old_record.get("episode_success", False)),
            "captured": bool(old_record.get("captured", False)),
            "collision_event": bool(old_record.get("collision_event", False)),
            "length": int(old_record.get("length", 0)),
        },
        "new": {
            "episode_success": bool(new_record.get("episode_success", False)),
            "captured": bool(new_record.get("captured", False)),
            "collision_event": bool(new_record.get("collision_event", False)),
            "length": int(new_record.get("length", 0)),
        },
        "event_parity": (
            bool(old_record.get("captured", False)) == bool(new_record.get("captured", False))
            and bool(old_record.get("collision_event", False)) == bool(new_record.get("collision_event", False))
        ),
    }


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026081201, 2026081202, 2026081203])
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(str(Path(args.config)))
    model = CoCapIQN.load(str(Path(args.checkpoint)), device=args.device)
    model.eval()
    results: List[Dict[str, Any]] = []
    with torch.no_grad():
        for scenario in SCENARIOS:
            for seed in args.seeds:
                result = run_pair(model, base_cfg, scenario, seed, args.device, args.max_steps)
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    state_max = [item["state_error"]["max"] for item in results]
    reward_max = [item["reward_abs_diff"]["max"] for item in results]
    event_parity = sum(bool(item["event_parity"]) for item in results)
    capture_parity = sum(
        bool(item["old"]["captured"]) == bool(item["new"]["captured"]) for item in results
    )
    collision_parity = sum(
        bool(item["old"]["collision_event"]) == bool(item["new"]["collision_event"]) for item in results
    )
    report = {
        "kind": "stage1_aw_bridge_parity",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "max_steps": int(args.max_steps),
        "seeds": list(args.seeds),
        "scenarios": list(SCENARIOS),
        "runs": len(results),
        "state_error_max": {
            "mean": float(np.mean(state_max)) if state_max else 0.0,
            "p50": percentile(state_max, 50),
            "p95": percentile(state_max, 95),
            "max": float(max(state_max)) if state_max else 0.0,
        },
        "reward_abs_diff_max": {
            "mean": float(np.mean(reward_max)) if reward_max else 0.0,
            "p95": percentile(reward_max, 95),
            "max": float(max(reward_max)) if reward_max else 0.0,
        },
        "event_parity_rate": event_parity / max(len(results), 1),
        "capture_parity_rate": capture_parity / max(len(results), 1),
        "collision_parity_rate": collision_parity / max(len(results), 1),
        "gate_passed": (
            len(results) > 0
            and event_parity == len(results)
            and (float(np.mean(state_max)) if state_max else float("inf")) < 5.0
        ),
        "results": results,
    }
    report_path = out_root / "parity_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# Stage 1 AW Bridge Parity Report",
        "",
        f"- config: `{args.config}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- seeds: {list(args.seeds)}",
        f"- max steps: {args.max_steps}",
        "",
        "## Gate",
        "",
        f"- event parity rate: {report['event_parity_rate']:.3f}",
        f"- capture parity rate: {report['capture_parity_rate']:.3f}",
        f"- collision parity rate: {report['collision_parity_rate']:.3f}",
        f"- state error max mean: {report['state_error_max']['mean']:.4f}",
        f"- state error max p95: {report['state_error_max']['p95']:.4f}",
        f"- reward abs diff max mean: {report['reward_abs_diff_max']['mean']:.4f}",
        f"- gate_passed: {report['gate_passed']}",
        "",
        "## Notes",
        "",
        "- Action indices are identical by construction (legacy IQN -> exact legacy action list -> continuous API).",
        "- Old discrete path integrates position explicitly then speed/theta; bridge path uses the 10-substep trapezoidal AW integrator, so small numerical drift is expected and must not change event-level behavior.",
        "",
    ]
    (out_root / "parity_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
