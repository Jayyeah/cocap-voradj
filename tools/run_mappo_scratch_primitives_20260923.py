#!/usr/bin/env python3
"""Run the isolated MAPPO Pure-Coverage and capability-first Pure-Capture lines."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from cocap_voradj.envs.density_sensing import runtime_metadata
from cocap_voradj.training.numeric_validation import assert_finite_numeric_tree
from tools import forward_final_single_task_20260915 as single_task
from tools import run_normsense_pure_capture_v2_20260916 as normsense
from tools import preflight_forward_final_scratch_20260914 as scratch
from tools import train_forward_final_ppo_20260909 as production


SCHEMA = "cocap-mappo-scratch-primitives-20260923-v1"
SEED = 2026091501
EVAL_SEED_BASE = 2026191501
TASKS = {
    "coverage": {"budget": 200_000, "label": "M-COV", "sensing": "legacy-r20"},
    "capture": {"budget": 500_000, "label": "M-CAP", "sensing": "normsense-v2"},
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__),
        Path(single_task.__file__),
        Path(normsense.__file__),
        Path(normsense.legacy.__file__),
        Path(production.__file__),
        Path(scratch.__file__),
        ROOT / "src/cocap_voradj/models/continuous/local_entity_token_encoder.py",
        ROOT / "src/cocap_voradj/models/small_step_ac.py",
        ROOT / "src/cocap_voradj/training/small_step_ac.py",
        ROOT / "src/cocap_voradj/training/continuous/central_schema.py",
        ROOT / "src/cocap_voradj/training/forward_final.py",
        ROOT / "src/cocap_voradj/training/forward_final_v2.py",
        ROOT / "src/cocap_voradj/training/numeric_validation.py",
        ROOT / "src/cocap_voradj/envs/density_sensing.py",
        ROOT / "src/cocap_voradj/envs/voronoi_adjacency.py",
        ROOT / "configs/experiments/forward_final_scratch_mappo_20260914/canonical.yaml",
        ROOT / "configs/experiments/forward_final_single_task_20260915/common.yaml",
        ROOT / "configs/experiments/forward_final_normsense_v2_20260915/NormSense-PureCapture.yaml",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(set(paths))}


def trainer_for(task: str, device: str):
    contract = scratch.load_contract()
    if task == "capture":
        # The official NormSense runner changes no learner field; this is the
        # same random-init contract as M-COV, including dropout=.1.
        contract = normsense.trainer_contract("baseline")
    scratch.seed_all(SEED)
    trainer = scratch.make_trainer(contract, SEED, device)
    return trainer, contract


def stream_for(task: str):
    if task == "capture":
        return normsense.PureCaptureStream("baseline", SEED)
    return single_task.SingleTaskStream("coverage", SEED)


def validate_task(task: str, stream, trainer) -> dict:
    if task == "capture":
        runtime = normsense.validate_runtime("baseline", stream.env, trainer.actor)
        sensing = runtime["sensing"]
        return {
            "runtime": runtime,
            "sensing": sensing,
            "task_contract": "CAPABILITY_FIRST_CAPTURE_BASELINE",
        }
    runtime = single_task.check_env(stream.env, "coverage", trainer.actor)
    assert stream.env._vct_ls_sensing_radius("enemy") == 20.0
    assert stream.env._vct_ls_sensing_radius("obstacle") == 20.0
    return {"runtime": runtime, "sensing": {"policy": "legacy-r20", "resolved_onboard_radius": 20.0}}


def initial_hashes(trainer) -> dict[str, str]:
    return {
        "actor": scratch.tensor_hash(trainer.actor.state_dict()),
        "value": scratch.tensor_hash(trainer.value.state_dict()),
        "value_norm": normsense.legacy.value_norm_hash(trainer.value_norm),
    }


def capture_sensing_record(runtime: dict) -> dict:
    sensing = runtime["sensing"]
    raw = float(sensing["k"] * np.sqrt(float(sensing["free_space_area"]) / float(sensing["num_pursuers"])))
    resolved = float(sensing["resolved_onboard_radius"])
    record = {
        "schema": "cocap-capture-sensing-resolution-v1",
        "A_eff": float(sensing["free_space_area"]),
        "k": float(sensing["k"]),
        "raw_R": raw,
        "resolved_R": resolved,
        "resolved_onboard_radius": resolved,
        "floor": float(sensing["radius_floor"]),
        "map_scale": 1.0,
        "map_size": sensing["map_size"],
        "num_pursuers": int(sensing["num_pursuers"]),
        "free_mask_sha256": sensing["free_mask_sha256"],
        "surface_distance_semantics": sensing["distance_semantics"],
        "friendly_information": sensing["friendly_information"],
        "resolver": "runtime_metadata -> resolve_density_normalized_onboard_radius",
    }
    np.testing.assert_allclose(raw, resolved, rtol=0.0, atol=1e-10)
    assert record["floor"] == 20.0 and record["map_scale"] == 1.0
    return record


def episode_censoring(report: dict) -> dict:
    records = report.get("records", [])
    if not records:
        return {"episodes": 0}
    captured = [bool(row.get("captured")) for row in records]
    times = [row["capture_seconds"] for row in records if row.get("capture_seconds") is not None]
    return {
        "episodes": len(records),
        "capture_completed": int(sum(captured)),
        "capture_censored": int(len(records) - sum(captured)),
        "censoring_fraction": float(1.0 - np.mean(captured)),
        "capture_time_observed_episodes": len(times),
        "capture_time_is_not_imputed_for_censored": True,
    }


def evaluate_formal(task: str, trainer, out: Path, step: int) -> dict:
    if task == "capture":
        report = normsense.evaluate(trainer.actor, "baseline", step, out, 20, EVAL_SEED_BASE)
        report["episode_censoring"] = {
            mode: episode_censoring({"records": [row for row in report["records"] if row["mode"] == mode]})
            for mode in ("argmax", "sample")
        }
    else:
        report = single_task.evaluate(trainer.actor, "coverage", out, step, episodes=20, seed_base=EVAL_SEED_BASE)
        report["episode_censoring"] = {"coverage": "not_applicable"}
    write_json(out / f"formal_step_{step:06d}.json", {
        "schema": "cocap-formal-eval-index-v1",
        "task": task,
        "step": step,
        "report": str(out / f"eval_step_{step:06d}.json"),
        "summary": report.get("summary", {}),
        "episode_censoring": report["episode_censoring"],
        "matched_seed_base": EVAL_SEED_BASE,
        "episodes_per_mode": 20,
    })
    return report


def save_latest(out: Path, task: str, step: int, trainer, stream, launch: dict, rollout: dict, metrics: dict) -> None:
    payload = {
        "schema": SCHEMA,
        "task": task,
        "step": int(step),
        "launch": launch,
        "trainer": trainer.state_dict(),
        "stream_state": stream.__dict__,
        "rollout": rollout,
        "metrics": metrics,
        "rng": scratch.rng_state(trainer.device),
    }
    temporary = out / "latest.pt.tmp"
    latest = out / "latest.pt"
    torch.save(payload, temporary)
    check = torch.load(temporary, map_location="cpu", weights_only=False)
    assert check["schema"] == SCHEMA and check["task"] == task and check["step"] == int(step)
    os.replace(temporary, latest)
    write_json(out / "latest.json", {
        "schema": "cocap-latest-checkpoint-v1",
        "task": task,
        "step": int(step),
        "checkpoint": str(latest),
        "checkpoint_sha256": sha256_file(latest),
        "partial_rollout_steps": len(rollout["rewards"]),
        "latest_only": True,
    })


def load_latest(path: Path, task: str, device: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema"] == SCHEMA and payload["task"] == task
    assert payload["launch"]["source_sha256"] == source_hashes()
    trainer, _ = trainer_for(task, device)
    trainer.load_state_dict(payload["trainer"])
    stream_cls = normsense.PureCaptureStream if task == "capture" else single_task.SingleTaskStream
    stream = stream_cls.__new__(stream_cls)
    stream.__dict__.update(payload["stream_state"])
    if task == "capture":
        normsense.validate_runtime("baseline", stream.env, trainer.actor)
    else:
        single_task.check_env(stream.env, "coverage", trainer.actor)
    scratch.restore_rng(payload["rng"], trainer.device)
    return trainer, stream, payload


def run(task: str, out: Path, device: str, resume: Path | None = None, smoke: bool = False) -> None:
    spec = TASKS[task]
    if resume is None:
        if out.exists():
            raise RuntimeError(f"fresh output required: {out}")
        out.mkdir(parents=True)
        trainer, contract = trainer_for(task, device)
        stream = stream_for(task)
        runtime = validate_task(task, stream, trainer)
        hashes = initial_hashes(trainer)
        launch = {
            "schema": SCHEMA,
            "label": spec["label"],
            "task": task,
            "scientific_name": "CAPABILITY_FIRST_CAPTURE_BASELINE" if task == "capture" else "PURE_COVERAGE_POSITIVE_CONTROL",
            "seed": SEED,
            "budget": 256 if smoke else spec["budget"],
            "formal_checkpoints": [0] if smoke else list(range(0, spec["budget"] + 1, 25_000)),
            "eval_episodes_per_mode": 1 if smoke else 20,
            "eval_modes": ["argmax", "sample"],
            "initialization": "random-init",
            "teacher_dependency": 0,
            "architecture_unchanged": True,
            "contract_parent": "2026-09-15 Pure-Coverage corrected scratch MAPPO",
            "ppo": contract["ppo"],
            "runtime": runtime,
            "initial_hashes": hashes,
            "source_sha256": source_hashes(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": device,
            "latest_only_checkpoints": True,
            "metric_early_stop": False,
        }
        write_json(out / "launch.json", launch)
        if task == "capture":
            sensing = capture_sensing_record(runtime)
            write_json(out / "resolved_capture_sensing.json", sensing)
            launch["resolved_capture_sensing"] = sensing
            write_json(out / "launch.json", launch)
        rollout = production.empty_rollout()
        metrics = {}
        step = 0
        save_latest(out, task, step, trainer, stream, launch, rollout, metrics)
        if not smoke:
            evaluate_formal(task, trainer, out, 0)
    else:
        trainer, stream, payload = load_latest(resume, task, device)
        launch = payload["launch"]
        rollout = payload["rollout"]
        metrics = payload["metrics"]
        step = int(payload["step"])
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "resume.json", {
            "status": "PASS_LATEST_RESUME_LOAD",
            "task": task,
            "step": step,
            "checkpoint": str(resume),
            "checkpoint_sha256": sha256_file(resume),
        })

    budget = int(launch["budget"])
    cadence = 25_000
    started = time.monotonic()
    start_step = step

    def progress(status: str) -> None:
        elapsed = time.monotonic() - started
        speed = (step - start_step) / elapsed if elapsed else 0.0
        write_json(out / "progress.json", {
            "schema": "cocap-progress-v1",
            "status": status,
            "task": task,
            "pid": os.getpid(),
            "step": step,
            "budget": budget,
            "training_updates": trainer.update_count,
            "steps_per_second": speed,
            "eta_seconds": (budget - step) / speed if speed else None,
            "partial_rollout_steps": len(rollout["rewards"]),
            "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "latest_checkpoint": str(out / "latest.pt"),
            "last_update": metrics,
        })

    progress("RUNNING")
    while step < budget:
        stream.set_clock(step)
        row, episode = production.collect_transition(trainer, stream)
        assert_finite_numeric_tree(row, path=f"transition[{step}]")
        for key, value in row.items():
            rollout[key].append(value)
        step += 1
        if episode is not None:
            with (out / "episodes.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, **episode}, ensure_ascii=False, allow_nan=False) + "\n")
        if len(rollout["rewards"]) == 256 or step == budget:
            metrics = normsense.legacy.update_one(
                trainer, rollout, step, "pure_capture" if task == "capture" else "coverage"
            )
            assert_finite_numeric_tree(metrics, path=f"update[{step}]")
            with (out / "learning.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + "\n")
            rollout = production.empty_rollout()
        if step % 100 == 0 or step == budget:
            progress("RUNNING" if step < budget else "COMPLETE_BUDGET")
        if step % cadence == 0 or step == budget:
            save_latest(out, task, step, trainer, stream, launch, rollout, metrics)
            if not smoke:
                evaluate_formal(task, trainer, out, step)
            write_json(out / f"review_{step:06d}.json", {
                "task": task,
                "step": step,
                "review_only": True,
                "continue_to": budget,
                "metric_early_stop": False,
                "last_update": metrics,
            })
    progress("COMPLETE_BUDGET")
    write_json(out / "report.json", {
        "schema": "cocap-report-v1",
        "status": "COMPLETE_BUDGET",
        "task": task,
        "step": step,
        "budget": budget,
        "final_checkpoint": str(out / "latest.pt"),
        "latest_only": True,
        "first_meaningful_signal": None,
        "metric_early_stop": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.resume:
        parser.error("--smoke and --resume are mutually exclusive")
    torch.set_num_threads(1)
    output = args.output.resolve()
    try:
        # Fail closed on IQN/BC/old artifact reads for both primitives.
        scratch.forbid_teacher_dependencies(output)
        run(args.task, output, args.device, args.resume, args.smoke)
    except BaseException:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "failure.json", {
            "schema": "cocap-failure-v1",
            "status": "STOP_IMPLEMENTATION",
            "task": args.task,
            "traceback": __import__("traceback").format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
