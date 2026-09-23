#!/usr/bin/env python3
"""Bounded A1/A2 audit. No formal run(), backend switches, or silent repairs.

Capture checks the real logits entrypoint after its unmodified forward. It
does not install module hooks (which can disable Transformer fused paths).
Replay consumes saved weights and observations, never regenerates a bad row.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/2026-09-22_a1_a2_cuda_root_cause"
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(item):
        if isinstance(item, float) and not math.isfinite(item):
            return "NaN" if math.isnan(item) else "Infinity" if item > 0 else "-Infinity"
        if isinstance(item, dict):
            return {key: encode(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [encode(val) for val in item]
        return item
    path.write_text(json.dumps(encode(value), indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def heartbeat():
    result = {}
    for arm in ("z05", "z07"):
        p = ROOT.parent / "iqn-z-unified-decay-dual-curriculum-20260919-runtime" / arm / "heartbeat.json"
        s = json.loads(p.read_text())
        result[arm] = {k: s.get(k) for k in ("pid", "status", "phase", "stage", "current_step", "target_step", "updated_at")}
        result[arm]["pid_alive"] = Path(f"/proc/{s['pid']}").exists()
        result[arm]["mtime_age_seconds"] = time.time() - p.stat().st_mtime
    return result


def resource(args):
    result = {"heartbeat_before": heartbeat(), "samples": [], "pid": os.getpid()}
    start = time.monotonic()
    for i in range(10):
        rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free,temperature.gpu", "--format=csv,noheader,nounits"], text=True)
        result["samples"].append({"time": time.time(), "gpus": [[int(x.strip()) for x in r.split(",")] for r in rows.splitlines()]})
        time.sleep(max(0, start + i + 1 - time.monotonic()))
    result["heartbeat_after"] = heartbeat()
    result["compute_apps"] = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv"], text=True)
    result["disk"] = subprocess.check_output(["df", "-h", str(OUT)], text=True)
    result["gpu_summary"] = {}
    for index in (0, 1):
        rows = [s["gpus"][index] for s in result["samples"]]
        result["gpu_summary"][index] = {"mean_util": sum(r[1] for r in rows)/10, "peak_util": max(r[1] for r in rows), "min_free_mib": min(r[2] for r in rows)}
    write_json(OUT / f"resource_{args.label}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)


def tensor_info(t):
    import torch
    t = t.detach().cpu()
    finite = torch.isfinite(t)
    good = t[finite]
    return {"shape": list(t.shape), "dtype": str(t.dtype), "finite": bool(finite.all()), "finite_count": int(finite.sum()), "numel": t.numel(), "max_abs_finite": float(good.float().abs().max()) if good.numel() else None}


def mask_info(obs, row):
    import torch
    mask = obs["masks"][row].detach().cpu().bool()
    return {"valid_tokens": int(mask.sum()), "self_valid": bool(mask[0]), "friends_valid": int(mask[1:9].sum()), "enemies_valid": int(mask[9:17].sum()), "obstacles_valid": int(mask[17:].sum()), "all_masked": bool((~mask).all()), "all_values_zero": all(bool((v[row] == 0).all()) for v in obs.values()), "mask": mask.tolist(), "types": obs["types"][row].detach().cpu().tolist()}


def backend_info(device=None):
    import torch
    # A2's existing runner combines precision APIs; the aggregate getter may
    # reject that state even though the forward path is usable. Record the
    # introspection error without changing the process's backend settings.
    try:
        precision = torch.get_float32_matmul_precision()
    except RuntimeError as error:
        precision = {"query_error": str(error)}
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"), "launch_blocking": os.getenv("CUDA_LAUNCH_BLOCKING"), "mha_fastpath": torch.backends.mha.get_fastpath_enabled(), "flash_sdp": torch.backends.cuda.flash_sdp_enabled(), "memory_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(), "math_sdp": torch.backends.cuda.math_sdp_enabled(), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32, "float32_precision": precision, "autocast": torch.is_autocast_enabled(), "cpu_threads": torch.get_num_threads(), "device_name": torch.cuda.get_device_name() if (device is None or str(device).startswith("cuda")) and torch.cuda.is_available() else None}


def capture(args):
    import numpy as np
    import torch
    from cocap_voradj.training.trainer import load_config
    from cocap_voradj.training.shared_local_ac import state_hash
    torch.set_num_threads(1)
    dest = OUT / args.label
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "result.json").exists():
        raise FileExistsError(dest)
    if args.task == "a1":
        source = ROOT.parent / "ac-entropy-cov-20260921"
        load_module("cocap_voradj.training.shared_local_ac", OUT / "source_snapshot/a1/src/cocap_voradj/training/shared_local_ac.py")
        runner_module = load_module("audit_a1_runner", OUT / "source_snapshot/a1/tools/run_shared_local_ac_capability.py")
        config_path = source / "configs/experiments/ac_entropy_localq_cov_20260921/stage1.yaml"
        cls = runner_module.CapabilityRunner
    else:
        if args.source == "formal":
            load_module("cocap_voradj.training.discrete_sac", OUT / "source_snapshot/a2_formal_135eca3/src/cocap_voradj/training/discrete_sac.py")
            runner_module = load_module("audit_a2_runner", OUT / "source_snapshot/a2_formal_135eca3/tools/run_discrete_sac_coverage_20260922.py")
        else:
            from tools import run_discrete_sac_coverage_20260922 as runner_module
        config_path = ROOT / "configs/experiments/ac_capability_20260920/ac_discrete_sac_cov_stage1.yaml"
        cls = runner_module.DiscreteSACCoverageRunner
    cfg = load_config(config_path)
    cfg["device"] = args.device
    cfg["run_name"] = "bounded_runtime"
    # Only output/device differ; formal network, episode length, replay, batch,
    # seed, optimizer, evaluation and scientific config remain intact.
    runner = cls(cfg, config_path, smoke=False, output_root=dest, resume_path=None)
    if runner.device != args.device:
        raise RuntimeError(f"Device fallback forbidden: {runner.device}")
    write_json(dest / "resolved_config.json", cfg)
    context = {"phase": "initial_evaluation", "sample": [], "provenance": {}, "calls": {}}
    result = {"task": args.task, "source": args.source, "pid": os.getpid(), "max_steps": args.steps, "entrypoint": "initial evaluation -> production _collect_step/_maybe_update", "device": runner.device, "backend": backend_info(runner.device), "heartbeat_before": heartbeat(), "instrumentation": "method wrappers; no module hooks; no pre-forward synchronization; logits finite check after exact forward", "network": dataclasses.asdict(runner.network_config)}
    original_add = runner.replay.add
    def add(item):
        original_add(item)
        context["provenance"][id(item)] = {"collection_global_step": runner.global_step, "episode_id": item.episode_id, "agent_id": item.agent_id, "active_before": True, "active_after": not bool(np.all(item.next_obs["masks"] == 0)), "next_obs_source": "inactive_zero_filled" if not item.next_obs["masks"].any() else "environment_observation", "task": "voradj_coverage", "phase": item.phase, "done": item.done, "terminated": item.terminated, "truncated": item.truncated, "replay_class": item.replay_class, "reset_source": "environment_default", "metadata": item.metadata}
    runner.replay.add = add
    original_sample = runner.replay.sample_uniform
    def sample(*a, **kw):
        rows = original_sample(*a, **kw)
        slots = {id(item): i for i, item in enumerate(runner.replay.data)}
        context["sample"] = [{**context["provenance"][id(item)], "replay_slot": slots[id(item)]} for item in rows]
        return rows
    runner.replay.sample_uniform = sample
    original_update = runner.trainer.update
    def update(*a, **kw):
        old = context["phase"]
        context["phase"] = "update"
        try:
            return original_update(*a, **kw)
        finally:
            context["phase"] = old
    runner.trainer.update = update
    class CapturedFailure(Exception):
        pass
    def install(module, name):
        original = module.logits
        def logits(obs):
            values = original(obs)
            call_name = name + (".next_obs_no_grad" if context["phase"] == "update" and not torch.is_grad_enabled() else "." + context["phase"])
            context["calls"][call_name] = context["calls"].get(call_name, 0) + 1
            if not bool(torch.isfinite(values).all()):
                bad = (~torch.isfinite(values)).any(-1).nonzero().flatten().cpu().tolist()
                metadata = context["sample"] if context["phase"] == "update" else context.get("active", [])
                manifest = {"task": args.task, "global_step": runner.global_step, "updates_completed": runner.trainer.update_count, "phase": context["phase"], "entrypoint": call_name, "bad_rows": bad, "logits": tensor_info(values), "rows": [{"batch_index": row, "provenance": metadata[row] if row < len(metadata) else None, "mask_tokens": mask_info(obs, row), "observation": {k: v[row].detach().cpu().tolist() for k, v in obs.items()}} for row in bad], "all_input_finite": all(bool(torch.isfinite(v).all()) for v in obs.values()), "model_hash": state_hash(module), "model_training": module.training, "submodule_training": {k: m.training for k, m in module.named_modules()}, "grad_enabled": torch.is_grad_enabled(), "autocast_enabled": torch.is_autocast_enabled(), "backend": backend_info(runner.device)}
                payload = {"state_dict": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}, "obs": {k: v.detach().cpu().clone() for k, v in obs.items()}, "logits": values.detach().cpu(), "network_config": dataclasses.asdict(module.config), "manifest": manifest, "batch_metadata": metadata, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else []}
                torch.save(payload, dest / "failure.pt")
                write_json(dest / "failing_row_manifest.json", manifest)
                result["failure"] = {k: manifest[k] for k in ("global_step", "updates_completed", "entrypoint", "bad_rows", "logits", "all_input_finite", "model_hash")}
                raise CapturedFailure(str(result["failure"]))
            return values
        module.logits = logits
    install(runner.trainer.actor, "actor")
    if hasattr(runner.trainer, "target_actor"):
        install(runner.trainer.target_actor, "target_actor")
    start = time.monotonic()
    try:
        if not args.skip_initial_eval:
            print("INITIAL_EVAL", args.task, flush=True)
            initial = runner._gate_evaluation(0) if args.task == "a1" else runner._evaluate(0)
            write_json(dest / "initial_evaluation.json", initial)
        context["phase"] = "collection"
        runner._pending_actions = {}
        print("COLLECTION_START", args.task, flush=True)
        for _ in range(args.steps):
            obs = runner.observations["voradj_coverage"] if args.task == "a1" else runner.observations
            context["active"] = [{"agent_id": i, "active": True, "episode_id": runner.episode_id, "task": "voradj_coverage", "phase": "pure_coverage", "reset_source": "environment_default"} for i, o in enumerate(obs) if o is not None]
            if args.task == "a1":
                runner._collect_step("voradj_coverage")
            else:
                runner._collect_step()
            if runner.global_step % 100 == 0:
                print("STEP", runner.global_step, "UPDATES", runner.trainer.update_count, flush=True)
        result["status"] = "PASS_BOUNDED_NO_FAILURE"
    except CapturedFailure:
        result["status"] = "CAPTURED_NONFINITE"
    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    modules = {name: getattr(runner.trainer, name) for name in ("actor", "target_actor", "critic", "target_critic", "critic1", "critic2", "target_critic1", "target_critic2") if hasattr(runner.trainer, name)}
    result["final_module_finite"] = {name: all(bool(torch.isfinite(v).all()) for v in module.state_dict().values()) for name, module in modules.items()}
    result["telemetry_finite"] = all(float(row.get("finite", 0)) == 1 for row in runner.telemetry)
    result["telemetry_last"] = list(runner.telemetry)[-1:]
    result["terminal_zero_rows_collected"] = sum(p["next_obs_source"] == "inactive_zero_filled" for p in context["provenance"].values())
    if result.get("status") == "PASS_BOUNDED_NO_FAILURE" and not (all(result["final_module_finite"].values()) and result["telemetry_finite"]):
        result["status"] = "NONFINITE_TRAINING_STATE"
    result.update({"seconds": time.monotonic()-start, "global_step": runner.global_step, "update_count": runner.trainer.update_count, "calls": context["calls"], "heartbeat_after": heartbeat(), "skip_initial_eval": args.skip_initial_eval})
    write_json(dest / "result.json", result)
    print(json.dumps(result, default=str), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("resource")
    r.add_argument("--label", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--task", choices=["a1", "a2"], required=True)
    c.add_argument("--source", choices=["formal", "current"], default="formal")
    c.add_argument("--device", choices=["cpu", "cuda:0"], required=True)
    c.add_argument("--steps", type=int, default=1000)
    c.add_argument("--label", required=True)
    c.add_argument("--skip-initial-eval", action="store_true")
    args = p.parse_args()
    if args.command == "resource":
        resource(args)
    else:
        if not 1 <= args.steps <= 1500:
            p.error("bounded diagnostic requires 1..1500 steps")
        capture(args)


if __name__ == "__main__":
    main()
