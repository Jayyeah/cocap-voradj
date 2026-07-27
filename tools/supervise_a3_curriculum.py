#!/usr/bin/env python3
"""A3 APF-v2 + CenterSqrtN curriculum supervisor.

Policy requested 2026-07-23:
- stage checks begin at 300k and repeat every 100k, except stage1 scratch begins checks at 1M;
- early promotion/truncation is allowed only at or after 800k;
- if no >=800k checkpoint passes, including the case where only a pre-800k
  checkpoint passes, train the stage to cap and select the historical best tested
  checkpoint;
- launch the selected checkpoint's 20-rollout/10-GIF diagnostic asynchronously under artifacts/2026-07-23_a3_apfnew_sqrtn_curriculum;
- immediately start the next stage from the selected checkpoint;
- all training, screening and formal rollout use cuda:1.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
MILESTONE_ROOT = ROOT / "artifacts"
A3_ROOT = MILESTONE_ROOT / "2026-07-23_a3_apfnew_sqrtn_curriculum"
SCREEN_ROOT = A3_ROOT / "screening"
BEST_ROOT = A3_ROOT / "best_20rollout10gif"
ROLLOUT_SCRIPT = ROOT / "tools/batch_rollouts_parallel.py"
STATUS = LOG_DIR / "voradj_a3_apfnew_sqrtn_curriculum_20260723_status.json"

DEVICE = "cuda:1"
POLL_SECONDS = 60
CHECK_START = 300_000
STAGE1_CHECK_START = 1_000_000
CHECK_INTERVAL = 100_000
EARLY_PROMOTE_START = 800_000
CHECK_EPISODES = 12
CHECK_WORKERS = 4
CHECK_COVERAGE_MAX_STEPS = 500
FORMAL_EPISODES = 20
FORMAL_GIF_COUNT = 10
FORMAL_WORKERS = 4

STAGES: list[dict[str, Any]] = [
    {
        "stage": 1,
        "label": "4p1e1obs",
        "pursuers": 4,
        "evaders": 1,
        "obstacles": 1,
        "timesteps": 2_000_000,
        "settle_window": 50,
        "coverage_max_steps": 900,
        "seed": 2026072341,
        "config": "configs/experiments/voradj_a3_apfnew_sqrtn_20260723/a3_apfnew_sqrtn_4v1_scratch_mix_2m.yaml",
        "run_name": "voradj_a3_apfnew_sqrtn_4v1_scratch_mix_2m_20260723_run1",
        "scratch": True,
        "check_start": 1_000_000,
    },
    {"stage": 2, "label": "6p2e2obs", "pursuers": 6, "evaders": 2, "obstacles": 2, "timesteps": 1_200_000, "settle_window": 60, "coverage_max_steps": 1100, "seed": 2026072342, "config": "configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage2_6p2e2obs_1200k.yaml"},
    {"stage": 3, "label": "8p2e2obs", "pursuers": 8, "evaders": 2, "obstacles": 2, "timesteps": 1_200_000, "settle_window": 70, "coverage_max_steps": 1200, "seed": 2026072343, "config": "configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage3_8p2e2obs_1200k.yaml"},
    {"stage": 4, "label": "10p3e3obs", "pursuers": 10, "evaders": 3, "obstacles": 3, "timesteps": 1_400_000, "settle_window": 80, "coverage_max_steps": 1400, "seed": 2026072344, "config": "configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage4_10p3e3obs_1400k.yaml"},
    {"stage": 5, "label": "12p3e3obs", "pursuers": 12, "evaders": 3, "obstacles": 3, "timesteps": 1_400_000, "settle_window": 90, "coverage_max_steps": 1600, "seed": 2026072345, "config": "configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage5_12p3e3obs_1400k.yaml"},
    {"stage": 6, "label": "14p4e4obs", "pursuers": 14, "evaders": 4, "obstacles": 4, "timesteps": 1_600_000, "settle_window": 100, "coverage_max_steps": 1800, "seed": 2026072346, "config": "configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage6_14p4e4obs_1600k.yaml"},
    {
        "stage": 7,
        "label": "16p4e4obs",
        "pursuers": 16,
        "evaders": 4,
        "obstacles": 4,
        "timesteps": 1_150_000,
        "settle_window": 110,
        "coverage_max_steps": 2000,
        "seed": 2026072347,
        "config": "configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage7_16p4e4obs_resume_from450k_remaining1150k_20260727.yaml",
        "run_name": "voradj_a3_apfnew_sqrtn_curr_stage7_16p4e4obs_resume_from450k_remaining1150k_20260727_run2",
        "scratch": True,
        "check_start": 100_000,
    },
    {"stage": 8, "label": "18p5e5obs", "pursuers": 18, "evaders": 5, "obstacles": 5, "timesteps": 1_800_000, "settle_window": 120, "coverage_max_steps": 2200, "seed": 2026072348, "config": "configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage8_18p5e5obs_1800k.yaml"},
    {"stage": 9, "label": "20p5e5obs", "pursuers": 20, "evaders": 5, "obstacles": 5, "timesteps": 2_000_000, "settle_window": 130, "coverage_max_steps": 2400, "seed": 2026072349, "config": "configs/experiments/voradj_a3_apfnew_sqrtn_curriculum_20260723/stage9_20p5e5obs_2000k.yaml"},
]

PASS_THRESHOLDS = {
    "mix_capture_rate": 0.98,
    "capture_success_rate": 0.98,
    "mix_settled_rate": 0.85,
    "coverage_settled_rate": 0.90,
    "mix_geometric_rate": 0.95,
    "coverage_geometric_rate": 0.95,
    "max_collision_rate": 0.10,
}
POSITIVE_KEYS = [
    "mix_capture_rate",
    "capture_success_rate",
    "mix_geometric_rate",
    "mix_settled_rate",
    "coverage_geometric_rate",
    "coverage_settled_rate",
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_status(state: str, **extra: object) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": now(), "state": state, **extra}
    STATUS.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def config_path(stage: dict[str, Any]) -> Path:
    return ROOT / str(stage["config"])


def set_stage_pretrained(stage: dict[str, Any], checkpoint: Path | None) -> dict[str, Any]:
    cfg_path = config_path(stage)
    cfg = load_yaml(cfg_path)
    if checkpoint is not None:
        checkpoint_rel = rel(checkpoint)
        cfg.setdefault("pretrained", {})["path"] = checkpoint_rel
        for key in ("metadata", "experiment_metadata"):
            cfg.setdefault(key, {})["pretrained_checkpoint"] = checkpoint_rel
            cfg.setdefault(key, {})["selected_from_previous_stage"] = True
    write_yaml(cfg_path, cfg)
    return cfg


def run_name_for(stage: dict[str, Any], cfg: dict[str, Any]) -> str:
    return str(stage.get("run_name") or cfg["run_name"])


def run_dir_for(stage: dict[str, Any], cfg: dict[str, Any]) -> Path:
    return ROOT / cfg.get("output_root", "runs") / run_name_for(stage, cfg)


def checkpoint_dir(run_dir: Path) -> Path:
    return run_dir / "checkpoints"


def checkpoint_steps(run_dir: Path, total_steps: int) -> list[tuple[int, Path]]:
    ckpts: list[tuple[int, Path]] = []
    ckpt_dir = checkpoint_dir(run_dir)
    if not ckpt_dir.is_dir():
        return ckpts
    for path in ckpt_dir.glob("step_*.pt"):
        try:
            step = int(path.stem.removeprefix("step_"))
        except ValueError:
            continue
        if step <= total_steps:
            ckpts.append((step, path))
    final = ckpt_dir / f"final_step_{total_steps}.pt"
    if final.is_file():
        ckpts.append((total_steps, final))
    return sorted(ckpts, key=lambda item: (item[0], item[1].name.startswith("final_step_")))


def latest_checkpoint(run_dir: Path, total_steps: int) -> tuple[int, Path] | None:
    ckpts = checkpoint_steps(run_dir, total_steps)
    return ckpts[-1] if ckpts else None


def final_checkpoint_path(run_dir: Path, total_steps: int) -> Path:
    return checkpoint_dir(run_dir) / f"final_step_{total_steps}.pt"


def stage_check_start(stage: dict[str, Any]) -> int:
    return int(stage.get("check_start", CHECK_START))


def should_check(stage: dict[str, Any], step: int, total_steps: int) -> bool:
    check_start = stage_check_start(stage)
    if step < check_start:
        return False
    if step >= total_steps:
        return True
    return (step - check_start) % CHECK_INTERVAL == 0


def scalar(summary: dict[str, Any], key: str) -> float:
    try:
        value = float(summary.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if value == value else 0.0


def score_payload(metrics: dict[str, Any], step: int) -> dict[str, Any]:
    normalized: dict[str, float] = {}
    for key in POSITIVE_KEYS:
        normalized[key] = float(metrics.get(key, 0.0)) / max(float(PASS_THRESHOLDS[key]), 1e-9)
    collision = float(metrics.get("max_collision_rate", 1.0))
    collision_margin = (float(PASS_THRESHOLDS["max_collision_rate"]) - collision) / max(float(PASS_THRESHOLDS["max_collision_rate"]), 1e-9)
    capture_steps = float(metrics.get("capture_avg_steps", 9999.0))
    mix_steps = float(metrics.get("mix_avg_steps", 9999.0))
    coverage_steps = float(metrics.get("coverage_avg_steps", 9999.0))
    score_tuple = [
        min(normalized.values()) if normalized else 0.0,
        sum(normalized.values()) / max(len(normalized), 1),
        float(metrics.get("mix_settled_rate", 0.0)),
        float(metrics.get("coverage_settled_rate", 0.0)),
        float(metrics.get("mix_geometric_rate", 0.0)),
        float(metrics.get("coverage_geometric_rate", 0.0)),
        float(metrics.get("mix_capture_rate", 0.0)),
        float(metrics.get("capture_success_rate", 0.0)),
        collision_margin,
        -capture_steps / 700.0,
        -mix_steps / 1400.0,
        -coverage_steps / max(CHECK_COVERAGE_MAX_STEPS, 1),
        int(step),
    ]
    return {
        "score_tuple": score_tuple,
        "normalized_threshold_ratios": normalized,
        "collision_margin_ratio": collision_margin,
        "step_efficiency_terms": {
            "capture_avg_steps": capture_steps,
            "mix_avg_steps": mix_steps,
            "coverage_avg_steps": coverage_steps,
        },
        "policy": "primary success/settle/geometric/collision ranking; lower capture/mix/coverage avg_steps are late tie-breakers.",
    }


def passed(metrics: dict[str, Any]) -> bool:
    for key, threshold in PASS_THRESHOLDS.items():
        value = float(metrics.get(key, 0.0))
        if key == "max_collision_rate":
            if value > float(threshold):
                return False
        elif value < float(threshold):
            return False
    return True


def stage_screen_root(stage: dict[str, Any]) -> Path:
    return SCREEN_ROOT / f"stage{stage['stage']}_{stage['label']}"


def evaluate_checkpoint(stage: dict[str, Any], cfg_path: Path, checkpoint: Path, step: int) -> dict[str, Any]:
    out = stage_screen_root(stage) / f"check_step_{step}"
    summary_path = out / "all_summaries.json"
    log_path = LOG_DIR / f"check_a3_curr_stage{stage['stage']}_{stage['label']}_step_{step}_20260723.log"
    if not summary_path.is_file():
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROLLOUT_SCRIPT),
            "--config", str(cfg_path),
            "--checkpoint", str(checkpoint),
            "--output-root", str(out),
            "--episodes", str(CHECK_EPISODES),
            "--gif-count", "0",
            "--scenarios", "capture", "coverage", "mix",
            "--seed", str(int(stage["seed"]) + step),
            "--device", DEVICE,
            "--max-steps", str(max(1200, int(stage["coverage_max_steps"]) + 700)),
            "--capture-max-steps", "700",
            "--coverage-max-steps", str(min(int(stage["coverage_max_steps"]), CHECK_COVERAGE_MAX_STEPS)),
            "--capture-evaders", str(int(stage["evaders"])),
            "--max-gif-frames", "1",
            "--workers", str(CHECK_WORKERS),
        ]
        write_status("screening_running", stage=stage, checkpoint=rel(checkpoint), step=step, output_root=rel(out), log=rel(log_path))
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"A3 curriculum check failed stage={stage['stage']} step={step} code={result.returncode}; log={log_path}")
    summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    capture = summaries.get("capture", {}) or {}
    coverage = summaries.get("coverage", {}) or {}
    mix = summaries.get("mix", {}) or {}
    metrics = {
        "capture_success_rate": scalar(capture, "capture_success_rate"),
        "capture_collision_rate": scalar(capture, "collision_rate"),
        "capture_avg_steps": scalar(capture, "avg_steps"),
        "mix_capture_rate": scalar(mix, "capture_success_rate"),
        "mix_geometric_rate": scalar(mix, "coverage_geometric_rate"),
        "mix_settled_rate": scalar(mix, "coverage_settled_rate"),
        "mix_collision_rate": scalar(mix, "collision_rate"),
        "mix_avg_steps": scalar(mix, "avg_steps"),
        "coverage_geometric_rate": scalar(coverage, "coverage_geometric_rate"),
        "coverage_settled_rate": scalar(coverage, "coverage_settled_rate"),
        "coverage_collision_rate": scalar(coverage, "collision_rate"),
        "coverage_avg_steps": scalar(coverage, "avg_steps"),
    }
    metrics["max_collision_rate"] = max(metrics["capture_collision_rate"], metrics["mix_collision_rate"], metrics["coverage_collision_rate"])
    payload = {
        "series": "a3_apfnew_sqrtn_curriculum",
        "stage": stage,
        "checked_at": now(),
        "checkpoint": rel(checkpoint),
        "checkpoint_step": step,
        "output_root": rel(out),
        "summaries": summaries,
        "metrics": metrics,
        "thresholds": PASS_THRESHOLDS,
        "passed": passed(metrics),
        "eligible_for_early_promotion": step >= EARLY_PROMOTE_START,
        "selection_score": score_payload(metrics, step),
        "check_policy": f"12 episodes each for capture/coverage/mix, no GIF; this stage checks from {stage_check_start(stage)} every 100k; early promotion only at/after 800k; coverage cap 500 steps.",
    }
    stage_screen_root(stage).mkdir(parents=True, exist_ok=True)
    (stage_screen_root(stage) / f"check_step_{step}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_existing_payloads(stage: dict[str, Any]) -> tuple[set[int], list[dict[str, Any]]]:
    checked: set[int] = set()
    payloads: list[dict[str, Any]] = []
    root = stage_screen_root(stage)
    if root.is_dir():
        for path in sorted(root.glob("check_step_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                step = int(payload.get("checkpoint_step", path.stem.removeprefix("check_step_")))
            except Exception:
                continue
            checked.add(step)
            payloads.append(payload)
    return checked, payloads


def payload_key(payload: dict[str, Any]) -> list[Any]:
    score = payload.get("selection_score") or {}
    if isinstance(score, dict) and isinstance(score.get("score_tuple"), list):
        return list(score["score_tuple"])
    return score_payload(payload.get("metrics", {}) or {}, int(payload.get("checkpoint_step", 0)))["score_tuple"]


def best_payload(payloads: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not payloads:
        return None
    return max(payloads, key=payload_key)


def tmux_has_session(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def backup_formal_rollout_inputs(out: Path, cfg_path: Path, checkpoint: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    backups: dict[str, str] = {}
    for label, src in (("config_backup", cfg_path), ("checkpoint_backup", checkpoint)):
        dst = out / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        backups[label] = rel(dst)
    return backups


def launch_formal_rollout(stage: dict[str, Any], cfg_path: Path, checkpoint: Path, checkpoint_id: str) -> dict[str, Any]:
    out = BEST_ROOT / f"stage{stage['stage']}_{stage['label']}_{checkpoint_id}"
    summary = out / "all_summaries.json"
    sess = f"voradj_a3_curr_s{stage['stage']}_{checkpoint_id}_rollout_gpu1_0723"
    log_path = LOG_DIR / f"rollout_a3_curr_stage{stage['stage']}_{checkpoint_id}_20260723.log"
    backups = backup_formal_rollout_inputs(out, cfg_path, checkpoint)
    if summary.is_file():
        return {"action": "already_complete", "output_root": rel(out), "checkpoint": rel(checkpoint), **backups}
    if tmux_has_session(sess):
        return {"action": "already_running", "session": sess, "output_root": rel(out), "checkpoint": rel(checkpoint), **backups}
    cmd = [
        sys.executable,
        str(ROLLOUT_SCRIPT),
        "--config", str(cfg_path),
        "--checkpoint", str(checkpoint),
        "--output-root", str(out),
        "--episodes", str(FORMAL_EPISODES),
        "--gif-count", str(FORMAL_GIF_COUNT),
        "--scenarios", "capture", "coverage", "mix",
        "--seed", str(int(stage["seed"]) + 99_000),
        "--device", DEVICE,
        "--max-steps", str(max(1200, int(stage["coverage_max_steps"]) + 700)),
        "--capture-max-steps", "700",
        "--coverage-max-steps", str(int(stage["coverage_max_steps"])),
        "--capture-evaders", str(int(stage["evaders"])),
        "--max-gif-frames", "1000",
        "--frame-duration-ms", "100",
        "--workers", str(FORMAL_WORKERS),
    ]
    launch = f"cd {shlex.quote(str(ROOT))} && {shlex.join(cmd)} >> {shlex.quote(str(log_path))} 2>&1"
    log_path.write_text(f"[{now()}] launching A3 curriculum formal rollout\n", encoding="utf-8")
    subprocess.run(["tmux", "new-session", "-d", "-s", sess, launch], cwd=ROOT, check=True)
    return {"action": "launched", "session": sess, "output_root": rel(out), "checkpoint": rel(checkpoint), "log": rel(log_path), "workers": FORMAL_WORKERS, **backups}


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def terminate_pid(pid: int) -> None:
    if not pid_alive(pid):
        return
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 30
    while time.time() < deadline:
        if not pid_alive(pid):
            return
        time.sleep(0.5)
    if pid_alive(pid):
        os.kill(pid, signal.SIGKILL)


def terminate_proc(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def find_existing_training_pid(cfg_path: Path) -> int | None:
    result = subprocess.run(["pgrep", "-af", f"train.py --config {cfg_path}"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False)
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid != os.getpid() and "train.py" in parts[1] and str(cfg_path) in parts[1]:
            return pid
    return None


def maybe_handoff_path(stage: dict[str, Any]) -> Path:
    return A3_ROOT / f"stage{stage['stage']}_{stage['label']}_handoff.json"


def select_and_record(stage: dict[str, Any], cfg_path: Path, payload: dict[str, Any], reason: str) -> Path:
    checkpoint = ROOT / str(payload["checkpoint"])
    checkpoint_id = f"step_{payload['checkpoint_step']}"
    if checkpoint.name.startswith("final_step_"):
        checkpoint_id = "final"
    rollout = launch_formal_rollout(stage, cfg_path, checkpoint, checkpoint_id)
    handoff = {
        "series": "a3_apfnew_sqrtn_curriculum",
        "stage": stage,
        "selected_at": now(),
        "selected_checkpoint": rel(checkpoint),
        "selected_candidate_id": checkpoint_id,
        "selected_metrics": payload["metrics"],
        "selected_payload": payload,
        "formal_rollout": rollout,
        "selection_reason": reason,
        "policy": "stage1 scratch checks start at 1M; later stages check from 300k every 100k; early promotion only at/after 800k; otherwise train to cap and select historical best; launch 20rollout10gif asynchronously, then continue.",
    }
    path = maybe_handoff_path(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(handoff, indent=2, ensure_ascii=False), encoding="utf-8")
    write_status("stage_handoff", stage=stage, selected_checkpoint=rel(checkpoint), selected_candidate_id=checkpoint_id, selected_metrics=payload["metrics"], selection_reason=reason, formal_rollout=rollout)
    return checkpoint


def run_stage(stage: dict[str, Any], pretrained: Path | None) -> Path:
    cfg = set_stage_pretrained(stage, None if bool(stage.get("scratch")) else pretrained)
    cfg_path = config_path(stage)
    run_dir = run_dir_for(stage, cfg)
    total_steps = int(stage["timesteps"])
    handoff_path = maybe_handoff_path(stage)
    if handoff_path.is_file():
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        checkpoint = ROOT / str(handoff["selected_checkpoint"])
        if checkpoint.is_file():
            return checkpoint

    checked_steps, payloads = load_existing_payloads(stage)
    log_path = LOG_DIR / f"train_a3_curr_stage{stage['stage']}_{stage['label']}_20260723.log"
    final_path = final_checkpoint_path(run_dir, total_steps)
    proc: subprocess.Popen[Any] | None = None
    adopted_pid: int | None = None

    if final_path.is_file():
        write_status("training_already_complete", stage=stage, run_dir=rel(run_dir), final_checkpoint=rel(final_path))
    else:
        existing_pid = find_existing_training_pid(cfg_path)
        if existing_pid is not None:
            adopted_pid = existing_pid
            write_status("training_adopted", stage=stage, pid=existing_pid, config=rel(cfg_path), run_dir=rel(run_dir), log=rel(log_path))
        else:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, str(ROOT / "train.py"), "--config", str(cfg_path)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            handle.close()
            write_status("training_started", stage=stage, pid=proc.pid, config=rel(cfg_path), run_dir=rel(run_dir), log=rel(log_path))

    selected_payload: dict[str, Any] | None = None
    selection_reason = ""
    while True:
        if proc is not None and proc.poll() not in (None, 0):
            raise RuntimeError(f"A3 stage{stage['stage']} training failed with code {proc.returncode}; log={log_path}")
        if adopted_pid is not None and not pid_alive(adopted_pid) and not final_path.is_file():
            raise RuntimeError(f"A3 stage{stage['stage']} adopted training pid {adopted_pid} exited before final checkpoint; log={log_path}")

        for step, checkpoint in checkpoint_steps(run_dir, total_steps):
            if should_check(stage, step, total_steps) and step not in checked_steps:
                checked_steps.add(step)
                payload = evaluate_checkpoint(stage, cfg_path, checkpoint, step)
                payloads.append(payload)
                write_status("screening_checked", stage=stage, step=step, checkpoint=rel(checkpoint), passed=payload["passed"], eligible_for_early_promotion=payload["eligible_for_early_promotion"], metrics=payload["metrics"], selection_score=payload["selection_score"])
                if bool(payload["passed"]) and step >= EARLY_PROMOTE_START:
                    selected_payload = payload
                    selection_reason = "passed at or after early-promotion threshold"
                    if proc is not None:
                        terminate_proc(proc)
                    if adopted_pid is not None:
                        terminate_pid(adopted_pid)
                    break
        if selected_payload is not None:
            break

        if final_path.is_file():
            latest = latest_checkpoint(run_dir, total_steps)
            if latest is not None:
                step, checkpoint = latest
                if should_check(stage, step, total_steps) and step not in checked_steps:
                    checked_steps.add(step)
                    payload = evaluate_checkpoint(stage, cfg_path, checkpoint, step)
                    payloads.append(payload)
                    write_status("screening_checked", stage=stage, step=step, checkpoint=rel(checkpoint), passed=payload["passed"], eligible_for_early_promotion=payload["eligible_for_early_promotion"], metrics=payload["metrics"], selection_score=payload["selection_score"])
            selected_payload = best_payload(payloads)
            if selected_payload is None:
                raise RuntimeError(f"A3 stage{stage['stage']} reached cap but no screening payloads exist")
            selection_reason = "stage cap reached; selecting historical best tested checkpoint"
            break

        time.sleep(POLL_SECONDS)

    return select_and_record(stage, cfg_path, selected_payload, selection_reason)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-stage", type=int, default=1)
    args = parser.parse_args()

    selected: Path | None = None
    for stage in STAGES:
        if int(stage["stage"]) < int(args.start_stage):
            continue
        pretrained = selected
        if not bool(stage.get("scratch")):
            if pretrained is None:
                raise RuntimeError(f"stage{stage['stage']} needs previous selected checkpoint")
            if not pretrained.is_file():
                raise FileNotFoundError(pretrained)
        selected = run_stage(stage, pretrained)
    write_status("curriculum_complete", selected_checkpoint=rel(selected) if selected else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
