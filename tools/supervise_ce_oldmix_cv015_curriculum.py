#!/usr/bin/env python3
"""Old-mix capture + CE coverage curriculum supervised by CV<0.15 diagnostics.

Requested 2026-07-31:
- stage1/2/3 are 4v1 -> 8v2 -> 12v3;
- capture remains the A3 old-mix branch, coverage uses centroid-energy CE;
- CV<0.15 is a loose selection/statistics metric, not a reward term;
- checks begin at 300k every 100k, early promotion is allowed from 500k;
- selected checkpoints launch 20-rollout/10-GIF diagnostics asynchronously.
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
ARTIFACT_ROOT = ROOT / "artifacts" / "2026-07-31_ce_oldmix_cv015_curriculum"
SCREEN_ROOT = ARTIFACT_ROOT / "screening"
BEST_ROOT = ARTIFACT_ROOT / "best_20rollout10gif"
ROLLOUT_SCRIPT = ROOT / "tools/batch_rollouts_parallel.py"
STATUS = LOG_DIR / "ce_oldmix_cv015_curriculum_20260731_status.json"

DEVICE = "cuda:0"
POLL_SECONDS = 60
CHECK_START = 300_000
CHECK_INTERVAL = 100_000
EARLY_PROMOTE_START = 500_000
CHECK_EPISODES = 12
CHECK_WORKERS = 4
FORMAL_EPISODES = 20
FORMAL_GIF_COUNT = 10
FORMAL_WORKERS = 4

STAGES: list[dict[str, Any]] = [
    {
        "stage": 1,
        "label": "4p1e1obs",
        "config": "configs/experiments/ce_oldmix_cv015_curriculum_20260731/stage1_4p1e1obs_700k.yaml",
        "timesteps": 700_000,
        "evaders": 1,
        "coverage_max_steps": 1200,
        "seed": 2026073151,
        "scratch": True,
    },
    {
        "stage": 2,
        "label": "8p2e2obs",
        "config": "configs/experiments/ce_oldmix_cv015_curriculum_20260731/stage2_8p2e2obs_700k.yaml",
        "timesteps": 700_000,
        "evaders": 2,
        "coverage_max_steps": 1500,
        "seed": 2026073152,
    },
    {
        "stage": 3,
        "label": "12p3e3obs",
        "config": "configs/experiments/ce_oldmix_cv015_curriculum_20260731/stage3_12p3e3obs_700k.yaml",
        "timesteps": 700_000,
        "evaders": 3,
        "coverage_max_steps": 1800,
        "seed": 2026073153,
    },
]

PASS_THRESHOLDS = {
    "capture_success_rate": 0.95,
    "mix_capture_rate": 0.95,
    "coverage_cv015_rate": 0.85,
    "mix_cv015_rate": 0.75,
    "max_collision_rate": 0.10,
}
HARD_POSITIVE_KEYS = ["capture_success_rate", "mix_capture_rate", "coverage_cv015_rate", "mix_cv015_rate"]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_status(state: str, **extra: object) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(
        json.dumps({"updated_at": now(), "state": state, **extra}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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


def run_dir_for(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg.get("output_root", "runs") / str(cfg["run_name"])


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


def final_checkpoint_path(run_dir: Path, total_steps: int) -> Path:
    return checkpoint_dir(run_dir) / f"final_step_{total_steps}.pt"


def should_check(step: int, total_steps: int) -> bool:
    if step < CHECK_START:
        return False
    if step >= total_steps:
        return True
    return (step - CHECK_START) % CHECK_INTERVAL == 0


def scalar(summary: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(summary.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if value == value else default


def metric_payload(summaries: dict[str, Any]) -> dict[str, float]:
    capture = summaries.get("capture", {}) or {}
    coverage = summaries.get("coverage", {}) or {}
    mix = summaries.get("mix", {}) or {}
    metrics = {
        "capture_success_rate": scalar(capture, "capture_success_rate"),
        "capture_collision_rate": scalar(capture, "collision_rate"),
        "capture_avg_steps": scalar(capture, "avg_steps", 9999.0),
        "coverage_ce_strict_rate": scalar(coverage, "coverage_success_rate"),
        "coverage_cv015_rate": scalar(coverage, "coverage_cv015_rate"),
        "coverage_cv015_best_rate": scalar(coverage, "coverage_cv015_best_rate"),
        "coverage_collision_rate": scalar(coverage, "collision_rate"),
        "coverage_avg_final_voronoi_cv": scalar(coverage, "avg_final_voronoi_cv", 999.0),
        "coverage_avg_best_voronoi_cv": scalar(coverage, "avg_best_voronoi_cv", 999.0),
        "coverage_avg_steps": scalar(coverage, "avg_steps", 9999.0),
        "mix_capture_rate": scalar(mix, "capture_success_rate"),
        "mix_ce_strict_rate": scalar(mix, "coverage_success_rate"),
        "mix_cv015_rate": scalar(mix, "coverage_cv015_rate"),
        "mix_cv015_best_rate": scalar(mix, "coverage_cv015_best_rate"),
        "mix_collision_rate": scalar(mix, "collision_rate"),
        "mix_avg_final_voronoi_cv": scalar(mix, "avg_final_voronoi_cv", 999.0),
        "mix_avg_best_voronoi_cv": scalar(mix, "avg_best_voronoi_cv", 999.0),
        "mix_avg_steps": scalar(mix, "avg_steps", 9999.0),
    }
    metrics["max_collision_rate"] = max(
        metrics["capture_collision_rate"],
        metrics["coverage_collision_rate"],
        metrics["mix_collision_rate"],
    )
    return metrics


def passed(metrics: dict[str, Any]) -> bool:
    for key, threshold in PASS_THRESHOLDS.items():
        value = float(metrics.get(key, 0.0))
        if key == "max_collision_rate":
            if value > float(threshold):
                return False
        elif value < float(threshold):
            return False
    return True


def score_payload(metrics: dict[str, Any], step: int) -> dict[str, Any]:
    ratios = {
        key: float(metrics.get(key, 0.0)) / max(float(PASS_THRESHOLDS[key]), 1e-9)
        for key in HARD_POSITIVE_KEYS
    }
    collision_margin = (
        float(PASS_THRESHOLDS["max_collision_rate"]) - float(metrics.get("max_collision_rate", 1.0))
    ) / max(float(PASS_THRESHOLDS["max_collision_rate"]), 1e-9)
    score_tuple = [
        min(ratios.values()) if ratios else 0.0,
        sum(ratios.values()) / max(len(ratios), 1),
        float(metrics.get("mix_cv015_rate", 0.0)),
        float(metrics.get("coverage_cv015_rate", 0.0)),
        float(metrics.get("mix_ce_strict_rate", 0.0)),
        float(metrics.get("coverage_ce_strict_rate", 0.0)),
        -float(metrics.get("mix_avg_final_voronoi_cv", 999.0)),
        -float(metrics.get("coverage_avg_final_voronoi_cv", 999.0)),
        collision_margin,
        -float(metrics.get("capture_avg_steps", 9999.0)) / 1000.0,
        -float(metrics.get("mix_avg_steps", 9999.0)) / 2000.0,
        -float(metrics.get("coverage_avg_steps", 9999.0)) / 2000.0,
        int(step),
    ]
    return {
        "score_tuple": score_tuple,
        "normalized_hard_metric_ratios": ratios,
        "collision_margin_ratio": collision_margin,
        "policy": "Hard pass uses capture/mix capture, mix/coverage CV<0.15 and collision. CE strict rate, lower final CV, and fewer steps are tie-breakers.",
    }


def stage_screen_root(stage: dict[str, Any]) -> Path:
    return SCREEN_ROOT / f"stage{stage['stage']}_{stage['label']}"


def evaluate_checkpoint(stage: dict[str, Any], cfg_path: Path, checkpoint: Path, step: int) -> dict[str, Any]:
    out = stage_screen_root(stage) / f"check_step_{step}"
    summary_path = out / "all_summaries.json"
    log_path = LOG_DIR / f"check_ce_oldmix_cv015_stage{stage['stage']}_{stage['label']}_step_{step}_20260731.log"
    if not summary_path.is_file():
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROLLOUT_SCRIPT),
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(checkpoint),
            "--output-root",
            str(out),
            "--episodes",
            str(CHECK_EPISODES),
            "--gif-count",
            "0",
            "--scenarios",
            "capture",
            "coverage",
            "mix",
            "--seed",
            str(int(stage["seed"]) + step),
            "--device",
            DEVICE,
            "--max-steps",
            str(int(stage["coverage_max_steps"]) + 1000),
            "--capture-max-steps",
            "1000",
            "--coverage-max-steps",
            str(int(stage["coverage_max_steps"])),
            "--capture-evaders",
            str(int(stage["evaders"])),
            "--max-gif-frames",
            "1",
            "--workers",
            str(CHECK_WORKERS),
        ]
        write_status("screening_running", stage=stage, checkpoint=rel(checkpoint), step=step, output_root=rel(out), log=rel(log_path))
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"CE oldmix CV015 check failed stage={stage['stage']} step={step} code={result.returncode}; log={log_path}")
    summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = metric_payload(summaries)
    payload = {
        "series": "ce_oldmix_cv015_curriculum",
        "stage": stage,
        "checked_at": now(),
        "checkpoint": rel(checkpoint),
        "checkpoint_step": int(step),
        "output_root": rel(out),
        "summaries": summaries,
        "metrics": metrics,
        "thresholds": PASS_THRESHOLDS,
        "passed": passed(metrics),
        "eligible_for_early_promotion": int(step) >= EARLY_PROMOTE_START,
        "selection_score": score_payload(metrics, step),
        "check_policy": "12 episodes each for capture/coverage/mix, no GIF; CV<0.15 is loose coverage signal; CE strict success is a ranking tie-breaker.",
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
    sess = f"ce_oldmix_cv015_s{stage['stage']}_{checkpoint_id}_rollout_gpu0_0731"
    log_path = LOG_DIR / f"rollout_ce_oldmix_cv015_stage{stage['stage']}_{checkpoint_id}_20260731.log"
    backups = backup_formal_rollout_inputs(out, cfg_path, checkpoint)
    if summary.is_file():
        return {"action": "already_complete", "output_root": rel(out), "checkpoint": rel(checkpoint), **backups}
    if tmux_has_session(sess):
        return {"action": "already_running", "session": sess, "output_root": rel(out), "checkpoint": rel(checkpoint), **backups}
    cmd = [
        sys.executable,
        str(ROLLOUT_SCRIPT),
        "--config",
        str(cfg_path),
        "--checkpoint",
        str(checkpoint),
        "--output-root",
        str(out),
        "--episodes",
        str(FORMAL_EPISODES),
        "--gif-count",
        str(FORMAL_GIF_COUNT),
        "--scenarios",
        "capture",
        "coverage",
        "mix",
        "--seed",
        str(int(stage["seed"]) + 99_000),
        "--device",
        DEVICE,
        "--max-steps",
        str(int(stage["coverage_max_steps"]) + 1000),
        "--capture-max-steps",
        "1000",
        "--coverage-max-steps",
        str(int(stage["coverage_max_steps"])),
        "--capture-evaders",
        str(int(stage["evaders"])),
        "--max-gif-frames",
        "1000",
        "--frame-duration-ms",
        "100",
        "--workers",
        str(FORMAL_WORKERS),
    ]
    launch = f"cd {shlex.quote(str(ROOT))} && {shlex.join(cmd)} >> {shlex.quote(str(log_path))} 2>&1"
    log_path.write_text(f"[{now()}] launching CE oldmix CV015 formal rollout\n", encoding="utf-8")
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
    result = subprocess.run(
        ["pgrep", "-af", "train.py|cocap_voradj.training.trainer"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        command = parts[1]
        if pid != os.getpid() and str(cfg_path) in command:
            return pid
    return None


def handoff_path(stage: dict[str, Any]) -> Path:
    return ARTIFACT_ROOT / f"stage{stage['stage']}_{stage['label']}_handoff.json"


def select_and_record(stage: dict[str, Any], cfg_path: Path, payload: dict[str, Any], reason: str) -> Path:
    checkpoint = ROOT / str(payload["checkpoint"])
    checkpoint_id = f"step_{payload['checkpoint_step']}"
    if checkpoint.name.startswith("final_step_"):
        checkpoint_id = "final"
    rollout = launch_formal_rollout(stage, cfg_path, checkpoint, checkpoint_id)
    handoff = {
        "series": "ce_oldmix_cv015_curriculum",
        "stage": stage,
        "selected_at": now(),
        "selected_checkpoint": rel(checkpoint),
        "selected_candidate_id": checkpoint_id,
        "selected_metrics": payload["metrics"],
        "selected_payload": payload,
        "formal_rollout": rollout,
        "selection_reason": reason,
        "policy": "After selection, launch 20rollout10gif asynchronously and continue to the next stage from the selected checkpoint.",
    }
    path = handoff_path(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(handoff, indent=2, ensure_ascii=False), encoding="utf-8")
    write_status("stage_handoff", stage=stage, selected_checkpoint=rel(checkpoint), selected_metrics=payload["metrics"], selection_reason=reason, formal_rollout=rollout)
    return checkpoint


def run_stage(stage: dict[str, Any], pretrained: Path | None) -> Path:
    cfg = set_stage_pretrained(stage, None if bool(stage.get("scratch")) else pretrained)
    cfg_path = config_path(stage)
    run_dir = run_dir_for(cfg)
    total_steps = int(stage["timesteps"])
    existing_handoff = handoff_path(stage)
    if existing_handoff.is_file():
        handoff = json.loads(existing_handoff.read_text(encoding="utf-8"))
        checkpoint = ROOT / str(handoff["selected_checkpoint"])
        if checkpoint.is_file():
            return checkpoint

    checked_steps, payloads = load_existing_payloads(stage)
    log_path = LOG_DIR / f"train_ce_oldmix_cv015_stage{stage['stage']}_{stage['label']}_20260731.log"
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
            raise RuntimeError(f"CE oldmix CV015 stage{stage['stage']} training failed with code {proc.returncode}; log={log_path}")
        if adopted_pid is not None and not pid_alive(adopted_pid) and not final_path.is_file():
            raise RuntimeError(f"CE oldmix CV015 stage{stage['stage']} adopted pid {adopted_pid} exited before final checkpoint; log={log_path}")

        for step, checkpoint in checkpoint_steps(run_dir, total_steps):
            if should_check(step, total_steps) and step not in checked_steps:
                checked_steps.add(step)
                payload = evaluate_checkpoint(stage, cfg_path, checkpoint, step)
                payloads.append(payload)
                write_status("screening_checked", stage=stage, step=step, checkpoint=rel(checkpoint), passed=payload["passed"], eligible_for_early_promotion=payload["eligible_for_early_promotion"], metrics=payload["metrics"], selection_score=payload["selection_score"])
                if bool(payload["passed"]) and int(step) >= EARLY_PROMOTE_START:
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
            for step, checkpoint in checkpoint_steps(run_dir, total_steps):
                if should_check(step, total_steps) and step not in checked_steps:
                    checked_steps.add(step)
                    payload = evaluate_checkpoint(stage, cfg_path, checkpoint, step)
                    payloads.append(payload)
                    write_status("screening_checked", stage=stage, step=step, checkpoint=rel(checkpoint), passed=payload["passed"], eligible_for_early_promotion=payload["eligible_for_early_promotion"], metrics=payload["metrics"], selection_score=payload["selection_score"])
            selected_payload = best_payload(payloads)
            if selected_payload is None:
                raise RuntimeError(f"CE oldmix CV015 stage{stage['stage']} reached cap but no screening payloads exist")
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
