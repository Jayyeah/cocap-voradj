#!/usr/bin/env python3
"""Run the final CR-MS + VCT-LS + CE 4v1 -> 8v2 -> 12v3 course."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
ARTIFACT_ROOT = ROOT / "artifacts/2026-08-02_cr_ms_support_approach_ce_curriculum"
BEST_ROOT = ROOT / "artifacts/2026-08-02_three_line_stage_best_20rollout10gif"
STATUS_PATH = LOG_DIR / "crms_supportapproach_curriculum_status.json"
STAGES: tuple[dict[str, Any], ...] = (
    {"stage": 1, "label": "4p1e1obs", "line_label": "crms_supportapproach_4v1_s1",
     "config": "configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml",
     "run_dir": "runs/crms_supportapproach_ce_curr_stage1_4p1e1obs_scratch2m_20260802_run1",
     "total_steps": 2_000_000, "capture_evaders": 1, "coverage_max_steps": 1200,
     "mix_max_steps": 2200, "screen_seed": 2026081200, "formal_seed": 2026081201},
    {"stage": 2, "label": "8p2e2obs", "line_label": "crms_supportapproach_8v2_s2",
     "config": "configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage2_8p2e2obs_700k.yaml",
     "run_dir": "runs/crms_supportapproach_ce_curr_stage2_8p2e2obs_700k_20260802_run1",
     "total_steps": 700_000, "capture_evaders": 2, "coverage_max_steps": 1500,
     "mix_max_steps": 2500, "screen_seed": 2026082200, "formal_seed": 2026082201},
    {"stage": 3, "label": "12p3e3obs", "line_label": "crms_supportapproach_12v3_s3",
     "config": "configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage3_12p3e3obs_700k.yaml",
     "run_dir": "runs/crms_supportapproach_ce_curr_stage3_12p3e3obs_700k_20260802_run1",
     "total_steps": 700_000, "capture_evaders": 3, "coverage_max_steps": 1800,
     "mix_max_steps": 2800, "screen_seed": 2026082300, "formal_seed": 2026082301},
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def write_status(state: str, **fields: Any) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": now(), "state": state, **fields}
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, STATUS_PATH)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def load_selection(path: Path) -> Path | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(str(payload["selected_checkpoint"]))
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"selected checkpoint is missing: {checkpoint}")
    return checkpoint


def write_pretrained_wrapper(config_path: Path, runtime_path: Path, checkpoint: Path) -> Path:
    """Write an ignored runtime overlay without mutating the tracked config."""
    checkpoint_value = relative(checkpoint)
    payload = {
        "extends": str(config_path.resolve()),
        "pretrained": {"path": checkpoint_value},
        "experiment_metadata": {
            "pretrained_checkpoint": checkpoint_value,
            "selected_from_previous_stage": True,
            "automatic_promotion_injected_at": now(),
        },
    }
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = runtime_path.with_suffix(runtime_path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    os.replace(temporary, runtime_path)
    return runtime_path


def tmux_exists(session: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", session], cwd=ROOT,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def launch_tmux(session: str, command: list[str], log_path: Path) -> str:
    if tmux_exists(session):
        return "already_running"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    launch = f"cd {shlex.quote(str(ROOT))} && {shlex.join(command)} >> {shlex.quote(str(log_path))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, launch], cwd=ROOT, check=True)
    return "launched"


def stage_paths(stage: dict[str, Any]) -> dict[str, Path]:
    number, label = int(stage["stage"]), str(stage["label"])
    return {"config": ROOT / str(stage["config"]), "run_dir": ROOT / str(stage["run_dir"]),
            "screening": ARTIFACT_ROOT / "screening" / f"stage{number}_{label}",
            "selection": BEST_ROOT / f"{stage['line_label']}_selection.json",
            "train_log": LOG_DIR / f"train_crms_supportapproach_stage{number}_{label}_auto.log",
            "screen_log": LOG_DIR / f"watch_crms_supportapproach_stage{number}_{label}_auto.log",
            "final_log": LOG_DIR / f"finalize_crms_supportapproach_stage{number}_{label}_auto.log"}


def ensure_stage_started(
    stage: dict[str, Any],
    pretrained: Path | None,
    train_device: str = "cuda:0",
    eval_device: str = "cuda:1",
) -> dict[str, str]:
    paths = stage_paths(stage)
    launch_config = paths["config"]
    if pretrained is not None:
        launch_config = write_pretrained_wrapper(
            paths["config"],
            ROOT / "runs" / "_curriculum_runtime" / f"stage{stage['stage']}_{stage['label']}.yaml",
            pretrained,
        )
    number, total = int(stage["stage"]), int(stage["total_steps"])
    prefix = f"crms_sa_ce_s{number}_{stage['label']}_auto"
    actions: dict[str, str] = {}
    final_checkpoint = paths["run_dir"] / "checkpoints" / f"final_step_{total}.pt"
    actions["train"] = "already_complete" if final_checkpoint.is_file() else launch_tmux(
        f"{prefix}_train", ["python3", "train.py", "--config", relative(launch_config),
        "--device", train_device], paths["train_log"])
    screening_done = paths["screening"] / f"step_{total}" / "DONE"
    actions["screening"] = "already_complete" if screening_done.is_file() else launch_tmux(
        f"{prefix}_screen_gpu1", ["bash", "tools/watch_screened_run.sh", relative(paths["config"]),
        relative(paths["run_dir"]), relative(paths["screening"]), str(stage["screen_seed"]), eval_device,
        str(total), str(stage["capture_evaders"]), str(stage["coverage_max_steps"]), str(stage["mix_max_steps"])],
        paths["screen_log"])
    actions["finalizer"] = "selection_ready" if paths["selection"].is_file() else launch_tmux(
        f"{prefix}_finalize_gpu1", ["python3", "tools/finalize_screened_run.py", "--config", relative(paths["config"]),
        "--run-dir", relative(paths["run_dir"]), "--screening-root", relative(paths["screening"]),
        "--best-root", relative(BEST_ROOT), "--line-label", str(stage["line_label"]), "--total-steps", str(total),
        "--seed", str(stage["formal_seed"]), "--device", eval_device, "--capture-evaders", str(stage["capture_evaders"]),
        "--coverage-max-steps", str(stage["coverage_max_steps"]), "--max-steps", str(stage["mix_max_steps"])],
        paths["final_log"])
    return actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--check-once", action="store_true")
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:1")
    args = parser.parse_args()
    pretrained: Path | None = None
    for stage in STAGES:
        paths = stage_paths(stage)
        actions = ensure_stage_started(stage, pretrained, args.train_device, args.eval_device)
        write_status("stage_active", stage=int(stage["stage"]), label=str(stage["label"]),
                     pretrained_checkpoint=relative(pretrained) if pretrained else None, actions=actions,
                     selection_path=relative(paths["selection"]))
        if args.check_once:
            return 0
        while load_selection(paths["selection"]) is None:
            session = f"crms_sa_ce_s{stage['stage']}_{stage['label']}_auto_train"
            final_checkpoint = paths["run_dir"] / "checkpoints" / f"final_step_{stage['total_steps']}.pt"
            if not tmux_exists(session) and not final_checkpoint.is_file():
                raise RuntimeError(f"stage{stage['stage']} training exited before final checkpoint")
            write_status("waiting_stage_selection", stage=int(stage["stage"]), label=str(stage["label"]),
                         selection_path=relative(paths["selection"]))
            time.sleep(max(1, args.poll_seconds))
        pretrained = load_selection(paths["selection"])
        if pretrained is None:
            raise RuntimeError(f"stage{stage['stage']} selection disappeared")
    final_checkpoint = pretrained
    write_status("curriculum_complete", final_selected_checkpoint=relative(final_checkpoint) if final_checkpoint else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
