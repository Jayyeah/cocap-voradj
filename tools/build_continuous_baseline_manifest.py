#!/usr/bin/env python3
"""Build and verify the immutable P0 baseline manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = Path("artifacts/2026-08-02_three_line_stage_best_20rollout10gif")
SCREENING_ROOT = Path("artifacts/2026-08-02_cr_ms_support_approach_ce_curriculum/screening")

BASELINE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "4v1": {
        "line": "crms_supportapproach_4v1_s1",
        "formal_dir": FORMAL_ROOT / "crms_supportapproach_4v1_s1_step_2000000",
        "run_dir": Path("runs/crms_supportapproach_ce_curr_stage1_4p1e1obs_scratch2m_20260802_run1"),
        "config": Path("configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml"),
        "screening_dir": SCREENING_ROOT / "stage1_4p1e1obs",
        "selected_step": 2_000_000,
        "capture_evaders": 1,
    },
    "8v2": {
        "line": "crms_supportapproach_8v2_s2_cefirst_300k",
        "formal_dir": FORMAL_ROOT / "crms_supportapproach_8v2_s2_step_300000_cefirst_comparison",
        "run_dir": Path("runs/crms_supportapproach_ce_curr_stage2_8p2e2obs_700k_20260802_run1"),
        "config": Path("configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage2_8p2e2obs_700k.yaml"),
        "screening_dir": SCREENING_ROOT / "stage2_8p2e2obs",
        "selected_step": 300_000,
        "capture_evaders": 2,
        "historical_formal_dir": FORMAL_ROOT / "crms_supportapproach_8v2_s2_step_500000",
    },
    "12v3": {
        "line": "crms_supportapproach_12v3_s3",
        "formal_dir": FORMAL_ROOT / "crms_supportapproach_12v3_s3_step_700000",
        "run_dir": Path("runs/crms_supportapproach_ce_curr_stage3_12p3e3obs_700k_20260802_run1"),
        "config": Path("configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage3_12p3e3obs_700k.yaml"),
        "screening_dir": SCREENING_ROOT / "stage3_12p3e3obs",
        "selected_step": 700_000,
        "capture_evaders": 3,
    },
}

SUMMARY_KEYS = (
    "capture_success_rate", "coverage_success_rate", "coverage_settled_rate",
    "collision_rate", "coverage_cv015_rate", "avg_final_voronoi_cv", "avg_steps",
)


def rel(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def file_record(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": rel(path), "exists": path.is_file()}
    if path.is_file():
        item["bytes"] = path.stat().st_size
        item["sha256"] = sha256(path)
    return item


def summary_projection(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in SUMMARY_KEYS if key in payload}


def seed_values(summary_path: Path) -> list[int]:
    if not summary_path.is_file():
        return []
    values: set[int] = set()
    for record in read_json(summary_path).get("records", []):
        try:
            values.add(int(record["seed"]))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(values)


def scenario_summaries(root: Path, scenarios: tuple[str, ...]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scenario in scenarios:
        path = root / scenario / "batch_summary.json"
        if path.is_file():
            payload = read_json(path)
            output[scenario] = {
                "summary": summary_projection(payload.get("summary", {})),
                "seed_values": seed_values(path),
                "summary_path": rel(path),
            }
        else:
            output[scenario] = {"summary": {}, "seed_values": [], "summary_path": rel(path)}
    return output


def screening_steps(screening_dir: Path) -> list[int]:
    steps: list[int] = []
    for step_dir in screening_dir.glob("step_*"):
        if not step_dir.is_dir() or not (step_dir / "DONE").is_file():
            continue
        try:
            steps.append(int(step_dir.name.removeprefix("step_")))
        except ValueError:
            continue
    return sorted(steps)


def screening_seed_values(screening_dir: Path) -> list[int]:
    values: set[int] = set()
    for step_dir in screening_dir.glob("step_*"):
        args_path = step_dir / "run_args.json"
        if not args_path.is_file():
            continue
        value = read_json(args_path).get("seed")
        if value is not None:
            values.add(int(value))
    return sorted(values)


def build_baseline(root: Path, key: str, definition: dict[str, Any]) -> dict[str, Any]:
    formal_dir = root / definition["formal_dir"]
    run_dir = root / definition["run_dir"]
    config = root / definition["config"]
    screening_dir = root / definition["screening_dir"]
    step = int(definition["selected_step"])
    formal_summary = formal_dir / "all_summaries.json"
    screening_step_dir = screening_dir / f"step_{step}"
    screening_summary = screening_step_dir / "all_summaries.json"
    screening_args = screening_step_dir / "run_args.json"
    required = [
        config, run_dir / "effective_config.yaml",
        run_dir / "checkpoints" / f"step_{step}.pt",
        formal_summary, formal_dir / "run_args.json",
        formal_dir / "capture" / "batch_summary.json",
        formal_dir / "coverage" / "batch_summary.json",
        formal_dir / "mix" / "batch_summary.json",
        screening_step_dir / "DONE", screening_summary, screening_args,
    ]
    optional_files = [
        formal_dir / f"step_{step}.pt",
        formal_dir / "effective_config.yaml",
        formal_dir / "selection.json",
        screening_dir / "README_screening.txt",
    ]
    evidence_files = required + [path for path in optional_files if path.is_file()]
    formal_scenarios = scenario_summaries(formal_dir, ("capture", "coverage", "mix"))
    screen_all = read_json(screening_summary) if screening_summary.is_file() else {}
    trails: list[bool] = []
    for path in (formal_dir / "run_args.json", screening_args):
        if path.is_file():
            trails.append(bool(read_json(path).get("draw_trails", False)))
    formal_seed_union = sorted({
        seed for item in formal_scenarios.values() for seed in item["seed_values"]
    })
    regression_root = root / "artifacts/2026-08-04_continuous_marl_refactor/old_iqn_regression"
    regression_candidates = []
    if formal_seed_union:
        regression_candidates = [
            regression_root / (
                f"{key}_seed_{formal_seed_union[0]}_evaders_{int(definition['capture_evaders'])}"
            ) / "regression_report.json",
            regression_root / f"{key}_seed_{formal_seed_union[0]}" / "regression_report.json",
        ]
    regression_paths = []
    for candidate in regression_candidates:
        if candidate.is_file():
            regression_paths = [candidate]
            break
    regression_payloads = [read_json(path) for path in regression_paths]
    unique_files = list(dict.fromkeys(evidence_files + regression_paths))
    payload: dict[str, Any] = {
        "line_key": key,
        "line": definition["line"],
        "selected_step": step,
        "capture_evaders": int(definition["capture_evaders"]),
        "selection_policy": "capture and mix capture -> CE strict -> collision -> steps -> CV",
        "formal_dir": rel(formal_dir), "run_dir": rel(run_dir),
        "config": rel(config), "screening_dir": rel(screening_dir),
        "screening_steps": screening_steps(screening_dir),
        "formal_seed_values": formal_seed_union,
        "screening_seed_values": screening_seed_values(screening_dir),
        "draw_trails_values_observed": sorted(set(trails)),
        "regression_reports": [rel(path) for path in regression_paths],
        "regression_statuses": [payload.get("status") for payload in regression_payloads],
        "formal_scenarios": formal_scenarios,
        "screening_selected_summaries": {
            scenario: summary_projection(value) for scenario, value in screen_all.items()
        },
        "required_files": [rel(path) for path in required],
        "file_records": [file_record(path) for path in unique_files],
    }
    if definition.get("historical_formal_dir") is not None:
        payload["historical_formal_dir"] = rel(root / definition["historical_formal_dir"])
    payload["missing_required_files"] = [rel(path) for path in required if not path.is_file()]
    payload["formal_seed_contract_ok"] = bool(payload["formal_seed_values"])
    payload["screening_seed_contract_ok"] = bool(payload["screening_seed_values"])
    payload["trail_contract_ok"] = trails == [False] * len(trails) if trails else False
    payload["regression_contract_ok"] = bool(regression_payloads) and all(
        item.get("status") == "passed"
        and not item.get("core_field_mismatches")
        and item.get("draw_trails_observed") is False
        for item in regression_payloads
    )
    return payload


def build_manifest(root: Path = ROOT, generated_at: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "P0 immutable old-CR-MS reference for continuous-action experiments",
        "continuous_line_namespace": {
            "config_root": "configs/experiments/continuous_marl_20260804",
            "run_prefix": "runs/continuous_marl_",
            "artifact_root": "artifacts/2026-08-04_continuous_marl_refactor",
            "old_action_mode": "unicycle_discrete",
        },
        "git": {"commit": git_value("rev-parse", "HEAD"), "status_short": git_value("status", "--short")},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "baseline_selection": "4v1 step 2M; 8v2 CE-first step 300k; 12v3 step 700k",
        "baselines": {
            key: build_baseline(root, key, definition)
            for key, definition in BASELINE_DEFINITIONS.items()
        },
    }


def verify_manifest(manifest: dict[str, Any], root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    for key, item in (manifest.get("baselines", {}) or {}).items():
        if item.get("missing_required_files"):
            errors.append(f"{key}: missing {item['missing_required_files']}")
        if not item.get("formal_seed_contract_ok"):
            errors.append(f"{key}: formal seed contract is empty")
        if not item.get("screening_seed_contract_ok"):
            errors.append(f"{key}: screening seed contract is empty")
        if not item.get("regression_contract_ok"):
            errors.append(f"{key}: fixed-seed regression contract is incomplete")
        if item.get("draw_trails_values_observed") != [False]:
            errors.append(f"{key}: trail default is not false")
        for record in item.get("file_records", []):
            path = root / record["path"]
            if not path.is_file():
                errors.append(f"{key}: missing {record['path']}")
            elif record.get("sha256") and sha256(path) != record["sha256"]:
                errors.append(f"{key}: sha256 changed for {record['path']}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/2026-08-04_continuous_marl_refactor/baseline_manifest.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = ROOT / args.output
    if args.check:
        if not output.is_file():
            print(f"manifest does not exist: {output}", file=sys.stderr)
            return 2
        errors = verify_manifest(read_json(output))
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print(f"verified {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    errors = verify_manifest(manifest)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
