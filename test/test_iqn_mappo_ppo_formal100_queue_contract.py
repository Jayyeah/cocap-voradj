from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR_PATH = ROOT / "tools/supervise_iqn_mappo_ppo_formal100_20260903.py"
RUNNER_PATH = ROOT / "tools/run_small_step_ac_migration.py"


def load_supervisor():
    spec = importlib.util.spec_from_file_location("bc_ppo_formal100_supervisor", SUPERVISOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_only_is_deterministic_and_returns_before_training():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'parser.add_argument("--eval-only", action="store_true")' in source
    assert 'modes=((True, "deterministic"),)' in source
    assert 'seed=args.eval_seed' in source
    eval_only = source.index("    if args.eval_only:\n", source.index("def main()"))
    training_loop = source.index("    while step < total:")
    assert eval_only < training_loop
    assert '"schema": "small-step-ac-deterministic-formal-v1"' in source
    assert '"checkpoint_sha256": sha256_file(Path(args.resume).resolve())' in source


def test_formal_queue_uses_same_actor_and_exact_branch_contracts():
    module = load_supervisor()
    direct, warm = module.BRANCHES
    assert direct["total"] == 100_000
    assert warm["total"] == 110_000
    assert direct["warmup"] == ()
    assert warm["warmup"] == (
        "--critic-warmup-max-steps", "25000",
        "--critic-warmup-min-steps", "10000",
        "--critic-warmup-ev-threshold", "0.20",
        "--critic-warmup-ev-streak", "2",
    )
    source = SUPERVISOR_PATH.read_text(encoding="utf-8")
    assert '"--actor-init-sha256", ACTOR_SHA' in source
    assert '"--eval-only"' in source
    assert '"--eval-episodes", "100"' in source
    assert '"--eval-seed", str(EVAL_SEED)' in source
    assert module.EVAL_SEED == 2026090301
    assert 'set((report.get("evaluation") or {}).keys()) == {"deterministic"}' in source


def test_formal_report_gate_requires_100_episodes_exact_terminal_step(tmp_path, monkeypatch):
    module = load_supervisor()
    monkeypatch.setattr(module, "FORMAL_ROOT", tmp_path)
    branch = module.BRANCHES[0]
    report_path = tmp_path / branch["name"] / "formal100_report.json"
    report_path.parent.mkdir(parents=True)
    payload = {
        "status": "complete",
        "episodes": 100,
        "step": branch["total"],
        "evaluation_seed": module.EVAL_SEED,
        "evaluation": {"deterministic": {"capture": {}}},
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    assert module.report_complete(branch)
    payload["episodes"] = 20
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    assert not module.report_complete(branch)
