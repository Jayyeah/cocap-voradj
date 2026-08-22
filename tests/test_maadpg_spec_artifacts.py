from __future__ import annotations

import ast
import csv
from pathlib import Path

import yaml

from maadpg_reproduction.config import default_environment_config


ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "docs" / "maadpg_reproduction_20260823"
PACKAGE_DIR = ROOT / "src" / "maadpg_reproduction"


def test_r1_spec_artifacts_parse_and_are_traceable():
    ambiguities = yaml.safe_load((SPEC_DIR / "ambiguities.yaml").read_text())
    assumptions = yaml.safe_load((SPEC_DIR / "assumptions.yaml").read_text())
    with (SPEC_DIR / "contract_matrix.csv").open(newline="") as handle:
        contracts = list(csv.DictReader(handle))

    ambiguity_ids = {item["id"] for item in ambiguities["items"]}
    assert len(ambiguity_ids) == len(ambiguities["items"])
    assert all(item["materiality"] for item in ambiguities["items"])
    assert all(item["paper_primary_resolution"] for item in ambiguities["items"])
    assert assumptions["spec_version"] == "maadpg-paper-v1"
    assert assumptions["gate"]["beta_ratio"] == 0.1
    assert "A-GATE-004" in assumptions["gate"]["related_ambiguities"]

    assert len(contracts) == 38
    contract_ids = [row["contract_id"] for row in contracts]
    assert len(contract_ids) == len(set(contract_ids))
    assert all(row["requirement"] and row["verification"] for row in contracts)


def test_executable_environment_defaults_match_frozen_assumptions():
    assumptions = yaml.safe_load((SPEC_DIR / "assumptions.yaml").read_text())
    config = default_environment_config()
    assert config.dt == assumptions["scenario"]["dt"]
    assert config.horizon_steps == assumptions["scenario"]["horizon_steps"]
    assert config.pursuer_observation_dim == assumptions["observation"]["pursuer_dim"]
    assert config.distance_normalizer == assumptions["observation"]["dnorm_km"]
    assert config.capture.rho_min == assumptions["capture"]["rho_min_km"]
    assert config.capture.rho_max == assumptions["capture"]["rho_max_km"]
    assert (
        config.capture.max_angular_gap
        == assumptions["capture"]["max_angular_gap_rad"]
    )
    assert config.reward.success_bonus == assumptions["reward"]["success_bonus_c_succ"]
    assert config.target_policy.policy_id == assumptions["target_policy"]["primary_id"]


def test_validation_and_test_seed_partitions_are_disjoint():
    assumptions = yaml.safe_load((SPEC_DIR / "assumptions.yaml").read_text())
    evaluation = assumptions["evaluation"]

    def expand(name: str) -> set[int]:
        values = evaluation[name]
        return set(range(values["start"], values["stop_inclusive"] + 1))

    tuning = expand("tuning_scenario_seeds")
    validation = expand("validation_scenario_seeds")
    test = expand("test_scenario_seeds")
    assert len(tuning) == 20 and len(validation) == 100 and len(test) == 100
    assert tuning.isdisjoint(validation)
    assert tuning.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_independent_package_has_no_legacy_cocap_imports():
    imported_roots: set[str] = set()
    for path in PACKAGE_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
    assert "cocap_voradj" not in imported_roots
