from __future__ import annotations

from pathlib import Path

from tools.build_continuous_baseline_manifest import BASELINE_DEFINITIONS, build_manifest, verify_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_definitions_cover_ce_first_reference_scales() -> None:
    assert set(BASELINE_DEFINITIONS) == {"4v1", "8v2", "12v3"}
    assert BASELINE_DEFINITIONS["4v1"]["selected_step"] == 2_000_000
    assert BASELINE_DEFINITIONS["8v2"]["selected_step"] == 300_000
    assert BASELINE_DEFINITIONS["12v3"]["selected_step"] == 700_000
    assert [BASELINE_DEFINITIONS[key]["capture_evaders"] for key in ("4v1", "8v2", "12v3")] == [1, 2, 3]


def test_manifest_has_fixed_seeds_and_trails_off() -> None:
    manifest = build_manifest(ROOT, generated_at="test")
    assert manifest["baseline_selection"].startswith("4v1 step 2M")
    assert not verify_manifest(manifest, ROOT)
    for item in manifest["baselines"].values():
        assert item["formal_seed_contract_ok"]
        assert item["screening_seed_contract_ok"]
        assert item["draw_trails_values_observed"] == [False]
        assert item["missing_required_files"] == []
