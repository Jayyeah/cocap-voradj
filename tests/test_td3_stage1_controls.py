from __future__ import annotations

from tools.evaluate_td3_stage1_controls_20260916 import evaluate_task


def test_noop_and_random_controls_use_matched_seeds_and_legal_actions() -> None:
    report = evaluate_task("capture", episodes=1, max_steps=2)
    assert report["seed_base"] == 2026191501
    assert set(report["modes"]) == {"noop", "uniform_random"}
    noop = report["modes"]["noop"]["records"][0]
    random = report["modes"]["uniform_random"]["records"][0]
    assert noop["seed"] == random["seed"] == 2026191501
    assert noop["action_a_std"] == 0.0
    assert noop["action_w_std"] == 0.0
    assert 0.0 <= random["action_saturation_ratio"] <= 1.0
