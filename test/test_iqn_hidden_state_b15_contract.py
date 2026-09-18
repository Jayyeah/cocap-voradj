from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/audit_iqn_hidden_state_b15_20260918.py"


def test_b15_audit_is_offline_and_declares_fixed_grid():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "online_rl" in source
    assert "formal_iqn_observation_modified" in source
    assert "LAMBDA_GRID = (0.80, 0.90, 0.95, 0.98)" in source
    assert "ETA_GRID = (0.50, 0.70, 0.85, 0.95)" in source
    assert "HISTORY_K = (3, 5, 10)" in source
    assert "teacher_q" in source and "greedy_action" in source
    assert "np.random" not in source


def test_b15_does_not_modify_formal_runtime_contract():
    assert not (ROOT / "src/cocap_voradj/models/iqn.py").read_text(encoding="utf-8").count("b15")
