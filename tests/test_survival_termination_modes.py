from cocap_voradj.envs.voronoi_adjacency import survival_termination_requested


def test_configured_minimum_keeps_existing_behavior():
    assert survival_termination_requested(11, configured_minimum=12) is True
    assert survival_termination_requested(12, configured_minimum=12) is False


def test_degraded_team_mode_terminates_only_at_zero():
    assert survival_termination_requested(11, configured_minimum=12, zero_only=True) is False
    assert survival_termination_requested(1, configured_minimum=12, zero_only=True) is False
    assert survival_termination_requested(0, configured_minimum=12, zero_only=True) is True
