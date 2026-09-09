import numpy as np
from tools.evaluate_forward_final_critic_20260909 import full_returns,calibration


def test_mc_targets_are_full_returns_with_true_terminal_and_no_value_bootstrap():
    r=np.array([[1.,2.],[3.,4.],[5.,6.]])
    terminal=np.array([[False,True],[False,False],[True,True]])
    expected=np.array([[1+.5*3+.25*5,2.],[3+.5*5,4+.5*6],[5.,6.]])
    np.testing.assert_allclose(full_returns(r,terminal,.5),expected)


def test_ev_does_not_hide_constant_bias_in_rmse_gate():
    targets=np.array([-1.,-2.,-3.]);bad=calibration(targets,targets+20)
    assert bad['within_phase_ev']==1 and bad['rmse']==20 and bad['bias']==20
    assert calibration(targets,targets)['rmse']==0


def test_outcome_strata_keep_failed_episode_error_visible():
    from tools.evaluate_forward_final_critic_20260909 import outcome_strata, summarize_rows
    rows = [dict(phase='post_capture', episode=e, **{'return': target},
                 cold=0., warm=0., cold_adv=-1., warm_adv=-1.)
            for e, target in [(0, -1.), (1, -100.)]]
    episodes = [dict(episode=0, ce_success=True, collision=False, task='voradj', captured=True),
                dict(episode=1, ce_success=False, collision=True, task='voradj', captured=True)]
    split = outcome_strata(rows, episodes)['post_capture']
    assert split['safe_completion']['cold']['rmse'] == 1.
    assert split['other_outcome']['cold']['rmse'] == 100.
    assert summarize_rows(rows)['cold']['squared_error_sum'] == 10001.
    assert outcome_strata(rows, episodes)['pure_coverage']['safe_completion']['rows'] == 0
