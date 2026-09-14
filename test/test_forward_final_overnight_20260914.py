"""Operational bounds and read-only instrumentation regression."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from tools import forward_final_overnight_20260914 as o


def delta():
    return {group:dict(safe_complete={'delta':0},collision={'delta':0},common_safe_n=20,
        common_safe_mission_seconds={'relative_change':0},discounted_return={'baseline':10,'delta':0},
        return_negative_pair_fraction=0,return_median_delta=0,post_ce_success={'delta':0},
        common_safe_recovery_seconds={'relative_change':0})
        for group in ('argmax/mixed','argmax/coverage','sample/mixed','sample/coverage')}


def test_operational_gate_thresholds_and_missing_pairs():
    d=delta(); assert o.bc_decision(d)['continue_to_10k']
    d['sample/mixed']['safe_complete']['delta']=-.05
    d['sample/coverage']['collision']['delta']=.05
    d['sample/coverage']['common_safe_mission_seconds']['relative_change']=.05
    assert o.bc_decision(d)['continue_to_10k']
    d['sample/mixed']['safe_complete']['delta']=-.10
    assert o.bc_decision(d)['decision']=='REGRESSED'
    d=delta();d['sample/mixed']['common_safe_n']=9
    assert o.bc_decision(d)['decision']=='INCONCLUSIVE'
    assert not o.bc_decision(d)['continue_to_10k']


def test_return_and_recovery_stop_independently():
    d=delta(); r=d['sample/mixed'];r['discounted_return']['delta']=-1
    r['return_negative_pair_fraction']=.7;r['return_median_delta']=-1
    assert not o.bc_decision(d)['continue_to_10k']
    d=delta();d['sample/mixed']['common_safe_recovery_seconds']['relative_change']=.11
    assert not o.bc_decision(d)['continue_to_10k']


def test_cli_cannot_extend_or_change_seed(tmp_path):
    for tool in ('train_forward_final_scratch_mappo_20260914.py','train_forward_final_ppo_diagnostic_20260914.py'):
        for extra in (['--steps','200000'],['--seed','99'],['--formal']):
            r=subprocess.run([sys.executable,str(o.ROOT/'tools'/tool),'--output',str(tmp_path/'never'),*extra],capture_output=True,text=True)
            assert r.returncode==2 and 'unrecognized arguments' in r.stderr
    assert not (tmp_path/'never').exists()


def test_recording_does_not_change_native_stream_or_rng(tmp_path):
    torch.set_num_threads(1)
    o.p.seed_all(123)
    a=o.p.FinalMissionStream(123,tmp_path/'native')
    rng=o.p.rng_state(torch.device('cpu'))
    commands=[np.asarray([4,0,8,2]) for _ in range(12)]
    expected=[a.step(cmd) for cmd in commands]
    rng_after=o.p.rng_state(torch.device('cpu'))
    o.p.seed_all(123)
    b=o.RecordedStream(123,tmp_path/'recorded')
    o.p.assert_tree_equal(rng,o.p.rng_state(torch.device('cpu')))
    actual=[b.step(cmd) for cmd in commands]
    for x,y in zip(expected,actual):
        o.p.assert_tree_equal(x.observations,y.observations)
        np.testing.assert_array_equal(x.rewards,y.rewards)
        o.p.assert_tree_equal(x.dones,y.dones)
    o.p.assert_tree_equal(rng_after,o.p.rng_state(torch.device('cpu')))
    assert b.summary()['cumulative']['env_steps']==12


def test_hard_budget_and_no_checkpoint_rollout_flush():
    assert o.BUDGETS=={'bc_ppo':10000,'scratch':100000}
    assert o.CADENCE['scratch']==[0,25000,50000,75000,100000]
    assert 5000%256==136 and 25000%256==168 and 100000%256==160
