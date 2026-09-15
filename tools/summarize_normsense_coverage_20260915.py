#!/usr/bin/env python3
"""Merge disjoint matched-seed frozen evaluations and enforce predeclared gates."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
from tools import forward_final_single_task_20260915 as st
from cocap_voradj.envs.density_sensing import CALIBRATED_K
OUT=ROOT/'artifacts/2026-09-15_normsense_v2'

def main():
    parts=[json.loads((OUT/f'coverage_part_{i}.json').read_text()) for i in range(4)]
    assert all(p['k']==CALIBRATED_K and p['actor_sha256']==parts[0]['actor_sha256'] for p in parts)
    records=sorted([r for p in parts for r in p['records']],key=lambda r:(r['mode'],r['seed'],r['contract']))
    assert len(records)==80 and len({(r['mode'],r['seed'],r['contract']) for r in records})==80
    summary={};pairs={};rng=np.random.default_rng(2026091501)
    keys=('ce_success','collision','ce_rms','ce_max','area_cv','time_to_ce','obstacle_token_mean','obstacle_token_occupancy','same_state_argmax_disagreement','same_state_action_total_variation')
    for mode in ('argmax','sample'):
        for name in ('legacy','v2'):
            rows=[r for r in records if r['mode']==mode and r['contract']==name]
            assert len(rows)==20
            summary[mode+'/'+name]={k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in keys}
        a={r['seed']:r for r in records if r['mode']==mode and r['contract']=='legacy'}
        b={r['seed']:r for r in records if r['mode']==mode and r['contract']=='v2'}
        assert a.keys()==b.keys()
        for seed in a:assert a[seed]['initial_state_fingerprint']==b[seed]['initial_state_fingerprint']
        paired={}
        for k in keys:
            d=np.array([float(b[s][k])-float(a[s][k]) for s in a if a[s][k] is not None and b[s][k] is not None])
            paired[k]=dict(n=len(d),mean_delta=float(d.mean()) if len(d) else None,median_delta=float(np.median(d)) if len(d) else None,paired_bootstrap_95_mean_interval=np.quantile(d[rng.integers(len(d),size=(5000,len(d)))].mean(1),[.025,.975]).tolist() if len(d) else None)
        pairs[mode]=paired
    old=json.loads((ROOT/'artifacts/2026-09-15_single_task/coverage/eval_step_200000.json').read_text())
    max_error=0.
    for row in records:
        if row['contract']!='legacy':continue
        ref=next(r for r in old['records'] if r['mode']==row['mode'] and r['seed']==row['seed'])
        assert row['initial_state_fingerprint']==ref['initial_state_fingerprint']
        for k in ('ce_success','collision','length'):assert row[k]==ref[k],(row['seed'],row['mode'],k)
        for k in ('ce_rms','ce_max','area_cv','total_return','discounted_return'):
            err=abs(row[k]-ref[k]);max_error=max(max_error,err);assert err<1e-5,(row['seed'],row['mode'],k,err)
    neutral=all(summary[m+'/v2']['ce_success']>=summary[m+'/legacy']['ce_success']-.05 and summary[m+'/v2']['collision']<=summary[m+'/legacy']['collision']+.05 and all(summary[m+'/v2'][k] is not None and summary[m+'/v2'][k]<=summary[m+'/legacy'][k]*1.15 for k in ('ce_rms','ce_max','area_cv','time_to_ce')) for m in ('argmax','sample'))
    usable=all(summary[m+'/v2']['ce_success']>=.9 and summary[m+'/v2']['collision']<=.1 for m in ('argmax','sample'))
    verdict='COVERAGE_TRANSFER_NEUTRAL' if neutral else 'COVERAGE_TRANSFER_SHIFT_BUT_USABLE' if usable else 'COVERAGE_TRANSFER_BREAKS_POLICY'
    result=dict(k=CALIBRATED_K,actor_sha256=parts[0]['actor_sha256'],checkpoint=parts[0]['checkpoint'],checkpoint_sha256=parts[0]['checkpoint_sha256'],optimizer_updates=0,episodes_per_mode_contract=20,matched_initial_fingerprints=True,legacy_replay_matches_original_200k=True,legacy_replay_max_numeric_error=max_error,gate_definition=dict(neutral='each mode: success drop <=5pp, collision rise <=5pp, RMS/max/CV and success-only time <=1.15x legacy',usable='each mode: success >=90%, collision <=10%',breaks='otherwise; MASTER decides whether Coverage V2 training is needed'),verdict=verdict,summary=summary,paired_differences=pairs,records=records,time_to_ce_note='success-only / common-success paired deltas; failures are censored, never assigned a completion time',action_disagreement_note='same physical state, counterfactual sensing; argmax disagreement plus probability total variation, no extra action sampling RNG')
    st.p.write_json(OUT/'coverage_transfer.json',result);print(json.dumps({k:result[k] for k in ('verdict','summary','legacy_replay_max_numeric_error')},indent=2))
if __name__=='__main__':main()
