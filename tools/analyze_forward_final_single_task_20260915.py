#!/usr/bin/env python3
"""Predeclared checkpoint-window gates; best, terminal and historical are separate."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from tools import forward_final_single_task_20260915 as s


def reports(path):
    return [json.loads(p.read_text()) for p in sorted(Path(path).glob('eval_step_*.json'))]


def rate(r,mode,key):return r['summary'][mode][key]


def windows(rs,width=3):return [rs[i:i+width] for i in range(len(rs)-width+1)]


def capture_result(rs):
    if not rs:return dict(status='PENDING')
    g=s.contract()['spec']['gates']
    result={}
    for mode in ('argmax','sample'):
        rows=[dict(step=r['step'],**r['summary'][mode]) for r in rs]
        best=max(rows,key=lambda r:(r['captured_rate'],-r['collision_rate'],-r['step']))
        ws=[dict(steps=[r['step'] for r in w],capture_mean=float(np.mean([r['captured_rate'] for r in w])),
                 capture_min=min(r['captured_rate'] for r in w),collision_mean=float(np.mean([r['collision_rate'] for r in w])),
                 collision_max=max(r['collision_rate'] for r in w)) for w in windows(rows[1:])]
        result[mode]=dict(best=best,terminal=rows[-1],sustained_windows=ws,last_window=ws[-1] if ws else None)
    finished=rs[-1]['step']==500000
    strong=any(all(rate(r,m,'captured_rate')>=g['capture_strong_rate'] and
        rate(r,m,'collision_rate')<=g['capture_usable_collision_max'] for r in w for m in ('argmax','sample'))
        for w in windows(rs[1:]))
    # Strong also requires usable terminal performance, avoiding late-collapse success claims.
    strong=strong and all(rate(rs[-1],m,'captured_rate')>=g['capture_strong_rate'] and
        rate(rs[-1],m,'collision_rate')<=g['capture_usable_collision_max'] for m in ('argmax','sample'))
    weak=any(rate(r,m,'captured_rate')>0 for r in rs[1:] for m in ('argmax','sample'))
    decision='STRONG_STABLE_LEARNABILITY' if strong else ('WEAK_OR_UNSTABLE_LEARNABILITY' if weak else 'NO_MEANINGFUL_LEARNABILITY')
    return dict(status='COMPLETE' if finished else 'RUNNING',decision=decision if finished else None,
                modes=result,steps=[r['step'] for r in rs])


def coverage_result(rs):
    if not rs or rs[-1]['step']!=200000:return dict(status='PENDING',decision=None)
    g=s.contract()['spec']['gates'];base=rs[0]['summary']['sample'];signals=[]
    for r in rs[1:]:
        a=r['summary']['sample']
        success=a['ce_success_rate']>=g['coverage_sustained_success_rate']
        valid=all(a[k]['p50'] is not None and base[k]['p50'] is not None for k in ('ce_rms','area_cv'))
        geometry=valid and all(a[k]['p50']<=(1-g['coverage_geometry_relative_improvement'])*base[k]['p50']
                                  for k in ('ce_rms','area_cv'))
        hold=a['strict_max_hold']['mean']>=base['strict_max_hold']['mean']+g['coverage_hold_growth_steps']
        signals.append(dict(step=r['step'],success=success,geometry=geometry,hold=hold,
                            collision=a['collision_rate'],rms=a['ce_rms']['p50'],cv=a['area_cv']['p50'],
                            hold_mean=a['strict_max_hold']['mean']))
    evidence=[]
    for w in windows(signals):
        success=all(a['success'] for a in w)
        geometry=all(a['geometry'] and a['hold'] for a in w)
        trend=(w[-1]['rms']<=w[0]['rms'] and w[-1]['cv']<=w[0]['cv'] and w[-1]['hold_mean']>=w[0]['hold_mean'])
        if success or (geometry and trend):evidence.append([a['step'] for a in w])
    # Any sustained acquisition blocks automatic reward changes, even if it later regresses.
    learnable=bool(evidence)
    return dict(status='COMPLETE',decision='PURE_COVERAGE_LEARNABLE' if learnable else 'NO_MEANINGFUL_LEARNING',
        signals=signals,sustained_evidence=evidence,terminal=rs[-1]['summary'],
        later_regression=bool(evidence and 200000 not in evidence[-1]),
        next_action='STOP_REWARD_MODIFICATION' if learnable else 'RUN_ATTRIBUTION_PROBES',
        statistical_scope='Single seed, matched 20-episode screen per mode; no population significance claim.')


def historical():
    root=s.ROOT/'artifacts/2026-08-30_mappo9_v2'
    result={}
    for seed in (1,2,3):
        paths=sorted((root/f'mappo9_v2_seed{seed}'/'evaluations').glob('step_*.json'))
        rows=[json.loads(p.read_text()) for p in paths]
        modes={}
        for mode in ('deterministic','stochastic'):
            sequence=[dict(step=r['step'],capture=r[mode]['capture']['capture_rate'],
                           collision=r[mode]['capture']['collision_rate']) for r in rows]
            modes[mode]=dict(best=max(sequence,key=lambda r:(r['capture'],-r['collision'])),terminal=sequence[-1],
                sustained_windows=[dict(steps=[a['step'] for a in w],
                    capture_mean=float(np.mean([a['capture'] for a in w])),capture_min=min(a['capture'] for a in w),
                    collision_mean=float(np.mean([a['collision'] for a in w]))) for w in windows(sequence)],
                sequence=sequence)
        result[f'seed{seed}']=dict(modes=modes,files={str(p.relative_to(s.ROOT)):s.old.sha(p) for p in paths},
            effective_config_sha256=s.old.sha(root/f'mappo9_v2_seed{seed}'/'effective_config.yaml'))
    return dict(historical_reference_only=True,not_matched_to_new_reward_or_horizon=True,seeds=result,
                gate=json.loads((root/'gate_decision.json').read_text()))


def r2_improved(baseline,r2):
    g=s.contract()['spec']['gates'];checks=[]
    for a,b in zip(baseline[-3:],r2[-3:]):
        assert a['step']==b['step']
        x=a['summary']['sample'];y=b['summary']['sample']
        success=y['ce_success_rate']-x['ce_success_rate']>=g['r2_success_improvement_pp']
        geometry=all(y[k]['p50']<=(1-g['r2_geometry_relative_improvement'])*x[k]['p50'] for k in ('ce_rms','area_cv'))
        hold=y['strict_max_hold']['mean']>=x['strict_max_hold']['mean']+3
        checks.append((success or (geometry and hold)) and y['collision_rate']<=x['collision_rate']+.05)
    return len(checks)==3 and all(checks)


def master(root):
    root=Path(root);cr=reports(root/'capture');cv=reports(root/'coverage');r2=reports(root/'coverage_r2')
    capture=capture_result(cr);coverage=coverage_result(cv)
    probe_path=root/'attribution'/'report.json'
    probes=json.loads(probe_path.read_text()) if probe_path.exists() else None
    r2_result=coverage_result(r2) if r2 else dict(status='NOT_STARTED')
    conclusion=None
    coverage_done=coverage['status']=='COMPLETE'
    probes_done=probes is not None
    needs_probes=coverage_done and coverage['decision']=='NO_MEANINGFUL_LEARNING'
    gate_path=root/'attribution'/'r2_gate.json'
    gate=json.loads(gate_path.read_text()) if gate_path.exists() else None
    r2_required=gate and gate['decision']=='START_R2'
    complete=(capture['status']=='COMPLETE' and coverage_done and (not needs_probes or probes_done)
              and (not r2_required or r2_result['status']=='COMPLETE'))
    if complete:
        cap_ok=capture['decision']=='STRONG_STABLE_LEARNABILITY'
        cov_ok=coverage['decision']=='PURE_COVERAGE_LEARNABLE'
        rep=probes['representation']['decision'] if probes else None
        if rep=='REPRESENTATION_LEARNABILITY_SUSPECT':conclusion='COVERAGE_REPRESENTATION_LEARNABILITY_SUSPECT'
        elif r2_required and r2_improved(cv,r2):conclusion='COVERAGE_REWARD_SCALE_CAUSALLY_LIMITING'
        elif cap_ok and cov_ok:conclusion='CAPTURE_AND_COVERAGE_BOTH_INDIVIDUALLY_LEARNABLE'
        elif cap_ok and rep=='REPRESENTATION_SUFFICIENT_FOR_SUPERVISED_LEARNING':
            conclusion='CAPTURE_LEARNABLE_COVERAGE_RL_NOT_LEARNABLE_BUT_SUPERVISED_LEARNABLE'
        else:conclusion='SINGLE_TASKS_STILL_NOT_RELIABLY_LEARNABLE'
    result=dict(schema=s.SCHEMA,status='COMPLETE' if complete else 'IN_PROGRESS',causal_conclusion=conclusion,
        capture=capture,coverage=coverage,attribution=probes,r2=r2_result,r2_gate=gate,
        historical_reference=json.loads((root/'historical_reference.json').read_text()),
        R3='NOT_AUTHORIZED; ledger proposal only if R2 fails',full_task_training='NOT_STARTED')
    s.p.write_json(root/'MASTER.json',result)
    return result
