#!/usr/bin/env python3
"""Read-only evidence audit: join independent evaluations to exact-step PPO metrics.

This utility is outside the frozen scientific source tree. It does not update
models, checkpoints, evaluations, runtime state, or selection decisions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

EVAL_KEYS = ('episodes','capture_count','normal_capture_count','capture_rate','normal_capture_rate',
             'capture_time_mean','capture_time_median','capture_time_p90','success_n','censored_n',
             'ring2_rate','ring3_rate','strict_geometry_rate','collision_rate','collision_types',
             'censored_rate','reward_components','action_entropy')
PPO_KEYS = ('actor_loss','value_loss','explained_variance','approx_kl','actor_grad_norm','value_grad_norm',
            'clip_fraction','post_update_ratio_mean','post_update_ratio_min','post_update_ratio_max')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read_metrics(path):
    result={}; trailing_incomplete=False; previous=-1
    if not path.exists(): return result,trailing_incomplete
    lines=path.read_text().splitlines()
    for index,line in enumerate(lines):
        try: row=json.loads(line)
        except json.JSONDecodeError:
            if index==len(lines)-1:
                trailing_incomplete=True; break
            raise
        step=row['steps']
        if step<=previous: raise ValueError('metrics are duplicated/out of order; explicit resume lineage reconciliation required')
        previous=step; result[step]=row
    return result,trailing_incomplete

def audit(out):
    out=Path(out).resolve()
    manifest=json.loads((out/'manifest.json').read_text())
    cfg=manifest['config']; metrics,trailing=read_metrics(out/'metrics.jsonl')
    results=[]; domains={}
    for p in sorted((out/'evaluations').glob('*.json')):
        if p.name.endswith('.partial.json'): continue
        data=json.loads(p.read_text()); step=data['steps']; domain=data['seed_domain']
        if not set(data['modes'])=={'argmax','sample'}: raise ValueError('both action modes required')
        checkpoint=Path(data['checkpoint'])
        if not checkpoint.exists() or digest(checkpoint)!=data['checkpoint_sha256']:
            raise ValueError('evaluation checkpoint/hash cannot be verified')
        checkpoint_metrics=metrics.get(step)
        if step and checkpoint_metrics is None: raise ValueError('exact checkpoint PPO telemetry missing')
        if checkpoint_metrics:
            missing=set(PPO_KEYS)-set(checkpoint_metrics)
            if missing: raise ValueError(f'PPO telemetry missing {sorted(missing)}')
        mode_evidence={}
        for mode,m in data['modes'].items():
            if set(EVAL_KEYS)-set(m): raise ValueError('required evaluation statistics missing')
            rows=[r for r in data['episodes'] if r['mode']==mode]
            if len(rows)!=m['episodes']: raise ValueError('episode count mismatch')
            if len({r['seed'] for r in rows})!=len(rows): raise ValueError('duplicate episode seed')
            if sum(r['capture'] for r in rows)!=m['capture_count']: raise ValueError('capture count mismatch')
            if sum(r['normal_capture'] for r in rows)!=m['normal_capture_count']: raise ValueError('normal capture count mismatch')
            if sum(r['collision'] for r in rows)!=round(m['collision_rate']*m['episodes']): raise ValueError('collision count mismatch')
            if sum(not r['capture'] for r in rows)!=m['censored_n']: raise ValueError('censored count mismatch')
            if m['success_n']==0 and any(m[k] is not None for k in ('capture_time_mean','capture_time_median','capture_time_p90')):
                raise ValueError('capture times reported with no successful episode')
            # Expose collision-failure vs pure time censoring as separate counts;
            # preserve the evaluator's original all-no-capture censored convention.
            mode_evidence[mode]={
                'evaluation':{k:m[k] for k in EVAL_KEYS},
                'collision_failure_n':sum(not r['capture'] and r['collision'] for r in rows),
                'time_limit_censored_n':sum(not r['capture'] and not r['collision'] for r in rows),
                'ppo_training_at_checkpoint':{k:checkpoint_metrics[k] for k in PPO_KEYS} if checkpoint_metrics else None,
                'ppo_training_applicability':'training telemetry; shared across argmax/sample' if step else 'not applicable: random initialization has no optimizer update'}
            domains.setdefault(domain,set()).update(r['seed'] for r in rows)
        a={r['seed']:r['initial_fingerprint'] for r in data['episodes'] if r['mode']=='argmax'}
        b={r['seed']:r['initial_fingerprint'] for r in data['episodes'] if r['mode']=='sample'}
        if a!=b: raise ValueError('argmax/sample physical initial states differ')
        results.append({'steps':step,'seed_domain':domain,'checkpoint_sha256':data['checkpoint_sha256'],
                        'modes':mode_evidence})
    for a in domains:
        for b in domains:
            if a!=b and domains[a]&domains[b]: raise ValueError('screen/selection/final seed domains overlap')
    candidates=[0,25000,50000,75000,100000]
    screened={r['steps'] for r in results if r['seed_domain']=='screen'}
    pending=[s for s in candidates if s not in screened]
    progress=json.loads((out/'progress.json').read_text())
    return {'run':str(out),'terl_sha':manifest['terl_sha'],'scientific_config':cfg,
            'progress':progress,'metrics_last_step':max(metrics) if metrics else 0,
            'incomplete_trailing_metrics_line':trailing,'checkpoint_evidence':results,
            'pending_screen_steps':pending,'domains_disjoint':True,
            'scope_complete':not pending and 'final' in domains,
            'decision':'PENDING' if pending or 'final' not in domains else 'read supervisor selection.json; no reclassification by this auditor'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output')
    a=p.parse_args(); result=audit(a.run)
    if a.output:
        dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_suffix(dest.suffix+'.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(dest)
    print(json.dumps({'steps':result['progress']['steps'],'complete_evaluations':len(result['checkpoint_evidence']),
                      'pending_screen_steps':result['pending_screen_steps'],'scope_complete':result['scope_complete']},ensure_ascii=False))

if __name__=='__main__': main()
