#!/usr/bin/env python3
"""C3 paired frozen full-task evaluation. Same shared engine as C0 and C1."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.forward_final import CONTRACT,TEACHER,TEACHER_SHA,preflight
from cocap_voradj.training.runtime_semantics import assert_paired_records
from tools.run_forward_final_bridge_20260908 import run_episode,atomic_json,summarize,timing
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file
from tools.distill_forward_final_actor_20260908 import load_actor
MODES=('iqn_greedy','bc_argmax','bc_sample')
GATE={'episodes_per_scene':100,'minimum_safe_success':.90,'max_reliability_drop':.05,'max_collision_increase':.03,'max_collision_absolute':.10,'max_argmax_p90_ratio':1.25,'sample_efficiency_gap_allowed':True,'interpretation':'predeclared engineering gate on point estimates, not a statistical non-inferiority proof'}


def policy_summary(records):
    groups={}
    for row in records:
        raw=row['policy_diagnostics']
        for key,value in {'all':raw,**raw['groups']}.items():
            dest=groups.setdefault(key,{'rows':0,'agreement_hits':0,'entropy_sum':0.,'action_histogram':[0]*9})
            for k in ('rows','agreement_hits','entropy_sum'):dest[k]+=value[k]
            dest['action_histogram']=(np.asarray(dest['action_histogram'])+value['action_histogram']).tolist()
    for v in groups.values():
        v['action_agreement']=v['agreement_hits']/max(1,v['rows']);v['mean_entropy_nats']=v['entropy_sum']/max(1,v['rows'])
    return groups


def paired_delta(a,b,seed=2026096001):
    # A/B ordered by identical scene, seed, initial fingerprint. Resample whole episodes.
    rng=np.random.default_rng(seed);result={}
    for scene in ('mixed','coverage'):
        left=[r for r in a if r['scene']==scene];right=[r for r in b if r['scene']==scene]
        assert_paired_records(left,right);n=len(left);resample=rng.integers(0,n,size=(10000,n))
        delta={}
        for key in ('safe_complete','captured','collision','boundary','ce_success'):
            values=np.asarray([int(y[key])-int(x[key]) for x,y in zip(left,right)],float)
            delta[key]={'student_minus_teacher':float(values.mean()),'paired_bootstrap_95ci':np.percentile(values[resample].mean(1),[2.5,97.5]).tolist()}
        common=[(x,y) for x,y in zip(left,right) if x['safe_complete'] and y['safe_complete']]
        delta['mission_seconds_delta_common_safe']=timing([y['mission_seconds']-x['mission_seconds'] for x,y in common])
        delta['common_safe_pairs']=len(common);delta['total_pairs']=n
        result[scene]=delta
    return result


def decide(reports):
    if any(any(r['summary'][s]['episodes']<100 for s in ('mixed','coverage')) for r in reports.values()):return 'INSUFFICIENT_SAMPLE'
    teacher=reports['iqn_greedy']['summary'];g=GATE
    if min(teacher[s]['safe_complete_rate'] for s in ('mixed','coverage'))<.9:return 'STOP_TEACHER_RECHECK'
    efficiency_gap=False
    for mode in ('bc_argmax','bc_sample'):
        student=reports[mode]['summary']
        for scene in ('mixed','coverage'):
            a=teacher[scene];b=student[scene]
            for key in ('safe_complete_rate','ce_success_rate')+ (('captured_rate',) if scene=='mixed' else ()):
                if b[key]<g['minimum_safe_success']-1e-9 or a[key]-b[key]>g['max_reliability_drop']+1e-9:return 'STOP_REPAIR_DISTILLATION_RELIABILITY'
            if b['collision_rate']>g['max_collision_absolute']+1e-9 or b['collision_rate']-a['collision_rate']>g['max_collision_increase']+1e-9:return 'STOP_REPAIR_DISTILLATION_SAFETY'
            ratio=b['full_mission_seconds_completed']['p90']/a['full_mission_seconds_completed']['p90']
            if ratio>g['max_argmax_p90_ratio']:
                if mode=='bc_argmax':return 'STOP_REPAIR_DISTILLATION_ARGMAX_EFFICIENCY'
                efficiency_gap=True
    return 'PASS_WITH_STOCHASTIC_EFFICIENCY_GAP' if efficiency_gap else 'PASS'


def aggregate(root):
    reports={m:json.loads((root/m/'report.json').read_text()) for m in MODES}
    for mode,r in reports.items():
        assert r['status']=='complete' and r['mode']==mode and r['contract']==CONTRACT and not r['smoke_steps']
        assert r['teacher_sha256']==TEACHER_SHA
    assert reports['bc_argmax']['actor_sha256']==reports['bc_sample']['actor_sha256']
    assert len({r['evaluator_sha256'] for r in reports.values()})==1
    teacher=reports['iqn_greedy']['records'];paired={}
    for mode in MODES[1:]:
        rows=reports[mode]['records'];assert_paired_records(teacher,rows)
        assert [r['scene'] for r in teacher]==[r['scene'] for r in rows]
        paired[mode]=paired_delta(teacher,rows)
    result={'contract':CONTRACT,'decision':decide(reports),'gate':GATE,'paired':paired,'summary':{m:r['summary'] for m,r in reports.items()},'actor_sha256':reports['bc_argmax']['actor_sha256'],'teacher_sha256':TEACHER_SHA,'initial_state_fingerprints_verified':True,'uncertainty_note':'Paired percentile bootstrap; degenerate intervals at all-success do not establish zero population failure risk.'}
    atomic_json(root/'comparison.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);p.add_argument('--mode',choices=MODES)
    p.add_argument('--actor-checkpoint',type=Path);p.add_argument('--episodes',type=int,default=100);p.add_argument('--seed',type=int,default=2026096101)
    p.add_argument('--device',default='cuda:0');p.add_argument('--smoke-steps',type=int);p.add_argument('--aggregate',action='store_true');args=p.parse_args();out=args.output_root
    if args.aggregate:
        print(json.dumps(aggregate(out),indent=2));return
    if args.mode is None:p.error('--mode required')
    assert args.episodes>0
    if (out/'launch.json').exists():raise ValueError('Output exists; never overwrite frozen evidence')
    atomic_json(out/'runtime_preflight.json',preflight());assert sha256_file(TEACHER)==TEACHER_SHA
    teacher=CoCapIQN.load(str(TEACHER),device=args.device).eval();actor=None;actor_sha=None
    if args.mode!='iqn_greedy':
        if args.actor_checkpoint is None:p.error('--actor-checkpoint required')
        actor,payload=load_actor(args.actor_checkpoint,args.device);actor_sha=sha256_file(args.actor_checkpoint)
    engine_sha=sha256_file(ROOT/'tools/run_forward_final_bridge_20260908.py')
    launch={'schema':'forward-final-c3-v1','contract':CONTRACT,'mode':args.mode,'teacher_sha256':TEACHER_SHA,'actor_sha256':actor_sha,'actor_checkpoint':str(args.actor_checkpoint) if actor else None,'evaluator_sha256':engine_sha,'episodes_per_scene':args.episodes,'seed_base':args.seed,'smoke_steps':args.smoke_steps,'ppo_updates':0,'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'gate':GATE}
    atomic_json(out/'launch.json',launch);records=[];start=time.monotonic()
    for pair in range(args.episodes):
        for scene in ('mixed','coverage'):
            seed=args.seed+pair+(100000 if scene=='coverage' else 0)
            def progress(step):
                elapsed=time.monotonic()-start
                atomic_json(out/'progress.json',{'status':'running','completed':len(records),'total':args.episodes*2,'scene':scene,'current_step':step,'elapsed_seconds':elapsed,'eta_seconds':elapsed/max(1,len(records))*(args.episodes*2-len(records))})
            row=run_episode(teacher,scene,seed,args.device,max_steps=args.smoke_steps,actor=actor,policy_mode=args.mode,on_progress=progress)
            records.append(row)
            with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            progress(row['length'])
    report={**launch,'status':'complete','summary':summarize(records),'policy_summary':policy_summary(records),'records':records}
    atomic_json(out/'report.json',report);atomic_json(out/'progress.json',{'status':'complete','completed':len(records),'eta_seconds':0,'elapsed_seconds':time.monotonic()-start})
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

if __name__=='__main__':main()
