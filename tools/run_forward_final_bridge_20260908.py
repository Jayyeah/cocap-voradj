#!/usr/bin/env python3
"""One Forward Final full-task evaluator; capture is a mixed prefix statistic."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.forward_final import CONTRACT,TEACHER,TEACHER_SHA,preflight,make_env
from cocap_voradj.training.runtime_semantics import initial_state_fingerprint
from cocap_voradj.evaluation.mission_events import MissionEventTracker,snapshot,summarize_events
from tools.collect_iqn_aw_teacher_dataset_20260903 import fixed_midpoint_q,sha256_file
from tools.rollout_voradj_visual import act_evaders


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(tmp,path)


def timing(values):
    values=[float(v) for v in values if v is not None]
    return {'n':len(values),'mean':float(np.mean(values)) if values else None,
            'p50':float(np.percentile(values,50)) if values else None,
            'p90':float(np.percentile(values,90)) if values else None}


def summarize(records):
    result={}
    for scene in ('mixed','coverage'):
        rows=[r for r in records if r['scene']==scene]
        if not rows:continue
        rate=lambda key:sum(bool(r[key]) for r in rows)/len(rows)
        result[scene]={'episodes':len(rows),**{k+'_rate':rate(k) for k in ('captured','normal_capture','stationary_capture','collision','boundary','ce_success','safe_complete')},
            'full_mission_seconds_completed':timing([r['mission_seconds'] for r in rows if r['safe_complete']]),
            'observed_episode_seconds_all':timing([r['episode_seconds'] for r in rows]),
            'capture_seconds_completed':timing([r['capture_seconds'] for r in rows]),
            'capture_to_ce_seconds_completed':timing([r['recovery_seconds'] for r in rows if r['safe_complete']]),
            'detection_to_capture_seconds_completed':timing([r['detection_to_capture_seconds'] for r in rows]),
            'mission_event_summary':summarize_events([e for r in rows for e in r['mission_events']['events']]),
            'safe_completion_curve':{str(t):sum(r['safe_complete'] and r['mission_seconds']<=t for r in rows)/len(rows) for t in (30,60,120,180,300,600,1500)},
            'right_censored_or_failed_episodes':sum(not r['safe_complete'] for r in rows)}
        if scene=='mixed':
            result[scene]['capture_prefix_collision_rate']=rate('capture_prefix_collision')
    return result


def c0_gate(summary):
    if any(summary.get(s,{}).get('episodes',0)<20 for s in ('mixed','coverage')):return 'INSUFFICIENT_SAMPLE'
    mixed=summary['mixed'];coverage=summary['coverage']
    scores=[mixed['captured_rate'],mixed['normal_capture_rate'],mixed['ce_success_rate'],mixed['safe_complete_rate'],coverage['ce_success_rate'],coverage['safe_complete_rate']]
    collision=max(mixed['collision_rate'],coverage['collision_rate'])
    if min(scores)>=.9 and collision<=.1:return 'PASS'
    if min(scores)<.7 or collision>.3:return 'STOP_TEACHER_UNQUALIFIED'
    return 'INCONCLUSIVE_EXTEND_TO_50'


@torch.no_grad()
def run_episode(model,scene,seed,device,*,max_steps=None,on_transition=None,on_progress=None,actor=None,policy_mode="iqn_greedy"):
    if policy_mode not in ('iqn_greedy','bc_argmax','bc_sample'):raise ValueError(policy_mode)
    if (actor is None)!=(policy_mode=='iqn_greedy'):raise ValueError('Policy mode/actor mismatch')
    if actor is not None and on_transition is not None:raise ValueError('Teacher dataset collection must remain greedy IQN')
    env,observations=make_env(scene,seed)
    if actor is not None:
        from cocap_voradj.training.runtime_semantics import assert_runtime
        assert_runtime(env,actor,categorical=True)
        np.testing.assert_allclose(actor.action_grid.cpu().numpy(),env.pursuers[0].action_list,atol=1e-6)
        torch.manual_seed(int(seed))
    policy_stats={'rows':0,'agreement_hits':0,'entropy_sum':0.,'action_histogram':[0]*9,'groups':{}}
    fingerprint=initial_state_fingerprint(env);dt=env.pursuers[0].dt*env.pursuers[0].N
    initial=snapshot(env,observations);tracker=MissionEventTracker(dt);tracker.observe(initial,0)
    first_detection=0 if any(initial['direct'].values()) else None
    capture_step=None;ce_step=None;capture_types=[];collision=False;boundary=False;prefix_collision=False
    apf=[ApfAgent(e.a,e.w) for e in env.evaders]
    horizon=min(env.episode_max_length,max_steps) if max_steps else env.episode_max_length
    for step in range(1,horizon+1):
        active=[i for i,o in enumerate(observations) if o is not None]
        if not active:raise RuntimeError('No active observations before environment termination')
        local=[observations[i] for i in active]
        q,greedy=fixed_midpoint_q(model,local,device)
        state=snapshot(env,observations)
        phase='pure_coverage' if scene=='coverage' else ('post_capture' if capture_step is not None else 'pre_capture')
        global_state=None
        if on_transition:
            from cocap_voradj.training.continuous.central_schema import build_central_global_obs
            global_state=build_central_global_obs(env,max_agents=4,max_evaders=8,max_obstacles=5,self_feature_dim=9)
        chosen=greedy;entropy=np.zeros(len(active))
        if actor is not None:
            from cocap_voradj.training.small_step_ac import tensor_tree
            batch=tensor_tree({k:np.stack([o[k] for o in local]) for k in local[0]},torch.device(device))
            distribution=actor.distribution(batch)
            chosen=(distribution.logits.argmax(-1) if policy_mode=='bc_argmax' else distribution.sample()).cpu().numpy()
            entropy=distribution.entropy().cpu().numpy()
        hits=np.asarray(chosen)==np.asarray(greedy)
        policy_stats['rows']+=len(active);policy_stats['agreement_hits']+=int(hits.sum());policy_stats['entropy_sum']+=float(entropy.sum())
        policy_stats['action_histogram']=(np.asarray(policy_stats['action_histogram'])+np.bincount(chosen,minlength=9)).tolist()
        commands=[None]*len(observations)
        for i,a in zip(active,chosen):commands[i]=int(a)
        outcome=env.step(commands,act_evaders(env,apf))
        for row,i in enumerate(active):
            meta=outcome.infos[i]['replay_metadata']
            role='direct' if state['direct'].get(i) else 'pursuing_memory' if meta['effective_pursuing'] else 'support' if meta['support_candidate'] else 'coverage'
            for group in (f'phase/{phase}',f'role/{role}'):
                bucket=policy_stats['groups'].setdefault(group,{'rows':0,'agreement_hits':0,'entropy_sum':0.,'action_histogram':[0]*9})
                bucket['rows']+=1;bucket['agreement_hits']+=int(hits[row]);bucket['entropy_sum']+=float(entropy[row]);bucket['action_histogram'][int(chosen[row])]+=1
        events=list(env.last_capture_events);capture_types.extend(e['capture_type'] for e in events)
        hits=list(env.last_collision_events)
        collision=collision or bool(hits)
        boundary=boundary or any(e.get('type')=='boundary' for e in hits)
        if capture_step is None:prefix_collision=prefix_collision or bool(hits)
        captured=bool(env.evaders and all(e.deactivated and not e.collision for e in env.evaders))
        if captured and capture_step is None:capture_step=step
        if env.post_capture_coverage_success and ce_step is None:ce_step=step
        after=snapshot(env,outcome.observations);tracker.observe(after,step,events)
        if first_detection is None and any(after['direct'].values()):first_detection=step
        if on_transition:
            on_transition(env,local,global_state,q,greedy,active,state,phase,step,outcome)
        observations=outcome.observations
        if on_progress and step%100==0:on_progress(step)
        if all(outcome.dones):break
    record=env.episode_record(task='coverage' if scene=='coverage' else 'mix')
    collision=collision or bool(record.get('collision_event',False))
    boundary=boundary or bool(record.get('boundary_collision_event',False))
    captured=bool(record['captured'])
    ce=bool(env.post_capture_coverage_success)
    safe=bool(ce and (scene=='coverage' or captured) and not collision and record['all_pursuers_active'])
    return {'scene':scene,'seed':int(seed),'initial_state_fingerprint':fingerprint,'contract':CONTRACT,
        'captured':captured,'normal_capture':captured and 'loose' in capture_types,
        'stationary_capture':captured and 'stationary' in capture_types,'capture_types':capture_types,
        'collision':collision,'boundary':boundary,'capture_prefix_collision':prefix_collision,
        'ce_success':ce,'safe_complete':safe,'length':step,'episode_seconds':step*dt,
        'capture_seconds':capture_step*dt if capture_step is not None else None,
        'recovery_seconds':(ce_step-capture_step)*dt if ce_step is not None and capture_step is not None else None,
        'mission_seconds':ce_step*dt if safe and ce_step is not None else None,
        'detection_to_capture_seconds':(capture_step-first_detection)*dt if capture_step is not None and first_detection is not None else None,
        'mission_events':tracker.finish(step),'smoke_horizon_override':max_steps,'policy_mode':policy_mode,'policy_diagnostics':policy_stats,
        'ce_rms':float(record['coverage_ce_center_rms']) if np.isfinite(record['coverage_ce_center_rms']) else None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed',type=int,default=2026092101)
    p.add_argument('--device',default='cuda:0');p.add_argument('--smoke-steps',type=int)
    args=p.parse_args();out=args.output_root
    if (out/'launch_contract.json').exists():raise ValueError('Output exists: use fresh run directory; never overwrite completed evidence')
    assert args.episodes>0
    contract=preflight();assert sha256_file(TEACHER)==TEACHER_SHA
    atomic_json(out/'runtime_preflight.json',contract)
    atomic_json(out/'launch_contract.json',{'schema':'forward-final-c0-v1','contract':CONTRACT,'teacher_sha256':TEACHER_SHA,'teacher_path':str(TEACHER),'episodes_per_scene':args.episodes,'seed_base':args.seed,'mode':'fixed_midpoint32_greedy','ppo_updates':0,'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'smoke_steps':args.smoke_steps,
        'gate':{'minimum_per_scene':20,'min_success_rates':.9,'max_collision_rate':.1,'stop_if_success_below':.7,'stop_if_collision_above':.3},
        'timing_contract':'completed timing conditional on safe success; failed/censored counts and all-episode observed durations reported separately; no failure-time imputation'})
    model=CoCapIQN.load(str(TEACHER),device=args.device).eval();records=[];start=time.monotonic()
    for i in range(args.episodes):
        for scene in ('mixed','coverage'):
            seed=args.seed+i+(100000 if scene=='coverage' else 0)
            def progress(step):
                elapsed=time.monotonic()-start
                atomic_json(out/'progress.json',{'status':'running','completed':len(records),'total':args.episodes*2,'current_scene':scene,'current_episode_step':step,'elapsed_seconds':elapsed,'eta_seconds':elapsed/max(len(records),1)*(args.episodes*2-len(records))})
            row=run_episode(model,scene,seed,args.device,max_steps=args.smoke_steps,on_progress=progress)
            records.append(row)
            with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            progress(row['length'])
    summary=summarize(records);decision='SMOKE_ONLY' if args.smoke_steps else c0_gate(summary)
    atomic_json(out/'report.json',{'status':'complete','decision':decision,'summary':summary,'records':records,'teacher_sha256':TEACHER_SHA,'contract':CONTRACT})
    atomic_json(out/'progress.json',{'status':'complete','completed':len(records),'total':len(records),'elapsed_seconds':time.monotonic()-start,'eta_seconds':0,'decision':decision})
    print(json.dumps({'decision':decision,'summary':summary},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
