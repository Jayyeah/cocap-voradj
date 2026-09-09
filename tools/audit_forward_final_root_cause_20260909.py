#!/usr/bin/env python3
"""P0 frozen reward replay and P1 same-observation context counterexamples; no training."""
import argparse, copy, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from tools import run_forward_final_bridge_20260908 as engine
from tools.evaluate_forward_final_ppo_20260909 import load_frozen_actor
from tools.train_forward_final_ppo_20260909 import tensor_hash
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file
from cocap_voradj.training.forward_final import make_env,preflight,TEACHER
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.continuous.central_schema import build_central_global_obs


def central(env):
    return build_central_global_obs(env,max_agents=4,max_evaders=8,max_obstacles=5,self_feature_dim=9)


def same(a,b):
    x,y=central(a),central(b)
    return all(np.array_equal(x[k],y[k]) for k in x)


def context_probe(out):
    env,_=make_env('mixed',2026098101)
    for e in env.evaders:e.deactivated=True
    env.post_capture_started=True;env.post_capture_step=10
    near=copy.deepcopy(env);near.post_capture_step=499
    assert same(env,near)
    commands=[4]*4
    a=env.step(commands,[None]);b=near.step(commands,[None])
    assert not all(a.dones) and all(b.dones)
    report={'classification':'RUNTIME FACT','post_window':{'central_input_bit_identical':True,'same_action':commands,'post_steps_before':[10,499],'done_after':[all(a.dones),all(b.dones)],'decision':'STATE ALIASING / NON-MARKOV CRITIC INPUT'},'scope':'controlled context intervention, not an estimate of frequency or causal PPO effect'}
    env,_=make_env('coverage',2026098102)
    # Probe the actual success consumer with identical current geometry metrics.
    env.distribution_hold_steps=0;other=copy.deepcopy(env);other.distribution_hold_steps=29
    report['ce_hold']={'central_input_bit_identical':same(env,other),'hold_before':[0,29], 'required':env.reward_cfg['coverage_ce_success_hold_steps'],'consumer':'_voradj_coverage_converged: consecutive history is not a function of current geometry'}
    assert report['ce_hold']['central_input_bit_identical']
    engine.atomic_json(out/'context_probe.json',report)


def replay(out,device,count):
    engine.atomic_json(out/'runtime_preflight.json',preflight())
    teacher=CoCapIQN.load(str(TEACHER),device=device).eval()
    checkpoint=ROOT/'artifacts/2026-09-09_forward_final_d/lr_probe/low_lr/actor_step_000512.pt'
    if not checkpoint.exists():
        candidates=list((ROOT/'artifacts/2026-09-09_forward_final_d/lr_probe').rglob('actor_step_000512.pt'))
        candidates=[p for p in candidates if 'low' in str(p)]
        assert len(candidates)==1,candidates
        checkpoint=candidates[0]
    models=[('bc',None,ROOT/'artifacts/2026-09-09_forward_final_d/smoke_task_eval20/bc_parent/report.json'),('ppo_low_lr',checkpoint,ROOT/'artifacts/2026-09-09_forward_final_d/lr_probe/low_lr_eval20/report.json'),('iqn',None,None)]
    start=time.monotonic();records=[]
    engine.atomic_json(out/'launch.json',{'pid':os.getpid(),'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'episodes_per_scene_policy':count,'seed':2026098101,'reward_ce_speed':.0005,'training_updates':0,'source_sha256':sha256_file(Path(__file__)),'scope':'same diagnostic seeds, training reward on frozen evaluation reset distribution; not online recovery-pool distribution','ppo_checkpoint':str(checkpoint)})
    for name,path,ref in models:
        actor=None if name=='iqn' else load_frozen_actor(path,device)[0]
        before=tensor_hash((actor or teacher).state_dict())
        reference={}
        if ref:
            if not ref.exists():
                raise FileNotFoundError(ref)
            reference={(r['scene'],r['seed']):r for r in json.loads(ref.read_text())['records']}
        for i in range(count):
            for scene in ('mixed','coverage'):
                rows=[]
                def factory(s,seed):
                    env,obs=make_env(s,seed)
                    env.reward_cfg['coverage_ce_speed_weight']=.0005
                    original=env.step
                    def step(actions,evaders):
                        phase=2 if s=='coverage' else int(not any(not e.deactivated for e in env.evaders))
                        context=[env.episode_step,env.post_capture_step,env.distribution_hold_steps]
                        outcome=original(actions,evaders)
                        meta=[v['replay_metadata'] for v in outcome.infos]
                        assert all(m['coverage_ce_speed_weight']==.0005 for m in meta)
                        row={'phase':phase,'context':context,'reward':list(map(float,outcome.rewards))}
                        for key in ('reward_ce_center','reward_ce_control','reward_ce_pbrs','reward_ce_terminal_correction','reward_capture','reward_terminal','reward_safety'):
                            row[key]=[m[key] for m in meta]
                        rows.append(row)
                        return outcome
                    env.step=step
                    return env,obs
                engine.make_env=factory
                seed=2026098101+i+(100000 if scene=='coverage' else 0)
                def progress(step):
                    elapsed=time.monotonic()-start
                    engine.atomic_json(out/'progress.json',{'status':'running','policy':name,'completed':len(records),'total':6*count,'current_step':step,'elapsed_seconds':elapsed,'episodes_per_minute':60*len(records)/max(elapsed,1),'eta_seconds':elapsed/max(len(records),1)*(6*count-len(records))})
                r=engine.run_episode(teacher,scene,seed,device,actor=actor,policy_mode='iqn_greedy' if actor is None else 'bc_sample',on_progress=progress)
                if reference:
                    old=reference[scene,seed]
                    for k in ('initial_state_fingerprint','safe_complete','captured','collision','length','mission_seconds','capture_seconds','recovery_seconds','policy_diagnostics'):
                        assert r[k]==old[k],(name,scene,seed,k)
                r.update(policy=name,reference_parity=bool(reference),raw_path=f'rows/{name}_{scene}_{seed}.npz')
                p=out/r['raw_path'];p.parent.mkdir(exist_ok=True)
                np.savez_compressed(p,**{k:np.asarray([v[k] for v in rows]) for k in rows[0]})
                r['raw_sha256']=sha256_file(p);records.append(r)
                engine.atomic_json(out/'records.json',records);progress(r['length'])
        assert before==tensor_hash((actor or teacher).state_dict())
    engine.atomic_json(out/'progress.json',{'status':'complete','completed':len(records),'total':len(records),'elapsed_seconds':time.monotonic()-start,'eta_seconds':0})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--episodes',type=int,default=20);p.add_argument('--probe-only',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);context_probe(a.out)
    if not a.probe_only:
        assert not (a.out/'launch.json').exists(),'fresh output required'
        replay(a.out,a.device,a.episodes)
