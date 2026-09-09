#!/usr/bin/env python3
"""Frozen-backbone hard-greedy physical-AW head distillation, frozen eval, STOP."""
import argparse, copy, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from cocap_voradj.models.small_step_ac import DeterministicAWActor
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.forward_final import make_env,scene_config,preflight,TEACHER
from cocap_voradj.training.trainer import set_global_config
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.runtime_semantics import assert_runtime,assert_only_changed,initial_state_fingerprint
from tools.distill_forward_final_actor_20260908 import load_actor,load_data,split_pairs,local_batch
from tools.train_forward_final_ppo_20260909 import BC,BC_SHA,tensor_hash
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file
from tools.run_forward_final_bridge_20260908 import atomic_json,run_episode,summarize


def continuous_env(scene,seed):
    cfg=scene_config(scene)
    cfg.update(action_mode='acceleration_angular_velocity_body',v_max=3.,a_max=.4,w_max=float(np.pi/6),decision_dt=.5)
    cfg.setdefault('dynamics',{})['linear_drag_coefficient']=.4/3
    cfg['yaw']={'init':'random_uniform'}  # Match canonical AW9 reset and RNG consumption.
    set_global_config(cfg);env=VorAdjEnv(copy.deepcopy(cfg),seed=seed);obs=env.reset()
    assert env.continuous_aw_action and env.collision_semantics=='synchronized_swept_v1'
    assert env.action_adapter.a_max==.4 and env.action_adapter.w_max==float(np.pi/6)
    assert env.reward_cfg['post_capture_coverage_window_steps']==500 and env.episode_max_length==3000
    assert not env.config['voradj'].get('capture_episode_ends_on_capture',False)
    assert env._ce_coverage_enabled() and env._ring_importance_ms_enabled()
    assert env._support_reward_capture_component_mode()=='approach_only'
    for p in env.pursuers:
        assert p.max_speed==3. and p.dt*p.N==.5 and p.coefficient_water_resistance==.4/3
    return env,obs


def physical_probe():
    report={}
    for scene in ('mixed','coverage'):
        a,oa=make_env(scene,2026099901);b,ob=continuous_env(scene,2026099901)
        # Continuous resets may have a different default heading: require exact initial parity.
        assert initial_state_fingerprint(a)==initial_state_fingerprint(b),'continuous reset drift'
        differences=assert_only_changed(assert_runtime(a),assert_runtime(b),['action_mode','adapter_a_max','adapter_w_max'])
        for x,y in zip(oa,ob):
            for k in x:np.testing.assert_array_equal(x[k],y[k])
        max_error=0.
        for action in range(9):
            da,db=copy.deepcopy(a),copy.deepcopy(b)
            x=da.step([action]*4,[4]*len(da.evaders));y=db.step([da.pursuers[0].action_list[action]]*4,[4]*len(db.evaders))
            for p,q in zip(da.pursuers,db.pursuers):
                left=np.array([p.x,p.y,p.theta,p.speed]);right=np.array([q.x,q.y,q.theta,q.speed]);max_error=max(max_error,float(abs(left-right).max()));np.testing.assert_allclose(left,right,atol=1e-10,rtol=0)
            np.testing.assert_allclose(x.rewards,y.rewards,atol=1e-8,rtol=0)
            assert x.dones==y.dones
        report[scene]={'runtime_changed':differences,'max_grid_center_state_error':max_error,'initial_fingerprint':initial_state_fingerprint(a),'all_9_grid_centers_tested':True}
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--probe-only',action='store_true');p.add_argument('--summarize-only',action='store_true');a=p.parse_args();out=a.out
    if a.summarize_only:
        records=json.loads((out/'records.json').read_text());assert len(records)==80
        elapsed=json.loads((out/'progress.json').read_text())['elapsed_seconds']
        finish_report(out,records,time.monotonic()-elapsed);return
    out.mkdir(parents=True,exist_ok=True);atomic_json(out/'physical_probe.json',physical_probe())
    if a.probe_only:return
    assert not (out/'launch.json').exists(),'fresh output required'
    atomic_json(out/'runtime_preflight.json',preflight());torch.manual_seed(2026099401);np.random.seed(2026099401)
    bc,_=load_actor(BC,a.device);actor=DeterministicAWActor(copy.deepcopy(bc.encoder),.4,np.pi/6).to(a.device).eval()
    actor.policy[:3].load_state_dict(bc.policy.state_dict())
    for param in actor.encoder.parameters():param.requires_grad_(False)
    backbone_sha=tensor_hash(actor.encoder.state_dict());assert backbone_sha==tensor_hash(bc.encoder.state_dict())
    arrays,manifest=load_data(ROOT/'artifacts/2026-09-08_forward_final/c1_dataset');train,valid=split_pairs(arrays['episode']);start=time.monotonic()
    atomic_json(out/'launch.json',{'pid':os.getpid(),'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'bc_parent_sha256':BC_SHA,'source_sha256':sha256_file(Path(__file__)),'engine_sha256':sha256_file(ROOT/'tools/run_forward_final_bridge_20260908.py'),'dataset_manifest_sha256':sha256_file(ROOT/'artifacts/2026-09-08_forward_final/c1_dataset/manifest.json'),'backbone_sha256':backbone_sha,'label':'teacher greedy AW9 physical center, never probability-weighted center','loss':'MSE on normalized hard-greedy physical command','epochs':30,'batch':2048,'lr':.0003,'selection':'minimum validation MSE; 30 fixed epochs; no heldout rollout selection','sample_std_normalized':.05,'sample_rule':'clip(mean_normalized + independent N(0,.05^2), -1,1); boundary mass explicitly measured','eval_seed':2026096101,'eval_episodes_per_scene_per_mode':20,'gate':'both mean/sample per-scene safe>=.9, capture mixed>=.9, collision<=.1; mean mission P90<=1.25 paired frozen BC argmax reference. Otherwise HOLD/FAIL representation; no RL in either case','ppo_updates':0})
    cached=[]
    with torch.no_grad():
        for offset in range(0,len(arrays['episode']),512):cached.append(actor.encoder(local_batch(arrays,np.arange(offset,min(offset+512,len(arrays['episode']))),a.device)))
    features=torch.cat(cached);grid=bc.action_grid;target=grid[torch.as_tensor(arrays['greedy_action'],device=a.device)]/actor.scale
    optimizer=torch.optim.Adam(actor.policy.parameters(),lr=.0003,eps=1e-5);history=[];best=float('inf')
    def score(ids):
        with torch.no_grad():
            y=torch.tanh(actor.policy(features[ids]));truth=target[ids];near=((y[:,None,:]*actor.scale-grid[None,:,:])/actor.scale).square().sum(-1).argmin(-1)
            labels=arrays['greedy_action'][ids]
            return {'mse_normalized':float((y-truth).square().mean()),'physical_mae':((y-truth).abs()*actor.scale).mean(0).cpu().tolist(),'nearest_center_agreement':float((near==torch.as_tensor(labels,device=a.device)).float().mean())}
    for epoch in range(1,31):
        order=np.random.permutation(train)
        for offset in range(0,len(order),2048):
            ids=order[offset:offset+2048];loss=(torch.tanh(actor.policy(features[ids]))-target[ids]).square().mean();assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.policy.parameters(),1.);optimizer.step()
        assert tensor_hash(actor.encoder.state_dict())==backbone_sha
        scores=score(valid);history.append({'epoch':epoch,**scores})
        if scores['mse_normalized']<best:
            best=scores['mse_normalized'];best_state=copy.deepcopy(actor.state_dict());best_epoch=epoch
        atomic_json(out/'progress.json',{'status':'distilling','epoch':epoch,'total_epochs':30,'elapsed_seconds':time.monotonic()-start})
    actor.load_state_dict(best_state);path=out/'continuous_actor.pt';torch.save({'actor':best_state,'backbone_sha256':backbone_sha,'bc_parent_sha256':BC_SHA,'best_epoch':best_epoch},path)
    imitation={s:score(ids) for s,ids in [('train',train),('validation',valid)]}
    imitation['validation_groups']={k:{str(v):score(valid[arrays[k][valid]==v]) for v in np.unique(arrays[k][valid])} for k in ('phase_id','role_id')}
    atomic_json(out/'distillation.json',{'history':history,'scores':imitation,'checkpoint_sha256':sha256_file(path),'backbone_bit_exact':True,'best_epoch':best_epoch})
    teacher=CoCapIQN.load(str(TEACHER),device=a.device).eval();actor_hash=tensor_hash(actor.state_dict());records=[]
    for mode in ('mean','sample'):
        for i in range(20):
            for scene in ('mixed','coverage'):
                seed=2026096101+i+(100000 if scene=='coverage' else 0);rng=np.random.default_rng(seed);clipped=0;count=0
                @torch.no_grad()
                def policy(local,q):
                    nonlocal clipped,count
                    obs={k:torch.as_tensor(np.stack([o[k] for o in local]),device=a.device) for k in local[0]}
                    command=actor(obs).cpu().numpy();normalized=command/actor.scale.cpu().numpy()
                    if mode=='sample':
                        noisy=normalized+rng.normal(0,.05,size=normalized.shape);clipped+=int((abs(noisy)>1).sum());count+=noisy.size;command=np.clip(noisy,-1,1)*actor.scale.cpu().numpy()
                    nearest=(((command[:,None,:]-grid.cpu().numpy()[None,:,:])/actor.scale.cpu().numpy())**2).sum(-1).argmin(-1)
                    return command,nearest
                def progress(step):
                    elapsed=time.monotonic()-start
                    atomic_json(out/'progress.json',{'status':'frozen_eval','mode':mode,'completed':len(records),'total':80,'current_step':step,'elapsed_seconds':elapsed,'episodes_per_minute':60*len(records)/max(elapsed,1),'eta_seconds':elapsed/max(len(records),1)*(80-len(records))})
                row=run_episode(teacher,scene,seed,a.device,env_factory=continuous_env,physical_policy=policy,on_progress=progress)
                row.update(policy_mode='continuous_'+mode,clipped_coordinate_fraction=clipped/max(count,1),agreement_definition='nearest AW9 center vs teacher greedy; not exact action equality',entropy_note='discrete entropy is unavailable; inherited zero placeholder must not be interpreted as continuous entropy')
                records.append(row);atomic_json(out/'records.json',records);progress(row['length'])
    assert tensor_hash(actor.state_dict())==actor_hash
    finish_report(out,records,start)


def finish_report(out,records,start):
    summary={m:summarize([r for r in records if r['policy_mode']=='continuous_'+m]) for m in ('mean','sample')}
    reference=json.loads((ROOT/'artifacts/2026-09-08_forward_final/c3_formal100/bc_argmax/report.json').read_text())['records'];ref={(r['scene'],r['seed']):r for r in reference}
    checks={};paired={}
    for mode in ('mean','sample'):
        for scene in ('mixed','coverage'):
            r=[r for r in records if r['policy_mode']=='continuous_'+mode and r['scene']==scene]
            for x in r:assert x['initial_state_fingerprint']==ref[scene,x['seed']]['initial_state_fingerprint']
            s=summary[mode][scene];checks[mode+'/'+scene]=s['safe_complete_rate']>=.9 and s['collision_rate']<=.1 and (scene=='coverage' or s['captured_rate']>=.9)
            common=[x for x in r if x['safe_complete'] and ref[scene,x['seed']]['safe_complete']]
            paired[mode+'/'+scene]={'common_safe':len(common),'mean_seconds_delta':float(np.mean([x['mission_seconds']-ref[scene,x['seed']]['mission_seconds'] for x in common])) if common else None}
            if mode=='mean':
                rt=[ref[scene,x['seed']]['mission_seconds'] for x in r if ref[scene,x['seed']]['safe_complete']]
                checks[mode+'/'+scene+'/p90']=bool(s['full_mission_seconds_completed']['p90'] is not None and s['full_mission_seconds_completed']['p90']<=1.25*np.percentile(rt,90))
    decision='REPRESENTATION_BRIDGE_PASS_NOT_RL_PASS' if all(checks.values()) else 'HOLD_CONTINUOUS_REPRESENTATION'
    atomic_json(out/'report.json',{'decision':decision,'checks':checks,'summary':summary,'paired_vs_bc_argmax':paired,'actor_frozen_during_eval':True,'records':records,'next':'STOP; no continuous PPO regardless of outcome'})
    atomic_json(out/'progress.json',{'status':'complete','completed':80,'total':80,'eta_seconds':0,'elapsed_seconds':time.monotonic()-start,'decision':decision})

if __name__=='__main__':main()
