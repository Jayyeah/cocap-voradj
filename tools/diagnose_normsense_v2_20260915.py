#!/usr/bin/env python3
"""Frozen matched-seed Coverage transfer and matched-motion visibility; no updates."""
import sys,json,copy,argparse,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from tools import forward_final_single_task_20260915 as st
from cocap_voradj.envs.density_sensing import enable_v2,runtime_metadata,CALIBRATED_K
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.small_step_ac import tensor_tree
OLD=ROOT.parent/'cocap-voradj-small-step-ac'
OUT=ROOT/'artifacts/2026-09-15_normsense_v2'

def stream_for(task,seed,v2):
    stream=st.SingleTaskStream(task,seed)
    if v2:
        cfg=enable_v2(st.task_config(task));st.p.set_global_config(cfg)
        env=VorAdjEnv(cfg,seed=seed);stream.observations=env.reset();stream.envs[task]=env
        stream.apf_agents={task:[st.p.ApfAgent(e.a,e.w) for e in env.evaders]}
        stream.telemetry=st.Telemetry(env,task)
    stream.set_clock(0)
    return stream

def actor_obs(observations,device):
    ids=[i for i,o in enumerate(observations) if o is not None]
    return ids,tensor_tree({k:np.stack([observations[i][k] for i in ids]) for k in observations[ids[0]]},device)

def token_counts(obs):return [int(o['masks'][o['types']==3].sum()) for o in obs if o is not None]

@torch.no_grad()
def coverage(device,episodes,part=0,parts=1):
    checkpoint=OLD/'artifacts/2026-09-15_single_task/coverage/step_200000.pt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    actor=st.p.make_actor(st.contract()['scratch'],2026091501,device)
    actor.load_state_dict(payload['trainer']['actor']);actor.eval()
    h=st.p.tensor_hash(actor.state_dict());records=[]
    for mode in ('argmax','sample'):
        for episode in range(episodes):
            if ((0 if mode=='argmax' else episodes)+episode)%parts!=part:continue
            seed=2026191501+episode
            for v2 in (False,True):
                st.p.seed_all(seed);stream=stream_for('coverage',seed,v2)
                occupancy=[];disagree=[];tv=[]
                for step in range(3000):
                    ids,obs=actor_obs(stream.observations,device);d=actor.distribution(obs)
                    # Same state, counterfactual obstacle tokens. Coverage has no enemy role-history difference.
                    original=stream.env.config
                    stream.env.config=st.task_config('coverage') if v2 else enable_v2(original)
                    other=stream.env.get_observations();stream.env.config=original
                    _,other_obs=actor_obs(other,device);od=actor.distribution(other_obs)
                    disagree.extend((d.logits.argmax(-1)!=od.logits.argmax(-1)).cpu().tolist())
                    tv.extend((.5*(d.probs-od.probs).abs().sum(-1)).cpu().tolist())
                    occupancy.extend(token_counts(stream.observations))
                    actions=np.full(4,4);actions[ids]=(d.logits.argmax(-1) if mode=='argmax' else d.sample()).cpu().numpy()
                    result=stream.step(actions)
                    if all(result.dones):break
                assert all(result.dones)
                row=stream.telemetry.finish(stream.env)
                row.update(k=CALIBRATED_K,mode=mode,seed=seed,contract='v2' if v2 else 'legacy',obstacle_token_mean=float(np.mean(occupancy)),obstacle_token_occupancy=float(np.mean(np.array(occupancy)>0)),same_state_argmax_disagreement=float(np.mean(disagree)),same_state_action_total_variation=float(np.mean(tv)))
                records.append(row)
                print('coverage',mode,episode,'v2',v2,'success',row['ce_success'],'steps',row['length'],flush=True)
                st.p.write_json(OUT/f'coverage_progress_part_{part}.json',dict(pid=os.getpid(),completed=len(records),total=4*episodes))
    st.p.write_json(OUT/f'coverage_part_{part}.json',dict(k=CALIBRATED_K,records=records,actor_sha256=h,checkpoint=str(checkpoint),checkpoint_sha256=st.old.sha(checkpoint)))
    if parts>1:return
    summary={}
    for mode in ('argmax','sample'):
        for name in ('legacy','v2'):
            rows=[r for r in records if r['mode']==mode and r['contract']==name]
            summary[mode+'/'+name]={k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in ('ce_success','collision','ce_rms','ce_max','area_cv','time_to_ce','obstacle_token_mean','obstacle_token_occupancy','same_state_argmax_disagreement','same_state_action_total_variation')}
    assert h==st.p.tensor_hash(actor.state_dict())
    for i in range(0,len(records),2):assert records[i]['initial_state_fingerprint']==records[i+1]['initial_state_fingerprint']
    rates=[summary[m+'/v2']['ce_success'] for m in ('argmax','sample')]
    # Predeclared operational transfer gate; mode-level, 20 matched pairs each.
    neutral=all(summary[m+'/v2']['ce_success']>=summary[m+'/legacy']['ce_success']-.05 and summary[m+'/v2']['collision']<=summary[m+'/legacy']['collision']+.05 and all(summary[m+'/v2'][k]<=summary[m+'/legacy'][k]*1.15 for k in ('ce_rms','ce_max','area_cv','time_to_ce')) for m in ('argmax','sample'))
    usable=all(summary[m+'/v2']['ce_success']>=.9 and summary[m+'/v2']['collision']<=.1 for m in ('argmax','sample'))
    verdict='COVERAGE_TRANSFER_NEUTRAL' if neutral else 'COVERAGE_TRANSFER_SHIFT_BUT_USABLE' if usable else 'COVERAGE_TRANSFER_BREAKS_POLICY'
    report=dict(k=CALIBRATED_K,checkpoint=str(checkpoint),checkpoint_sha256=st.old.sha(checkpoint),actor_sha256=h,optimizer_updates=0,episodes_per_mode_contract=episodes,matched_initial_fingerprints=True,verdict=verdict,summary=summary,records=records)
    st.p.write_json(OUT/'coverage_transfer.json',report)
    print(verdict,flush=True)


def visibility(episodes):
    rows=[]
    # Exogenous random AW9 actions, same physical rollout queried under both sensors.
    # This holds APF/positions/resets fixed and avoids policy reaction confounding.
    for ep in range(episodes):
        seed=2026391501+ep;st.p.seed_all(seed);stream=stream_for('capture',seed,False)
        rng=np.random.default_rng(seed);counts={'legacy':[],'v2':[]};tokens={'legacy':[],'v2':[]}
        configs={'legacy':stream.env.config,'v2':enable_v2(stream.env.config)}
        for t in range(512):
            for name in counts:
                stream.env.config=configs[name]
                counts[name].append(next(iter(stream.env._vct_ls_direct_detect_counts().values()),0))
                tokens[name].extend(token_counts(stream.env.get_observations()))
            stream.env.config=configs['legacy']
            result=stream.step(rng.integers(0,9,4))
            if all(result.dones):break
        for name,c in counts.items():
            hits=np.flatnonzero(c)
            rows.append(dict(seed=seed,contract=name,steps=len(c),initially_visible=c[0]>0,first_detection_step=int(hits[0]) if len(hits) else None,censored=not len(hits),zero_fraction=float(np.mean(np.array(c)==0)),all_four_fraction=float(np.mean(np.array(c)==4)),detector_counts=np.bincount(c,minlength=5).tolist(),obstacle_token_mean=float(np.mean(tokens[name]))))
        print('visibility',ep,flush=True)
    summary={}
    for name in ('legacy','v2'):
        rr=[r for r in rows if r['contract']==name];hist=np.sum([r['detector_counts'] for r in rr],0)
        summary[name]=dict(initially_visible_probability=float(np.mean([r['initially_visible'] for r in rr])),never_detected_fraction=float(np.mean([r['censored'] for r in rr])),first_detection_step_detected_only=st.p.stats([r['first_detection_step'] for r in rr if not r['censored']]),detector_count_distribution=(hist/hist.sum()).tolist(),zero_detector_fraction=float(hist[0]/hist.sum()),all_four_fraction=float(hist[4]/hist.sum()),obstacle_token_mean=float(np.mean([r['obstacle_token_mean'] for r in rr])))
    # Most states globally visible is a veto, not an excuse to retain k.
    st.p.write_json(OUT/'capture_visibility.json',dict(k=CALIBRATED_K,method='matched physical states, random AW9 motion, 512-step cap or native terminal; censored latency explicit',episodes=episodes,global_visibility_veto=summary['v2']['all_four_fraction']>=.5,summary=summary,records=rows))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['coverage','visibility']);parser.add_argument('--device',default='cpu');parser.add_argument('--episodes',type=int,default=20);parser.add_argument('--part',type=int,default=0);parser.add_argument('--parts',type=int,default=1);a=parser.parse_args()
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    coverage(a.device,a.episodes,a.part,a.parts) if a.mode=='coverage' else visibility(a.episodes)
