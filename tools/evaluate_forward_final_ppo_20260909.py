#!/usr/bin/env python3
"""Frozen D health screen using exactly the C3 full-task transition engine."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.forward_final import CONTRACT,TEACHER,TEACHER_SHA,preflight
from tools.train_forward_final_ppo_20260909 import BC,BC_SHA,SCHEMA,POLICY_CONTRACT
from tools.distill_forward_final_actor_20260908 import load_actor
from tools.run_forward_final_bridge_20260908 import run_episode,atomic_json,summarize
from tools.evaluate_forward_final_actor_20260908 import policy_summary,paired_delta
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file


def load_frozen_actor(checkpoint,device):
    actor,_=load_actor(BC,device)
    sha=BC_SHA;step=0
    if checkpoint:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert payload['schema']==SCHEMA and payload['policy_contract']==POLICY_CONTRACT
        assert payload['contract']==CONTRACT and payload['bc_parent_sha256']==BC_SHA and payload['teacher_sha256']==TEACHER_SHA
        actor.load_state_dict(payload['actor_state_dict'],strict=True);sha=sha256_file(checkpoint);step=payload['step']
    actor.eval()
    return actor,sha,step


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);p.add_argument('--actor-checkpoint',type=Path)
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed',type=int,default=2026098101);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();out=args.output_root
    if (out/'launch.json').exists():raise ValueError('Fresh output required')
    atomic_json(out/'runtime_preflight.json',preflight())
    actor,sha,step=load_frozen_actor(args.actor_checkpoint,args.device)
    actor.eval();teacher=CoCapIQN.load(str(TEACHER),device=args.device).eval()
    launch={'contract':CONTRACT,'bc_parent_sha256':BC_SHA,'actor_sha256':sha,'actor_checkpoint':str(args.actor_checkpoint or BC),'training_steps':step,'mode':'bc_sample','episodes_per_scene':args.episodes,'seed_base':args.seed,'evaluator_sha256':sha256_file(ROOT/'tools/run_forward_final_bridge_20260908.py'),'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'scope':'finite-budget frozen health screen, not formal100 or PPO advancement PASS'}
    atomic_json(out/'launch.json',launch);records=[];start=time.monotonic()
    for i in range(args.episodes):
        for scene in ('mixed','coverage'):
            def progress(s):
                elapsed=time.monotonic()-start
                atomic_json(out/'progress.json',{'status':'running','completed':len(records),'total':args.episodes*2,'scene':scene,'current_step':s,'elapsed_seconds':elapsed,'eta_seconds':elapsed/max(1,len(records))*(args.episodes*2-len(records))})
            row=run_episode(teacher,scene,args.seed+i+(100000 if scene=='coverage' else 0),args.device,actor=actor,policy_mode='bc_sample',on_progress=progress)
            records.append(row)
            with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            progress(row['length'])
    atomic_json(out/'report.json',{**launch,'status':'complete','summary':summarize(records),'policy_summary':policy_summary(records),'records':records})
    atomic_json(out/'progress.json',{'status':'complete','completed':len(records),'eta_seconds':0,'elapsed_seconds':time.monotonic()-start})

if __name__=='__main__':main()
