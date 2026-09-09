#!/usr/bin/env python3
"""CPU frozen-BC shadow rollout: reward/GAE phase audit, no optimizer steps."""
import argparse,json,random,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from tools.train_forward_final_ppo_20260909 import make_trainer,FinalMissionStream,collect_transition,empty_rollout,stack_rollout,tensor_hash
from cocap_voradj.training.small_step_ac import compute_gae
from tools.run_forward_final_bridge_20260908 import atomic_json
COMPONENTS=('reward_total','reward_capture','reward_coverage','reward_terminal','reward_safety','reward_ce_center','reward_ce_control','reward_ce_pbrs','reward_ce_terminal_correction','reward_support_blend_capture','reward_support_blend_coverage')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);args=p.parse_args();out=args.output_root
    if (out/'report.json').exists():raise ValueError('Fresh output required')
    seed=2026097101;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    trainer=make_trainer('cpu',seed);stream=FinalMissionStream(seed,out)
    actor_before=tensor_hash(trainer.actor.state_dict());critic_before=tensor_hash(trainer.value.state_dict())
    diagnostics=[];all_batches=[];rollout=empty_rollout();phases=[];roles=[];episodes=[]
    original_step=stream.step
    def tapped(indices):
        result=original_step(indices)
        diagnostics.append([{k:float(info['replay_metadata'].get(k,0)) for k in COMPONENTS} for info in result.infos])
        roles.append([info['reward_role'] for info in result.infos])
        return result
    stream.step=tapped
    for step in range(512):
        phase='pure_coverage' if stream.task=='voradj_coverage' else 'pre_capture' if any(not e.deactivated for e in stream.env.evaders) else 'post_capture'
        phases.append([phase]*4)
        row,episode=collect_transition(trainer,stream)
        for k,v in row.items():rollout[k].append(v)
        if episode:episodes.append({'step':step+1,**episode})
        if (step+1)%256==0:all_batches.append(stack_rollout(rollout));rollout=empty_rollout()
    windows=[]
    for w,batch in enumerate(all_batches):
        active=torch.as_tensor(batch['active_mask']).bool();rewards=torch.as_tensor(batch['rewards']);values=torch.as_tensor(batch['values']);next_values=torch.as_tensor(batch['next_values'])
        term=torch.as_tensor(batch['terminated']);trunc=torch.as_tensor(batch['truncated'])
        adv,returns=compute_gae(rewards,values,next_values,term,active,gamma=.99,gae_lambda=.95,truncated=trunc)
        zero_adv,_=compute_gae(rewards,torch.zeros_like(values),torch.zeros_like(values),term,active,gamma=.99,gae_lambda=.95,truncated=trunc)
        valid=adv[active];center=valid.mean();scale=valid.std(unbiased=False).clamp_min(1e-6);normalized=(adv-center)/scale
        phase=np.asarray(phases[w*256:(w+1)*256]);role=np.asarray(roles[w*256:(w+1)*256]);components=diagnostics[w*256:(w+1)*256]
        groups={}
        masks={**{'phase/'+k:phase==k for k in np.unique(phase)},**{'role/'+k:role==k for k in np.unique(role)}}
        for name,mask in masks.items():
            mask=torch.as_tensor(mask)&active
            component={k:np.array([[row[k] for row in t] for t in components])[mask.numpy()] for k in COMPONENTS}
            groups[name]={'rows':int(mask.sum()),'reward_mean':float(rewards[mask].mean()),'raw_adv_mean':float(adv[mask].mean()),'normalized_adv_mean':float(normalized[mask].mean()),'raw_adv_positive_fraction':float((adv[mask]>0).float().mean()),'normalized_adv_positive_fraction':float((normalized[mask]>0).float().mean()),'positive_raw_flipped_negative_fraction':float(((adv[mask]>0)&(normalized[mask]<0)).float().mean()),'zero_value_adv_mean':float(zero_adv[mask].mean()),'reward_component_sums':{k:float(v.sum()) for k,v in component.items()}}
        windows.append({'window':w,'rows':int(active.sum()),'global_raw_adv_mean':float(center),'global_raw_adv_std':float(scale),'reward_total':float(rewards[active].sum()),'done_transitions':int((term|trunc).any(1).sum()),'groups':groups})
    assert actor_before==tensor_hash(trainer.actor.state_dict()) and critic_before==tensor_hash(trainer.value.state_dict())
    report={'scope':'CPU frozen BC shadow512, NOT bit-exact replay of prior CUDA sampling; neither actor nor critic updates; each256-window uses same cold critic and ValueNorm initial state','seed':seed,'optimizer_updates':0,'actor_and_critic_unchanged':True,'windows':windows,'episodes':episodes,'interpretation_limits':['A negative normalized advantage does not prove an incorrect gradient or reward: state-independent baseline subtraction is unbiased in expectation; finite rollout phase mixture and a cold value baseline may still impair optimization.','Reward component columns overlap by construction; do not sum all columns as disjoint reward terms.','This audit does not compare learned critic calibration or demonstrate a causal fix.']}
    atomic_json(out/'report.json',report)
    print(json.dumps({**report,'windows':[{**v,'groups':{k:{kk:vv for kk,vv in g.items() if kk!='reward_component_sums'} for k,g in v['groups'].items()}} for v in windows]},indent=2))

if __name__=='__main__':main()
