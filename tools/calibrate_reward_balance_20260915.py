#!/usr/bin/env python3
"""Frozen full-task BC actor, actual PPO surrogate gradients; zero updates."""
import sys,json,os,copy,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from tools import forward_final_single_task_20260915 as st
from tools.train_forward_final_ppo_20260909 import FinalMissionStream,collect_transition
from cocap_voradj.training.small_step_ac import compute_gae,tensor_tree
from cocap_voradj.envs.density_sensing import POLICY,CALIBRATED_K
OLD=ROOT.parent/'cocap-voradj-small-step-ac';OUT=ROOT/'artifacts/2026-09-15_normsense_v2'
PHASES=('capture','support','coverage')

def gradients(trainer,batch,roles,components,alpha=1.):
    device=trainer.device;t=lambda k:torch.as_tensor(batch[k],device=device)
    rewards=t('rewards').clone()
    capture_component=torch.as_tensor(components,device=device)
    rewards+=(alpha-1)*capture_component
    active=t('active_mask').bool()
    values=trainer._denormalize_values(t('values')).detach();nxt=trainer._denormalize_values(t('next_values')).detach()
    adv,targets=compute_gae(rewards,values,nxt,t('terminated'),active,gamma=.99,gae_lambda=.95,truncated=t('truncated'),episode_end=t('episode_end'))
    valid=adv[active];norm=(adv-valid.mean())/valid.std(unbiased=False).clamp_min(1e-6)
    flat={k:np.asarray(v).reshape(-1,*np.asarray(v).shape[2:]) for k,v in batch['local_obs'].items()}
    actions=t('latent').reshape(-1).long();logp=t('log_prob').reshape(-1).detach()
    total=int(active.sum());vectors=[];report={};all_params=[p for p in trainer.actor.parameters() if p.requires_grad]
    for phase in PHASES:
        mask=active & torch.as_tensor(roles==phase,device=device);ids=np.flatnonzero(mask.cpu().numpy().reshape(-1))
        trainer.actor.zero_grad(set_to_none=True);loss_total=0.;entropy_total=0.
        for start in range(0,len(ids),64):
            ix=ids[start:start+64];obs=tensor_tree({k:v[ix] for k,v in flat.items()},device)
            lp,entropy=trainer.actor.evaluate_indices(obs,actions[ix]);ratio=(lp-logp[ix]).exp()
            surrogate=torch.minimum(ratio*norm.reshape(-1)[ix],ratio.clamp(.8,1.2)*norm.reshape(-1)[ix])
            loss=-surrogate.sum()/total;loss.backward();loss_total+=float(loss.detach());entropy_total+=float(entropy.detach().sum())/total
        vector=torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel(),device=device)) for p in all_params]).double().cpu()
        vectors.append(vector)
        get=lambda x:st.p.stats(x[mask].detach().cpu().numpy())
        report[phase]=dict(sample_rows=len(ids),raw_reward=get(rewards),gae_target=get(targets),raw_advantage=get(adv),normalized_advantage=get(norm),policy_loss=loss_total,entropy_contribution=entropy_total,gradient_norm=float(vector.norm()),within_phase_mean_gradient_norm=float(vector.norm())*total/max(1,len(ids)))
    cosine={}
    for i in range(3):
        for j in range(i+1,3):
            denom=float(vectors[i].norm()*vectors[j].norm())
            cosine[PHASES[i]+'__'+PHASES[j]]=float(vectors[i]@vectors[j])/denom if denom>1e-15 else None
    return dict(phases=report,cosine=cosine,global_rows=total,capture_to_coverage_norm_ratio=report['capture']['gradient_norm']/max(report['coverage']['gradient_norm'],1e-30)), vectors

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--windows',type=int,default=16);args=parser.parse_args()
    torch.set_num_threads(1);seed=2026091401;st.p.seed_all(seed)
    trainer=st.p.make_trainer(st.contract()['scratch'],seed,args.device)
    checkpoint=OLD/'artifacts/2026-09-08_forward_final/c2_distillation/actor_epoch_030.pt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert st.old.sha(checkpoint)=='7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd'
    trainer.actor.load_state_dict(payload['actor_state_dict'],strict=True)
    for param in trainer.actor.parameters():param.requires_grad_(True)
    trainer.actor.eval();trainer.value.eval()
    actor_hash=st.p.tensor_hash(trainer.actor.state_dict());value_hash=st.p.tensor_hash(trainer.value.state_dict())
    stream=FinalMissionStream(seed,OUT/'gradient_stream',sensing_policy=POLICY)
    pending_roles=[];pending_components=[];original=stream.step
    def tapped(indices):
        result=original(indices);metas=[r['replay_metadata'] for r in result.infos]
        pending_roles.append([m['reward_role'] for m in metas])
        # Capture terminal is present only on capture event; CE terminal remains untouched.
        capture_terminal=bool(stream.env.last_capture_events)
        pending_components.append([m['reward_capture']+(m['reward_terminal'] if capture_terminal else 0.) for m in metas])
        return result
    stream.step=tapped
    reports=[];batches=[];groups=[];comps=[];episodes=[];aggregate=None
    for w in range(args.windows):
        rollout=st.p.empty_rollout();pending_roles.clear();pending_components.clear()
        for step in range(256):
            row,ep=collect_transition(trainer,stream)
            for k,v in row.items():rollout[k].append(v)
            if ep:episodes.append(ep)
        batch=st.p.stack_rollout(rollout);roles=np.array(pending_roles);components=np.array(pending_components,dtype=np.float32)
        result,vec=gradients(trainer,batch,roles,components)
        reports.append(result);batches.append(batch);groups.append(roles);comps.append(components)
        if aggregate is None:aggregate=[v.clone() for v in vec]
        else:
            for a,v in zip(aggregate,vec):a.add_(v)
        print('gradient',w,'rows',{k:v['sample_rows'] for k,v in result['phases'].items()},'norms',{k:v['gradient_norm'] for k,v in result['phases'].items()},flush=True)
        st.p.write_json(OUT/'reward_balance_progress.json',dict(pid=os.getpid(),windows_complete=w+1,windows_total=args.windows))
    # Use paired windows with both components represented, avoiding pure-phase zeros.
    paired=[r for r in reports if all(r['phases'][k]['sample_rows']>=32 for k in ('capture','coverage'))]
    medians={k:float(np.median([r['phases'][k]['gradient_norm'] for r in paired])) if paired else None for k in PHASES}
    adequate=len(paired)>=4 and sum(r['phases']['support']['sample_rows'] for r in reports)>=32
    alpha=float(np.clip(medians['coverage']/max(medians['capture'],1e-30),.25,1.)) if adequate else None
    cosine={}
    for i in range(3):
        for j in range(i+1,3):
            denom=float(aggregate[i].norm()*aggregate[j].norm())
            cosine[PHASES[i]+'__'+PHASES[j]]=float(aggregate[i]@aggregate[j])/denom if denom else None
    # Recompute fixed batch PPO signals under the declared static intervention.
    scaled=[]
    if alpha is not None and alpha<1:
        for index,r in enumerate(reports):
            if r not in paired:continue
            sr,_=gradients(trainer,batches[index],groups[index],comps[index],alpha)
            scaled.append(dict(window=index,**sr))
    assert actor_hash==st.p.tensor_hash(trainer.actor.state_dict()) and value_hash==st.p.tensor_hash(trainer.value.state_dict())
    assert trainer.update_count==0 and not trainer.actor_optimizer.state and not trainer.value_optimizer.state
    raw_path=OUT/'frozen_reward_rollouts.pt';torch.save(dict(batches=batches,roles=groups,capture_components=comps),raw_path)
    report=dict(k=CALIBRATED_K,schema='static-capture-gradient-calibration-v1',seed=seed,actor_checkpoint=str(checkpoint),actor_checkpoint_sha256=st.old.sha(checkpoint),actor_hash=actor_hash,value_hash=value_hash,critic='fresh random canonical central V, initial ValueNorm; frozen no updates',sensing=POLICY,windows=reports,paired_windows=len(paired),adequate=adequate,robust_statistic='median global-loss gradient norm across same windows with >=32 direct-capture and >=32 coverage rows; at least 4 windows and >=32 support rows required',robust_norms=medians,alpha_capture=alpha,coverage_scale=1.,support_capture_scale=alpha,support_coverage_scale=1.,safety_scale=1.,aggregate_gradient_cosine=cosine,scaled_fixed_rollout_gradients=scaled,optimizer_updates=0,episodes=episodes,raw_rollouts=dict(path=str(raw_path),sha256=st.old.sha(raw_path)),policy_loss='actual clipped PPO surrogate at frozen behavior policy ratio=1, globally normalized GAE, divided by global active rows; entropy reported separately; no clipping/Adam on measured gradients',decision='INSUFFICIENT_PHASE_EVIDENCE' if not adequate else 'NO_CAPTURE_DOMINANCE_RECOMMEND_CANCEL_BALANCED' if alpha==1 else 'STATIC_CAPTURE_SCALE_DECLARED')
    st.p.write_json(OUT/'reward_balance.json',report)
    import runpy
    runpy.run_path(str(ROOT/'tools/summarize_reward_balance_20260915.py'))
if __name__=='__main__':main()
