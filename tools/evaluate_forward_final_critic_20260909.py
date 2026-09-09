#!/usr/bin/env python3
"""Heldout frozen-BC full-episode MC calibration of cold versus warm central V."""
import argparse,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from tools.train_forward_final_ppo_20260909 import make_trainer,FinalMissionStream,collect_transition,tensor_hash,BC_SHA,SCHEMA,POLICY_CONTRACT
from cocap_voradj.training.small_step_ac import tensor_tree,compute_gae
from tools.run_forward_final_bridge_20260908 import atomic_json
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file


def full_returns(rewards,terminated,gamma=.99):
    result=np.zeros_like(rewards,dtype=np.float64);future=np.zeros(rewards.shape[1],dtype=np.float64)
    for t in range(len(rewards)-1,-1,-1):
        future=np.asarray(rewards[t],float)+gamma*future*(~np.asarray(terminated[t],bool));result[t]=future
    return result


def calibration(target,prediction):
    error=prediction-target;variance=float(np.var(target))
    return {'target_mean':float(np.mean(target)),'prediction_mean':float(np.mean(prediction)),'bias':float(np.mean(error)),'rmse':float(np.sqrt(np.mean(error**2))),'within_phase_ev':1-float(np.var(error))/variance if variance>1e-10 else None,'target_std':float(np.std(target))}



def summarize_rows(rows):
    if not rows:
        return {'rows': 0, 'episodes': 0}
    targets = np.array([r['return'] for r in rows])
    entry = {'rows': len(rows), 'episodes': len({r['episode'] for r in rows})}
    for name in ('cold', 'warm'):
        entry[name] = calibration(targets, np.array([r[name] for r in rows]))
        entry[name]['raw_gae_positive_fraction'] = float(np.mean([r[name + '_adv'] > 0 for r in rows]))
        entry[name]['squared_error_sum'] = float(np.sum((np.array([r[name] for r in rows]) - targets) ** 2))
    entry['rmse_ratio'] = entry['warm']['rmse'] / max(entry['cold']['rmse'], 1e-10)
    return entry


def outcome_strata(rows, episodes):
    # Diagnostic conditioning on future outcome; never replace the all-outcome Gate.
    safe_ids = {e['episode'] for e in episodes if e['ce_success'] and not e['collision']
                and (e['task'] == 'voradj_coverage' or e['captured'])}
    return {phase: {label: summarize_rows([r for r in rows if r['phase'] == phase
                                           and ((r['episode'] in safe_ids) == safe)])
                    for label, safe in [('safe_completion', True), ('other_outcome', False)]}
            for phase in ('pre_capture', 'post_capture', 'pure_coverage')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--pairs',type=int,default=6);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=2026099101);args=p.parse_args();out=args.output_root
    if (out/'launch.json').exists():raise ValueError('Fresh output required')
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert payload['schema']==SCHEMA and payload['policy_contract']==POLICY_CONTRACT and payload['step']==4096
    assert payload['launch']['warmup_steps']==4096 and payload['launch']['bc_parent_sha256']==BC_SHA
    cold=make_trainer(args.device,payload['launch']['seed']);warm=make_trainer(args.device,payload['launch']['seed']);warm.load_state_dict(payload['trainer'])
    assert tensor_hash(cold.value.state_dict())==payload['launch']['critic_initial_tensor_sha256']
    assert tensor_hash(cold.actor.state_dict())==tensor_hash(warm.actor.state_dict())==payload['launch']['actor_initial_tensor_sha256']
    for trainer in (cold,warm):
        trainer.actor.eval();trainer.value.eval()
        for module in (trainer.actor,trainer.value):
            for parameter in module.parameters():parameter.requires_grad_(False)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    stream=FinalMissionStream(args.seed,out);start=time.monotonic();all_rows=[];episode_reports=[]
    gate={'min_episodes_per_phase':3,'max_warm_to_cold_rmse_ratio':.75,'min_within_phase_ev':0.,'focus':['post_capture','pure_coverage'],'scope':'engineering calibration screen, not a confidence claim; fixed Actor, no PPO update'}
    atomic_json(out/'launch.json',{'checkpoint_sha256':sha256_file(args.checkpoint),'bc_parent_sha256':BC_SHA,'seed':args.seed,'pairs':args.pairs,'gate':gate,'reward_clock':'Final2m -> CE speed .0005','target':'discounted full-episode Monte Carlo returns, gamma.99, no value bootstrap; truncated episodes excluded and counted','actor_bit_exact':True,'cold_initial_critic_verified':True,'evaluator_sha256':sha256_file(Path(__file__)),'trainer_source_sha256':sha256_file(ROOT/'tools/train_forward_final_ppo_20260909.py')})
    for episode in range(args.pairs*2):
        rows=[];phases=[]
        while True:
            phases.append('pure_coverage' if stream.task=='voradj_coverage' else 'pre_capture' if any(not e.deactivated for e in stream.env.evaders) else 'post_capture')
            row,done=collect_transition(warm,stream);rows.append(row)
            if done:break
        truncated=bool(np.stack([r['truncated'] for r in rows]).any())
        episode_reports.append({'episode':episode,'length':len(rows),'truncated_excluded':truncated,**done})
        if not truncated:
            rewards=np.stack([r['rewards'] for r in rows]);term=np.stack([r['terminated'] for r in rows]);active=np.stack([r['active_mask'] for r in rows])
            target=full_returns(rewards,term);cold_values=[]
            with torch.no_grad():
                for offset in range(0,len(rows),64):
                    central={k:np.stack([r['global_obs'][k] for r in rows[offset:offset+64]]) for k in rows[0]['global_obs']}
                    cold_values.append(cold.value_for_gae(tensor_tree(central,cold.device)).cpu().numpy())
                cold_values=np.concatenate(cold_values)
                warm_values=warm._denormalize_values(torch.as_tensor(np.stack([r['values'] for r in rows]),device=warm.device)).cpu().numpy()
            advantages={}
            for name,values in [('cold',cold_values),('warm',warm_values)]:
                next_values=np.concatenate([values[1:],np.zeros_like(values[:1])])
                advantages[name]=compute_gae(torch.as_tensor(rewards),torch.as_tensor(values),torch.as_tensor(next_values),torch.as_tensor(term),torch.as_tensor(active),gamma=.99,gae_lambda=.95)[0].numpy()
            for t,phase in enumerate(phases):
                for i in np.flatnonzero(active[t]):all_rows.append({'phase':phase,'episode':episode,'return':target[t,i],'cold':float(cold_values[t,i]),'warm':float(warm_values[t,i]),'cold_adv':float(advantages['cold'][t,i]),'warm_adv':float(advantages['warm'][t,i])})
        elapsed=time.monotonic()-start
        atomic_json(out/'progress.json',{'status':'running','completed':episode+1,'total':args.pairs*2,'elapsed_seconds':elapsed,'eta_seconds':elapsed/(episode+1)*(args.pairs*2-episode-1)})
    groups={};checks={}
    for phase in ('pre_capture','post_capture','pure_coverage'):
        rows=[r for r in all_rows if r['phase']==phase];assert rows,'Missing phase in calibration set'
        entry=summarize_rows(rows);groups[phase]=entry
        if phase in gate['focus']:
            checks[phase]=entry['episodes']>=3 and entry['rmse_ratio']<=.75 and entry['warm']['within_phase_ev'] is not None and entry['warm']['within_phase_ev']>0
    report={'decision':'CALIBRATION_PASS_NOT_PPO_PASS' if all(checks.values()) else 'HOLD_CRITIC_CALIBRATION','checks':checks,'groups':groups,'episodes':episode_reports,'gate':gate,'rows':len(all_rows),'actor_bit_exact':tensor_hash(cold.actor.state_dict())==tensor_hash(warm.actor.state_dict()),'limits':'Same heldout trajectories for both critics; Monte Carlo targets include stochastic return noise. State counts are correlated within episodes. Whole-episode GAE sign summaries are diagnostics, not PPO rollout256 targets.'}
    report['outcome_strata']=outcome_strata(all_rows,episode_reports)
    report['strata_limit']='Post-hoc outcome conditioning for error attribution only; safe subset is not an unbiased policy-value target or a replacement Gate.'
    np.savez_compressed(out/'calibration_rows.npz',**{key:np.asarray([r[key] for r in all_rows]) for key in all_rows[0]})
    report['rows_file']={'path':'calibration_rows.npz','sha256':sha256_file(out/'calibration_rows.npz'),'rows':len(all_rows)}
    atomic_json(out/'report.json',report);atomic_json(out/'progress.json',{'status':'complete','completed':args.pairs*2,'eta_seconds':0,'decision':report['decision']});print(json.dumps(report,indent=2))

if __name__=='__main__':main()
