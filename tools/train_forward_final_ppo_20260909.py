#!/usr/bin/env python3
"""D: Final mission reset/schedule with P0 categorical MAPPO, finite budget."""
from __future__ import annotations
import argparse,copy,dataclasses,hashlib,json,os,random,sys,time
from collections import deque,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from cocap_voradj.training.forward_final import CONTRACT,TEACHER_SHA,preflight,make_env,check_env,scene_config
from cocap_voradj.training.trainer import CoCapTrainer,set_global_config,parse_scalar_step_schedule
from cocap_voradj.training.small_step_ac import MAPPOTrainer,MAPPOConfig,tensor_tree
from cocap_voradj.models.small_step_ac import CentralValueNetwork
from cocap_voradj.training.continuous.central_schema import build_central_global_obs
from cocap_voradj.training.runtime_semantics import assert_runtime,initial_state_fingerprint
from tools.distill_forward_final_actor_20260908 import load_actor
from tools.run_forward_final_bridge_20260908 import atomic_json
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file
from tools.run_continuous_ctde_training import _split_termination_flags
from tools.run_small_step_ac_migration import empty_rollout,stack_rollout
BC=ROOT/'artifacts/2026-09-08_forward_final/c2_distillation/actor_epoch_030.pt'
BC_SHA='7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd'
SCHEMA='forward-final-ppo-v1'
POLICY_CONTRACT='categorical-eval-forward-zero-ratio-v1'


def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FinalMissionStream:
    """Reuse the actual Final trainer reset/schedule methods without its IQN learner."""
    def __init__(self,seed,run_dir):
        self.config=scene_config('mixed');self.train_mode='voradj_mixed_coverage';self.run_dir=Path(run_dir);self.run_dir.mkdir(parents=True,exist_ok=True)
        self.envs={task:make_env(scene,seed+delta)[0] for task,scene,delta in [('voradj','mixed',0),('voradj_coverage','coverage',100000)]}
        recovery=self.config['voradj']['recovery']
        self.recovery_init_pool=deque(maxlen=int(recovery['capture_state_pool_capacity']))
        self.recovery_from_capture_ratio=float(recovery['captured_state_ratio'])
        self.recovery_map_random_ratio_within_non_capture=float(recovery['map_random_ratio_within_non_capture'])
        assert (self.recovery_init_pool.maxlen,self.recovery_from_capture_ratio,self.recovery_map_random_ratio_within_non_capture)==(1000,.75,.5)
        self.apf_agents={};self.last_reset_source={};self.reset_counts=Counter();self.episode=-1;self.task=None;self.observations=None
        reward=self.config['reward'];self.base_coverage_ce_speed_weight=float(reward['coverage_ce_speed_weight'])
        self.coverage_ce_speed_weight_schedule=parse_scalar_step_schedule(reward['coverage_ce_speed_weight_schedule'],default_value=self.base_coverage_ce_speed_weight,field_name='coverage_ce_speed_weight')
        self.global_step=2000000;self.set_clock(0);self.reset()

    @property
    def env(self):return self.envs[self.task]

    def set_clock(self,online_step):
        # Continue the teacher's Final reward age, not the old AC schedule/reset default.
        self.global_step=2000000+int(online_step)
        value=CoCapTrainer._set_coverage_ce_control_weights(self)
        assert value==.0005
        for env in self.envs.values():assert env.reward_cfg['coverage_ce_speed_weight']==value

    def reset(self):
        self.episode+=1;self.task='voradj' if self.episode%2==0 else 'voradj_coverage'
        self.observations=CoCapTrainer._reset_task(self,self.task)
        self.reset_counts[self.last_reset_source[self.task]]+=1
        check_env(self.env)
        return self.observations

    def step(self,indices):
        set_global_config(self.env.config)
        outcome=self.env.step([int(a) if o is not None else None for a,o in zip(indices,self.observations)],CoCapTrainer._evader_actions(self,self.task))
        self.observations=outcome.observations
        return outcome

    def finish(self):
        record=self.env.episode_record(task='mix' if self.task=='voradj' else 'coverage')
        snap=record.get('capture_snapshot')
        if self.task=='voradj' and isinstance(snap,dict):
            self.recovery_init_pool.append({k:copy.deepcopy(snap.get(k,0 if k=='step' else [])) for k in ('step','positions','active_mask')})
        row={'episode':self.episode,'task':self.task,'reset_source':self.last_reset_source[self.task],'captured':bool(record['captured']),'ce_success':bool(self.env.post_capture_coverage_success),'collision':bool(record.get('collision_event',False)),'pool_size':len(self.recovery_init_pool)}
        self.reset();return row


def local_and_global(stream):
    live=next(o for o in stream.observations if o is not None)
    local={k:np.stack([o[k] if o is not None else np.zeros_like(v) for o in stream.observations]) for k,v in live.items()}
    central=build_central_global_obs(stream.env,max_agents=4,max_evaders=8,max_obstacles=5,self_feature_dim=9)
    return local,central


def make_trainer(device,seed):
    assert sha256_file(BC)==BC_SHA
    actor,_=load_actor(BC,device)
    for p in actor.parameters():p.requires_grad_(True)
    torch.manual_seed(seed)  # Exactly the same initial V in Direct and Warm-up.
    value=CentralValueNetwork(hidden_dim=256,num_heads=8,num_layers=4,self_feature_dim=9,max_agents=4,max_evaders=8,max_obstacles=5)
    config=MAPPOConfig(ppo_epochs=3,minibatches=2,actor_lr=3e-5,critic_lr=1e-4,target_kl=.02,value_norm=True)
    return MAPPOTrainer(actor,value,config,device)


def collect_transition(trainer,stream):
    local,central=local_and_global(stream);active=central['active_mask'].copy()
    physical,logp,indices,values=trainer.act(local,{k:v[None] for k,v in central.items()})
    np.testing.assert_allclose(physical,trainer.actor.action_grid.detach().cpu().numpy()[indices],atol=1e-6)
    outcome=stream.step(indices);term,trunc=_split_termination_flags(outcome.dones,outcome.infos)
    # Obtain the last physical state BEFORE any reset, including timeout bootstrap.
    next_central=build_central_global_obs(stream.env,max_agents=4,max_evaders=8,max_obstacles=5,self_feature_dim=9)
    with torch.no_grad():next_values=trainer.value(tensor_tree({k:v[None] for k,v in next_central.items()},trainer.device))[0].cpu().numpy()
    row={'local_obs':local,'global_obs':central,'actions':physical,'latent':indices,'log_prob':logp,'values':values[0],'next_values':next_values,'rewards':np.asarray(outcome.rewards,np.float32),'active_mask':active,'terminated':term,'truncated':trunc,'episode_end':term|trunc}
    episode=stream.finish() if all(outcome.dones) else None
    return row,episode


def save_checkpoint(out,step,trainer,stream,launch,rollout,last_metrics):
    name=out/f'step_{step:06d}.pt'
    if name.exists():raise ValueError('Never overwrite a valid checkpoint')
    payload={'schema':SCHEMA,'policy_contract':POLICY_CONTRACT,'contract':CONTRACT,'step':step,'launch':launch,'trainer':trainer.state_dict(),'stream_state':stream.__dict__,'rollout':rollout,'last_metrics':last_metrics,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(trainer.device) if trainer.device.type=='cuda' else None}
    temp=name.with_suffix('.tmp');torch.save(payload,temp)
    verified=torch.load(temp,map_location='cpu',weights_only=False)
    assert verified['step']==step and verified['policy_contract']==POLICY_CONTRACT
    os.replace(temp,name)
    actor_path=out/f'actor_step_{step:06d}.pt'
    torch.save({'schema':SCHEMA,'policy_contract':POLICY_CONTRACT,'contract':CONTRACT,'teacher_sha256':TEACHER_SHA,'bc_parent_sha256':BC_SHA,'step':step,'actor_state_dict':{k:v.detach().cpu() for k,v in trainer.actor.state_dict().items()}},actor_path)
    return {'checkpoint':str(name),'checkpoint_sha256':sha256_file(name),'actor_checkpoint':str(actor_path),'actor_sha256':sha256_file(actor_path)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);p.add_argument('--branch',choices=('direct','warmup'),required=True)
    p.add_argument('--steps',type=int,default=25000);p.add_argument('--warmup-steps',type=int,default=4096);p.add_argument('--rollout-length',type=int,default=256)
    p.add_argument('--seed',type=int,default=2026097101);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');args=p.parse_args();out=args.output_root
    assert 0<args.steps<=25000,'25k first Gate; do not silently extend to 50k'
    assert args.rollout_length>0 and args.warmup_steps>=0
    if not args.smoke:assert args.steps==25000 and args.rollout_length==256 and args.warmup_steps==4096
    if (out/'launch.json').exists():raise ValueError('Fresh output required; no silent checkpoint overwrite/resume')
    c3=json.loads((ROOT/'artifacts/2026-09-08_forward_final/c3_formal100/comparison.json').read_text())
    assert c3['decision'] in ('PASS','PASS_WITH_STOCHASTIC_EFFICIENCY_GAP') and c3['actor_sha256']==BC_SHA and c3['initial_state_fingerprints_verified']
    atomic_json(out/'frozen_contract_preflight.json',preflight())
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    trainer=make_trainer(args.device,args.seed);stream=FinalMissionStream(args.seed,out)
    parent_hash=tensor_hash(trainer.actor.state_dict());critic_hash=tensor_hash(trainer.value.state_dict());warm=args.warmup_steps if args.branch=='warmup' else 0
    launch={'schema':SCHEMA,'policy_contract':POLICY_CONTRACT,'contract':CONTRACT,'bc_parent_sha256':BC_SHA,'c3_comparison_sha256':sha256_file(ROOT/'artifacts/2026-09-08_forward_final/c3_formal100/comparison.json'),'actor_initial_tensor_sha256':parent_hash,'critic_initial_tensor_sha256':critic_hash,'initial_fingerprint':initial_state_fingerprint(stream.env),'branch':args.branch,'warmup_steps':warm,'total_env_steps':args.steps,'budget':'warm-up counts within the common 25k total interaction budget; D2 has 4096 fewer actor-update interaction steps','seed':args.seed,'rollout_length':args.rollout_length,'ppo_config':dataclasses.asdict(trainer.config),'actor_backbone':'all parameters trainable for PPO; entire actor unchanged during warm-up','reward_schedule_clock':'teacher Stage1 age 2m + online steps; effective CE speed .0005, same in both branches','reset':'native Final trainer reset; alternating mixed/coverage; own online capture pool initially empty, cap1000, capture .75, map-random among noncapture .5','runtime':{task:check_env(env) for task,env in stream.envs.items()},'zero_update_probe':assert_runtime(stream.env,trainer.actor,categorical=True),'smoke':args.smoke,'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES')}
    atomic_json(out/'launch.json',launch)
    rollout=empty_rollout();start=time.monotonic();updates=[];checkpoint={};episode_counts=Counter()
    for step in range(1,args.steps+1):
        stream.set_clock(step-1);row,episode=collect_transition(trainer,stream)
        for k,v in row.items():rollout[k].append(v)
        if episode:
            episode_counts[episode['task']]+=1
            with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps({'env_step':step,**episode})+'\n')
        boundary=len(rollout['rewards'])>=args.rollout_length or step==args.steps or step==warm
        if boundary:
            batch=stack_rollout(rollout);error=trainer.assert_behavior_log_probs(batch)
            if step<=warm:
                metrics=trainer.update_critic_only(batch)
                assert tensor_hash(trainer.actor.state_dict())==parent_hash,'Warm-up changed the actor'
            else:metrics=trainer.update(batch,categorical=True)
            assert all(np.isfinite(v) for v in metrics.values()),'Nonfinite PPO metrics'
            metrics.update({'step':step,'zero_update_max_log_prob_error':error,'critic_only':int(step<=warm),'ppo_env_steps':max(step-warm,0),'ce_speed_weight':stream.current_coverage_ce_speed_weight})
            updates.append(metrics);rollout=empty_rollout();atomic_json(out/'updates.json',updates)
            if step==warm:atomic_json(out/'warmup_gate.json',{'actor_bit_exact':True,'steps':step,'critic_explained_variance':metrics['explained_variance'],'interpretation':'fixed-budget warm-up finished, not proof of an accurate critic'})
        if step%25000==0 or step==args.steps:checkpoint=save_checkpoint(out,step,trainer,stream,launch,rollout,updates[-1])
        if step%100==0 or step==args.steps:
            elapsed=time.monotonic()-start
            atomic_json(out/'progress.json',{'status':'complete' if step==args.steps else 'running','step':step,'total_steps':args.steps,'steps_per_second':step/elapsed,'elapsed_seconds':elapsed,'eta_seconds':elapsed/step*(args.steps-step),'last_update':updates[-1] if updates else None,'episodes':dict(episode_counts),'reset_counts':dict(stream.reset_counts),'pool_size':len(stream.recovery_init_pool),**checkpoint})
    atomic_json(out/'report.json',{'status':'complete','launch':launch,'last_update':updates[-1],'checkpoints':checkpoint,'next_gate':'Frozen sample eval against the same BC parent required before 50k; no automatic continuation','smoke':args.smoke})

if __name__=='__main__':main()
