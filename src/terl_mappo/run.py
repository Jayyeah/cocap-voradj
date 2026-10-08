"""Bounded native Stage1 runner with atomic resumable checkpoints."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .native import ROOT, VENDOR, TERL_SHA, NativeStage1, central_state, pack_local
from .model import TERLActor, make_critic
from cocap_voradj.training.small_step_ac import MAPPOTrainer, MAPPOConfig, tensor_tree, flatten_local

DEFAULT = ROOT/'configs/experiments/terl_mappo_20261008/stage1.json'

def atomic_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def source_hash():
    files=list((ROOT/'src/terl_mappo').glob('*.py'))
    files += [ROOT/'src/cocap_voradj/training/small_step_ac.py', ROOT/'src/cocap_voradj/models/small_step_ac.py']
    files += list(VENDOR.rglob('*.py'))+[VENDOR/'config/config.yaml']
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(files)}

def build(config, device):
    random.seed(config['actor_seed']); np.random.seed(config['actor_seed']); torch.manual_seed(config['actor_seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config['actor_seed'])
    actor=TERLActor(config['hidden_dim'],config['num_heads'],config['num_layers'],config['actor_seed'],config['masked_pool'])
    value=make_critic(config['hidden_dim'],config['num_heads'],config['num_layers'])
    trainer=MAPPOTrainer(actor,value,MAPPOConfig(**config['ppo']),device)
    trainer.value.eval()
    return trainer

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state, device):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if str(device).startswith('cuda') and state['cuda']: torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])

def save_checkpoint(path,trainer,adapter,config,steps,agent_steps,optimizer_steps):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data={'version':1,'terl_sha':TERL_SHA,'config':config,'trainer':trainer.state_dict(),
          'runtime':adapter.state_dict(),'rng':rng_state(),'steps':steps,'agent_transitions':agent_steps,
          'optimizer_steps':optimizer_steps,'sources':source_hash()}
    temporary=path.with_suffix('.pt.tmp'); torch.save(data,temporary); temporary.replace(path)
    return file_hash(path)

def load_checkpoint(path,trainer,adapter,config,device,strict=True):
    data=torch.load(path,map_location=device,weights_only=False)
    if data['terl_sha']!=TERL_SHA: raise ValueError('wrong TERL SHA')
    if strict:
        a={k:v for k,v in data['config'].items() if k!='budget'}
        b={k:v for k,v in config.items() if k!='budget'}
        if a!=b or data['sources']!=source_hash(): raise ValueError('resume contract/source mismatch')
    trainer.load_state_dict(data['trainer']); adapter.load_state_dict(data['runtime']); restore_rng(data['rng'],device)
    return data

def collect(trainer,adapter,length):
    rows=[]; episodes=[]
    for _ in range(length):
        local=pack_local(adapter.observations); central=central_state(adapter.env)
        indices,logp,latent,value=trainer.act(local,{k:v[None] for k,v in central.items()})
        _,reward,terminal,trunc,end,info=adapter.step(indices)
        next_state=central_state(adapter.env)
        with torch.no_grad():
            next_value=trainer.value(tensor_tree({k:v[None] for k,v in next_state.items()},trainer.device)).cpu().numpy()[0]
        rows.append({'local_obs':local,'global_obs':central,'actions':indices,'latent':latent,
                     'log_prob':logp,'values':value[0],'next_values':next_value,'rewards':reward,
                     'terminated':terminal,'truncated':trunc,'episode_end':np.full(3,end),
                     'active_mask':info['active']})
        if end:
            episodes.append({k:v for k,v in info.items() if k not in ('active','native_infos')})
            adapter.reset()
    result={}
    for k in rows[0]:
        result[k]={j:np.stack([r[k][j] for r in rows]) for j in rows[0][k]} if isinstance(rows[0][k],dict) else np.stack([r[k] for r in rows])
    return result,episodes

@torch.no_grad()
def post_update_policy_stats(trainer, batch):
    local,_=flatten_local(tensor_tree(batch['local_obs'],trainer.device))
    active=torch.as_tensor(batch['active_mask'],device=trainer.device).bool().reshape(-1)
    index=torch.nonzero(active,as_tuple=False).squeeze(-1)
    actions=torch.as_tensor(batch['latent'],device=trainer.device).long().reshape(-1)
    old=torch.as_tensor(batch['log_prob'],device=trainer.device).reshape(-1)
    log_ratios=[]
    for ids in index.split(256):
        new,_=trainer.actor.evaluate_indices({k:v[ids] for k,v in local.items()},actions[ids])
        log_ratios.append(new-old[ids])
    log_ratio=torch.cat(log_ratios); ratio=log_ratio.exp()
    return {'post_update_ratio_mean':float(ratio.mean()),'post_update_ratio_min':float(ratio.min()),
            'post_update_ratio_max':float(ratio.max()),
            'post_update_clip_fraction':float(((ratio-1).abs()>trainer.config.clip_param).float().mean()),
            'post_update_kl':float(((ratio-1)-log_ratio).mean())}

def train(args):
    torch.set_num_threads(args.threads)
    config=json.loads(Path(args.config).read_text())
    if config['terl_sha']!=TERL_SHA: raise ValueError('config TERL reference mismatch')
    if args.budget is not None: config['budget']=args.budget
    if not 0<config['budget']<=config['stage1_ceiling']: raise ValueError('Stage1 budget outside authorized ceiling')
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists() and not args.resume: raise ValueError('existing run requires explicit resume')
    trainer=build(config,args.device); adapter=NativeStage1(config['seed'])
    step=agent_steps=optimizer_steps=0
    if args.resume:
        data=load_checkpoint(args.resume,trainer,adapter,config,args.device)
        step=data['steps']; agent_steps=data['agent_transitions']; optimizer_steps=data['optimizer_steps']
    else:
        atomic_json(out/'manifest.json',{'config':config,'terl_sha':TERL_SHA,'sources':source_hash(),
                    'initialization':'random; no IQN/BC/teacher','device':args.device,
                    'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'pid':os.getpid(),
                    'started_at':datetime.now().astimezone().isoformat(),
                    'counter_units':{'steps':'joint environment decisions','agent_transitions':'active actor rows',
                                     'updates':'rollout PPO calls','optimizer_steps':'paired actor/critic minibatches'},
                    'initial_fingerprint':adapter.fingerprint(),'runtime_contract':adapter.contract(),
                    'actor_parameters':sum(p.numel() for p in trainer.actor.parameters()),
                    'critic_parameters':sum(p.numel() for p in trainer.value.parameters())})
        save_checkpoint(out/'checkpoints/step_000000000.pt',trainer,adapter,config,0,0,0)
    start=time.monotonic(); initial_step=step
    try:
        while step<config['budget']:
            boundary=((step//config['checkpoint_interval'])+1)*config['checkpoint_interval']
            count=min(config['rollout_length'],config['budget']-step,boundary-step)
            batch,episodes=collect(trainer,adapter,count)
            metrics=trainer.update(batch,categorical=True)
            metrics.update(post_update_policy_stats(trainer,batch))
            if not all(np.isfinite(x) for x in metrics.values()): raise FloatingPointError('nonfinite PPO telemetry')
            if not all(torch.isfinite(p).all() for p in list(trainer.actor.parameters())+list(trainer.value.parameters())):
                raise FloatingPointError('nonfinite model parameters')
            step+=count; agent_steps+=int(batch['active_mask'].sum()); optimizer_steps+=int(metrics['minibatch_updates'])
            row={'steps':step,'agent_transitions':agent_steps,'optimizer_steps':optimizer_steps,
                 'reward_mean':float(batch['rewards'][batch['active_mask']].mean()),
                 'reward_min':float(batch['rewards'][batch['active_mask']].min()),
                 'reward_max':float(batch['rewards'][batch['active_mask']].max()),**metrics,
                 'episodes':len(episodes),'training_capture':sum(e['capture'] for e in episodes),
                 'training_collision':sum(e['collision'] for e in episodes)}
            with open(out/'metrics.jsonl','a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
            with open(out/'episodes.jsonl','a') as f:
                for e in episodes: f.write(json.dumps({'recorded_at_step':step,**e})+'\n')
            speed=(step-initial_step)/(time.monotonic()-start)
            status={'status':'RUNNING' if step<config['budget'] else 'COMPLETE_BUDGET','pid':os.getpid(),
                    'steps':step,'budget':config['budget'],'agent_transitions':agent_steps,
                    'updates':trainer.update_count,'optimizer_steps':optimizer_steps,'finite':True,
                    'steps_per_second':speed,'eta_seconds':(config['budget']-step)/max(speed,1e-9),
                    'updated_at':datetime.now().astimezone().isoformat(),'last_metrics':row}
            if args.device.startswith('cuda'): status['peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
            atomic_json(out/'progress.json',status)
            print(json.dumps({k:v for k,v in status.items() if k!='last_metrics'}),flush=True)
            # Latest every ~2k, all fixed candidates retained (<6 checkpoints at 100k).
            if step==config['budget'] or step%config['checkpoint_interval']==0 or step%2048==0:
                digest=save_checkpoint(out/'checkpoints/latest.pt',trainer,adapter,config,step,agent_steps,optimizer_steps)
                atomic_json(out/'latest_checkpoint.json',{'steps':step,'sha256':digest})
            if step==config['budget'] or step%config['checkpoint_interval']==0:
                path=out/f'checkpoints/step_{step:09d}.pt'
                digest=save_checkpoint(path,trainer,adapter,config,step,agent_steps,optimizer_steps)
                atomic_json(path.with_suffix('.json'),{'steps':step,'sha256':digest})
        return status
    except BaseException as exc:
        atomic_json(out/'failure.json',{'steps':step,'error':repr(exc),'pid':os.getpid()})
        raise

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=str(DEFAULT)); p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu'); p.add_argument('--budget',type=int)
    p.add_argument('--threads',type=int,default=1); p.add_argument('--resume')
    train(p.parse_args())

if __name__=='__main__': main()
