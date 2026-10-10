"""Arm-aware, CPU-only screen/selection/final evaluator with paired seeds."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch

from .runner import prepare,read
from terl_mappo.batch01.checkpoints import load_bound
from terl_mappo.batch01.evaluation import isolated_rng,seed_manifest
from terl_mappo.native import pack_local
from terl_mappo.evaluate import summarize
from terl_mappo.run import atomic_json,file_hash
from cocap_voradj.training.small_step_ac import tensor_tree

@torch.no_grad()
def episode(runtime,seed,mode,offset,horizon_cap=None,scene_stage=None):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed+(offset if mode=='sample' else 0))
    if scene_stage is None:
        runtime.adapter=runtime.source_guard.call(runtime.hooks.env_factory,seed)
    else:
        from .t1 import NativeCurriculum
        runtime.adapter=runtime.source_guard.call(NativeCurriculum,seed,scene_stage)
    bridge=runtime.hooks.reward_transform
    if bridge is not None and hasattr(bridge,'bind'):runtime.source_guard.call(bridge.bind,runtime.adapter)
    initial=runtime.adapter.fingerprint();visited2=visited3=strict=False;entropy=[];components={};collisions=[];reward_sum=0.
    count=0
    with runtime.source_guard.scope():
        while True:
            obs=tensor_tree(pack_local(runtime.adapter.observations),'cpu')
            actor=runtime.trainer.actor
            actions,_,latent=actor.sample(obs,deterministic=mode in {'argmax','mean'})
            if runtime.hooks.backend.categorical:ent=actor.distribution(obs).entropy()
            else:
                # MC entropy diagnostics cannot consume the paired action stream.
                with isolated_rng():ent=actor.evaluate_latent(obs,latent)[1]
            entropy.append(float(ent.mean()))
            snap=getattr(bridge,'snapshot',None);pre=runtime.source_guard.call(snap,runtime.adapter.env) if callable(snap) else None
            _,raw,_,_,end,info=runtime.source_guard.call(runtime.adapter.step,actions.cpu().numpy())
            shaped=np.asarray(raw)
            if bridge is not None:
                if pre is not None:info.update(bridge_pre=pre,bridge_post=runtime.source_guard.call(snap,runtime.adapter.env))
                shaped=np.asarray(runtime.source_guard.call(bridge,np.asarray(raw).copy(),info))
            reward_sum+=float(np.mean(shaped));count+=1
            visited2|=info['ring2'];visited3|=info['ring3'];strict|=info['strict_geometry'];collisions+=info['collision_types']
            for k,v in info['components'].items():components[k]=components.get(k,0)+float(np.mean(v))
            if bridge is not None and hasattr(bridge,'last_components'):
                for k in ('raw','native_distance','shaping','shaped'):
                    components['bridge_'+k]=components.get('bridge_'+k,0)+float(np.mean(bridge.last_components[k]))
            if end or horizon_cap is not None and count>=horizon_cap:break
    return {'seed':seed,'sample_action_seed':seed+offset if mode=='sample' else None,'mode':mode,'initial_fingerprint':initial,
            'steps':info['episode_steps'],'capture':info['capture'],'normal_capture':info['normal_capture'],
            'collision':info['collision'],'collision_types':sorted(set(collisions)),
            'ring2':visited2,'ring3':visited3,'strict_geometry':strict,
            'capture_time':info['episode_steps']*.5 if info['capture'] else None,'censored':not info['capture'],
            'reward_components':components,'episode_reward':reward_sum,
            'native_episode_reward':float(np.mean(info['episode_return'])),'action_entropy':float(np.mean(entropy))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--domain',choices=['screen','selection','final','smoke'],required=True);p.add_argument('--horizon-cap',type=int);p.add_argument('--scene-stage',type=int,choices=[1,2,3])
    a=p.parse_args();torch.set_num_threads(1);manifest=read(a.manifest);out=Path(a.output)
    if out.exists():raise ValueError('refuse evaluation overwrite')
    if a.horizon_cap and a.domain!='smoke':raise ValueError('formal evaluations must run full episodes')
    if a.domain=='final':
        selection=out.parent/'selected.json'
        if not selection.exists() or read(selection).get('checkpoint_sha256')!=file_hash(a.checkpoint):
            raise ValueError('final requires frozen selection receipt')
        if manifest['execution_mode'] in {'PROVISIONAL','PROVISIONAL_LONG'}:raise ValueError('pilot cannot consume final holdout')
    # Keep full evaluations at4 slots; bounded8-step engineering smokes have2 separate CPU slots.
    lease_name='smoke_evaluator_leases' if a.domain=='smoke' else 'evaluator_leases'
    lockdir=Path('/home/yjq/rl/CoCap1/batch01-commander-runtime')/lease_name;lockdir.mkdir(parents=True,exist_ok=True)
    lease=None
    while lease is None:
        for i in range(2 if a.domain=='smoke' else 4):
            fd=(lockdir/f'{i}.lock').open('w')
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);lease=fd;break
            except BlockingIOError:fd.close()
        if lease is None:time.sleep(.5)
    runtime,lock,delta,resolved,_,_=prepare(Path(a.manifest).parent/'delta.json','cpu',Path(a.manifest).parent/'source_failures')
    ck=load_bound(a.checkpoint,runtime,manifest);protocol=manifest['evaluation_protocol'];seed_manifest(resolved,protocol)
    domain='screen' if a.domain=='smoke' else a.domain;setting=protocol[domain]
    n=1 if a.domain=='smoke' else setting['max_episodes_per_mode']
    if domain=='screen' and ck['steps'] not in protocol['screen_milestones'] and a.domain!='smoke':n=setting['regular_episodes_per_mode']
    seeds=[setting['seed_base']+i for i in range(n)]
    # This arm's scene is explicitly bound; all other scenes use their locked
    # offsets and are evaluated separately, never pooled with Stage1.
    if a.scene_stage is not None:
        if delta['line']!='T1':raise ValueError('scene override only for T1 scale evaluation')
        from .t1 import NativeCurriculum
        with isolated_rng():runtime.adapter=runtime.source_guard.call(NativeCurriculum,resolved['seed'],a.scene_stage)
    scene=runtime.adapter.contract()['counts'];label=f'{scene[0]}P{scene[1]}E{scene[2]}O{scene[3]}C'
    if label not in protocol['scene_offsets']:raise ValueError('scene missing from locked paired seed protocol')
    scene_offset=protocol['scene_offsets'][label];seeds=[s+scene_offset for s in seeds]
    digest=file_hash(a.checkpoint);rows=[];start=time.monotonic()
    modes=('argmax','sample') if runtime.hooks.backend.categorical else ('mean','sample')
    with isolated_rng():
        for mode in modes:
            for seed in seeds:
                rows.append(episode(runtime,seed,mode,protocol['sample_torch_offset'],a.horizon_cap,a.scene_stage))
                atomic_json(out.with_suffix('.partial.json'),{'completed':len(rows),'total':2*n,'rows':rows,'pid':os.getpid()})
    if file_hash(a.checkpoint)!=digest:raise ValueError('checkpoint changed during evaluation')
    result={'checkpoint':a.checkpoint,'checkpoint_sha256':digest,'steps':ck['steps'],'seed_domain':a.domain,
            'seed_base':seeds[0],'episodes_per_mode':n,'duration_seconds':time.monotonic()-start,
            'scene':label,'evaluation_protocol':protocol,'execution_mode':manifest['execution_mode'],
            'status':'SMOKE_NOT_FORMAL_EVIDENCE' if a.domain=='smoke' else 'PROVISIONAL_SCREEN' if manifest['execution_mode'] in {'PROVISIONAL','PROVISIONAL_LONG'} else 'COMPLETE',
            'modes':{mode:summarize([r for r in rows if r['mode']==mode]) for mode in modes},'action_backend':manifest['active_runtime_values']['action_backend'],'episodes':rows}
    atomic_json(out,result);out.with_suffix('.partial.json').unlink(missing_ok=True);print(json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)

if __name__=='__main__':main()
