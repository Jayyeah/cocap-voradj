"""Independent CPU evaluator with parallel episodes and isolated seed domains."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from .native import NativeStage1, pack_local
from .model import TERLActor
from .run import atomic_json, file_hash
from cocap_voradj.training.small_step_ac import tensor_tree

_WORKER = {}

def worker_init(checkpoint):
    torch.set_num_threads(1)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False); c=ck['config']
    actor=TERLActor(c['hidden_dim'],c['num_heads'],c['num_layers'],c['actor_seed'],c['masked_pool'])
    actor.load_state_dict(ck['trainer']['actor']); actor.eval()
    _WORKER.update(actor=actor,steps=ck['steps'])

@torch.no_grad()
def episode(task):
    seed,mode=task
    # Same physical initial conditions across modes; separate sampling stream.
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed+ (100000 if mode=='sample' else 0))
    a=NativeStage1(seed); initial=a.fingerprint(); actor=_WORKER['actor']
    visited2=visited3=strict=False; entropy=[]; components={}; collision_types=[]
    while True:
        obs=tensor_tree(pack_local(a.observations),'cpu')
        d=actor.distribution(obs)
        action=d.logits.argmax(-1) if mode=='argmax' else d.sample()
        entropy.append(float(d.entropy().mean()))
        _,_,_,_,end,info=a.step(action.numpy())
        visited2 |= info['ring2']; visited3 |= info['ring3']; strict |= info['strict_geometry']
        collision_types += info['collision_types']
        for k,v in info['components'].items(): components[k]=components.get(k,0)+float(np.mean(v))
        if end: break
    return {'seed':seed,'mode':mode,'initial_fingerprint':initial,'steps':info['episode_steps'],
            'capture':info['capture'],'normal_capture':info['normal_capture'],'collision':info['collision'],
            'collision_types':sorted(set(collision_types)),'ring2':visited2,'ring3':visited3,
            'strict_geometry':strict,'capture_time':info['episode_steps']*.5 if info['capture'] else None,
            'censored':not info['capture'],'reward_components':components,
            'episode_reward':float(np.mean(info['episode_return'])),'action_entropy':float(np.mean(entropy))}

def summarize(rows):
    n=len(rows); times=[r['capture_time'] for r in rows if r['capture']]
    result={k+'_count':sum(bool(r[k]) for r in rows) for k in ('capture','normal_capture','collision','ring2','ring3','strict_geometry','censored')}
    result.update({k+'_rate':result[k+'_count']/n for k in ('capture','normal_capture','collision','ring2','ring3','strict_geometry','censored')})
    result.update(episodes=n,success_n=len(times),censored_n=n-len(times),
                  capture_time_mean=float(np.mean(times)) if times else None,
                  capture_time_median=float(np.median(times)) if times else None,
                  capture_time_p90=float(np.quantile(times,.9)) if times else None,
                  action_entropy=float(np.mean([r['action_entropy'] for r in rows])),
                  episode_reward_mean=float(np.mean([r['episode_reward'] for r in rows])),
                  collision_types={k:sum(k in r['collision_types'] for r in rows) for k in sorted(set(k for r in rows for k in r['collision_types']))},
                  reward_components={k:float(np.mean([r['reward_components'][k] for r in rows])) for k in rows[0]['reward_components']})
    return result

def evaluate(checkpoint, output, episodes=20, seed_base=2026100800, workers=2, domain='screen'):
    start=time.monotonic(); output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise ValueError('evaluation output already exists')
    digest=file_hash(checkpoint)
    tasks=[(seed_base+i,mode) for mode in ('argmax','sample') for i in range(episodes)]
    rows=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=worker_init,initargs=(str(checkpoint),)) as pool:
        for row in pool.map(episode,tasks,chunksize=1):
            rows.append(row)
            atomic_json(output.with_suffix('.partial.json'),{'completed':len(rows),'total':len(tasks),'rows':rows})
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if file_hash(checkpoint)!=digest: raise ValueError('checkpoint changed during evaluation')
    result={'checkpoint':str(checkpoint),'checkpoint_sha256':digest,'steps':ck['steps'],'seed_domain':domain,
            'seed_base':seed_base,'episodes_per_mode':episodes,'duration_seconds':time.monotonic()-start,
            'modes':{m:summarize([r for r in rows if r['mode']==m]) for m in ('argmax','sample')},'episodes':rows}
    atomic_json(output,result)
    output.with_suffix('.partial.json').unlink(missing_ok=True)
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True)
    p.add_argument('--episodes',type=int,default=20); p.add_argument('--seed-base',type=int,default=2026100800)
    p.add_argument('--workers',type=int,default=2); p.add_argument('--domain',default='screen')
    a=p.parse_args(); evaluate(a.checkpoint,a.output,a.episodes,a.seed_base,a.workers,a.domain)

if __name__=='__main__': main()
