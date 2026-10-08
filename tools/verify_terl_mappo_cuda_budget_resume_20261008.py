#!/usr/bin/env python3
"""Verify full CUDA checkpoint replay while changing only the total budget."""
import argparse
import copy
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from terl_mappo.native import NativeStage1
from terl_mappo.run import build, collect, load_checkpoint, file_hash, rng_state, atomic_json

def cpu_copy(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_copy(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)

def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b): raise AssertionError('tensor mismatch: '+path)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b,err_msg=path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys(): raise AssertionError('keys mismatch: '+path)
        for k in a: equal(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b): raise AssertionError('length mismatch: '+path)
        for i,(x,y) in enumerate(zip(a,b)): equal(x,y,path+'.'+str(i))
    elif a!=b: raise AssertionError('value mismatch: '+path)

def replay(checkpoint,config,length,device):
    trainer=build(config,device);adapter=NativeStage1(config['seed'])
    loaded=load_checkpoint(checkpoint,trainer,adapter,config,device)
    before=cpu_copy(trainer.state_dict())
    batch,episodes=collect(trainer,adapter,length)
    telemetry=trainer.update(batch,categorical=True)
    if not all(np.isfinite(v) for v in telemetry.values()): raise AssertionError('nonfinite update')
    state=cpu_copy(trainer.state_dict())
    env=pickle.dumps(adapter.state_dict(),protocol=5)
    rng=cpu_copy(rng_state())
    return {'loaded_steps':loaded['steps'],'before':before,'batch':batch,'episodes':episodes,
            'telemetry':telemetry,'trainer_after':state,'environment_after':env,'rng_after':rng}

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--length',type=int,default=32)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(1);start=time.monotonic();config=json.loads(Path(a.config).read_text())
    extended=copy.deepcopy(config);extended['budget']=1000000
    digest=file_hash(a.checkpoint)
    original=replay(a.checkpoint,config,a.length,a.device)
    resumed=replay(a.checkpoint,extended,a.length,a.device)
    equal(original,resumed)
    if digest!=file_hash(a.checkpoint): raise AssertionError('immutable checkpoint changed')
    result={'passed':True,'reference_checkpoint':str(Path(a.checkpoint).resolve()),'sha256':digest,
        'reference_steps':original['loaded_steps'],'device':a.device,'torch_version':torch.__version__,
        'config_only_difference':{'budget':[config['budget'],extended['budget']]},
        'rollout_decisions_compared':a.length,'paired_minibatch_updates':original['telemetry']['minibatch_updates'],
        'bit_exact':['loaded actor/critic/Adam/ValueNorm/counters','all rollout fields and episodes',
                     'updated actor/critic/Adam/ValueNorm/counters','PPO telemetry','full serialized environment','Python/NumPy/CPU/CUDA RNG'],
        'sources_strictly_verified':True,'duration_seconds':time.monotonic()-start}
    atomic_json(a.output,result);print(json.dumps(result),flush=True)

if __name__=='__main__': main()
