"""Only target KL changes after strict 775k full-state restoration."""
import copy
from dataclasses import replace
import hashlib
import pickle

from terl_mappo.batch01.checkpoints import load_anchor
from terl_mappo.batch01.provenance import active_values
from terl_mappo.run import rng_state

ANCHOR='/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/runs/terl_mappo_stage1_seed9_1m_continuation/checkpoints/step_000775000.pt'
ANCHOR_HASH='590d486876d4fca1cd75311c2dcc1255f847a1f756d91eed202e737193b58db4'

def state_digest(runtime):
    def normalize(value):
        import numpy as np
        import torch
        if isinstance(value,torch.Tensor):
            v=value.detach().cpu().contiguous();return ('tensor',str(v.dtype),tuple(v.shape),v.numpy().tobytes())
        if isinstance(value,np.ndarray):return ('numpy',str(value.dtype),value.shape,value.tobytes())
        if isinstance(value,dict):return tuple((k,normalize(v)) for k,v in value.items())
        if isinstance(value,(list,tuple)):return tuple(normalize(v) for v in value)
        return value
    # Environment state is loaded unchanged; its independent physical fingerprint
    # supplements exact trainer and all RNG bytes in the controlled fork receipt.
    return hashlib.sha256(pickle.dumps(normalize((runtime.trainer.state_dict(),rng_state())),protocol=5)).hexdigest()

def fork(runtime,resolved):
    original=copy.deepcopy(runtime.config)
    changes=[]
    for key in original:
        if original[key]!=resolved[key]:changes.append(key)
    if set(changes)-{'experiment','budget','ppo'}:raise ValueError('P1 fork changes beyond permitted fields')
    if any(original['ppo'][k]!=resolved['ppo'][k] for k in original['ppo'] if k!='target_kl'):
        raise ValueError('P1 only target_kl may differ')
    if resolved['ppo']['target_kl'] not in {.01,.02}:raise ValueError('unregistered target KL')
    ck=load_anchor(ANCHOR,ANCHOR_HASH,runtime)
    if ck['steps']!=775000:raise ValueError('P1 requires exact selected775k')
    before=state_digest(runtime);environment=runtime.adapter.fingerprint()
    runtime.trainer.config=replace(runtime.trainer.config,target_kl=resolved['ppo']['target_kl'])
    runtime.config=copy.deepcopy(resolved)
    after=state_digest(runtime)
    if before!=after or environment!=runtime.adapter.fingerprint():raise ValueError('fork altered full state')
    active_values(runtime)
    receipt={'parent_checkpoint':ANCHOR,'parent_sha256':ANCHOR_HASH,'steps':ck['steps'],
             'exact_trainer_rng_before':before,'exact_trainer_rng_after':after,
             'environment_fingerprint':environment,'only_scientific_delta':{'target_kl':resolved['ppo']['target_kl']},
             'strict_load':True,'live_target_kl':runtime.trainer.config.target_kl}
    return ck,receipt
