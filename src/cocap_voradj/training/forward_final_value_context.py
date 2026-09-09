"""Versioned critic-only pre-action context. Never passed to the local Actor."""
import copy
import numpy as np
import torch
from torch import nn
SCHEMA='forward-final-value-context-v1'
NAMES=(['phase_pre','phase_post','phase_pure','episode_remaining','post_started','post_remaining','ce_hold','ce_success','release_initialized','ce_speed_weight']+
       [f'release_{i}' for i in range(4)]+[f'pursuing_{i}' for i in range(4)]+
       [f'{kind}_{i}' for kind in ('stationary_hold','target_present','target_captured','target_failed') for i in range(8)])


def value_context(env):
    assert len(env.pursuers)==4 and len(env.evaders)<=8
    assert env.episode_max_length==3000 and env.reward_cfg['post_capture_coverage_window_steps']==500
    assert env.reward_cfg['coverage_ce_success_hold_steps']==30
    assert env.reward_cfg['capture_stationary_hold_steps']==10
    assert env._pursuing_release_delay_steps==10 and not env._zone_enabled() and not env._coverage_settle_enabled()
    assert env.reward_cfg['coverage_ce_speed_weight'] in (0.,.0005)
    pure=not env.evaders;pre=not pure and any(not e.deactivated for e in env.evaders)
    post=not pure and not pre
    x=[pre,post,pure,max(0,3000-env.episode_step)/3000,env.post_capture_started,
       max(0,500-env.post_capture_step)/500 if post else 0.,min(env.distribution_hold_steps,30)/30,
       env.post_capture_coverage_success,env._pursuing_flags_initialized,env.reward_cfg['coverage_ce_speed_weight']/.0005]
    x.extend(np.asarray(env._pursuing_release_counters)/10)
    x.extend(bool(getattr(p,'is_pursuing',False)) for p in env.pursuers)
    x.extend(min(env._stationary_capture_counters.get(f'evader_{i}',0),10)/10 for i in range(8))
    x.extend(i<len(env.evaders) for i in range(8))
    x.extend(i<len(env.evaders) and env.evaders[i].deactivated and not env.evaders[i].collision for i in range(8))
    x.extend(i<len(env.evaders) and env.evaders[i].deactivated and env.evaders[i].collision for i in range(8))
    x=np.asarray(x,np.float32);assert x.shape==(len(NAMES),) and np.isfinite(x).all()
    return x


def augment(central,context):
    result=dict(central);x=np.asarray(context,np.float32)
    # Supports one joint state [A,F] and batched joint states [T,A,F].
    x=np.broadcast_to(x[...,None,:],(*central['self'].shape[:-1],len(NAMES)))
    result['self']=np.concatenate((central['self'],x),axis=-1)
    assert result['self'].shape[-1]==9+len(NAMES)
    return result


def expand_value_input(value):
    result=copy.deepcopy(value);old=result.self_encoder[0]
    # Zero extra columns preserve the exact initial function; initialization does not advance RNG.
    state=torch.get_rng_state()
    linear=nn.Linear(old.in_features+len(NAMES),old.out_features).to(old.weight.device)
    torch.set_rng_state(state)
    with torch.no_grad():
        linear.weight.zero_();linear.weight[:,:old.in_features].copy_(old.weight);linear.bias.copy_(old.bias)
    result.self_encoder[0]=linear;result.self_feature_dim=old.in_features+len(NAMES)
    return result
