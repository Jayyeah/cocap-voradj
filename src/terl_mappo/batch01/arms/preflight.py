"""Scientific ARM assertions and exact bound full-state continuation checks."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from .runner import prepare
from terl_mappo.batch01.provenance import runtime_manifest,active_values
from terl_mappo.batch01.contracts import fingerprint,resolve_config
from terl_mappo.batch01.checkpoints import save_bound,load_bound
from terl_mappo.batch01.evaluation import isolated_rng
from terl_mappo.run import rng_state,atomic_json
from terl_mappo.native import pack_local
from cocap_voradj.training.small_step_ac import tensor_tree
import sys
from terl_mappo.batch01.contracts import ROOT
sys.path.insert(0,str(ROOT/"test"))
from test_terl_mappo_batch01_base_20261009 import same

def same_graph(a,b,seen=None):
    seen=set() if seen is None else seen
    pair=(id(a),id(b))
    if pair in seen:return
    seen.add(pair)
    if isinstance(a,torch.Tensor):assert torch.equal(a.detach().cpu(),b.detach().cpu())
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:same_graph(a[key],b[key],seen)
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same_graph(x,y,seen)
    elif isinstance(a,np.random.RandomState):same_graph(a.get_state(),b.get_state(),seen)
    elif hasattr(a,'query') and hasattr(a,'data'):
        same_graph(a.data,b.data,seen);same_graph(a.query(a.data,k=len(a.data)),b.query(b.data,k=len(b.data)),seen)
        same_graph(a.leafsize,b.leafsize,seen);same_graph(a.boxsize,b.boxsize,seen)
    elif hasattr(a,'__dict__'):
        assert type(a) is type(b)
        same_graph(a.__dict__,b.__dict__,seen)
    else:assert a==b


def main():
    p=argparse.ArgumentParser();p.add_argument('--delta',required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);p.add_argument('--length',type=int,default=8)
    a=p.parse_args();torch.set_num_threads(1);start=time.monotonic();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    runtime,lock,delta,resolved,parent,receipt=prepare(a.delta,a.device,out.parent/'source_failures',fork=True)
    line=delta['line'];checks={};active=active_values(runtime)
    if line=='P1':
        from .p1 import fork
        control=copy.deepcopy(lock['anchor_config']);other,_,_,_,_,_=prepare(a.delta,a.device,out.parent/'source_failures',fork=True)
        original_rng=rng_state();same(runtime.trainer.state_dict(),other.trainer.state_dict());same_graph(runtime.adapter.state_dict(),other.adapter.state_dict())
        changed=copy.deepcopy(resolved);changed['ppo']['target_kl']=.01 if resolved['ppo']['target_kl']==.02 else .02
        from dataclasses import replace
        other.trainer.config=replace(other.trainer.config,target_kl=changed['ppo']['target_kl'])
        same(runtime.trainer.state_dict(),other.trainer.state_dict());same(original_rng,rng_state())
        checks['paired_full_state_before_first_rollout']=True;checks['strict_anchor_receipt']=receipt
        assert runtime.trainer.config.target_kl==resolved['ppo']['target_kl']
    if line=='N1':
        actor=runtime.trainer.actor;obs=tensor_tree(pack_local(runtime.adapter.observations),a.device)
        assert actor.encoder.config.self_feature_dim==4 and actor.encoder.config.max_pursuers==5
        assert (actor.encoder.config.hidden_dim,actor.encoder.config.num_heads,actor.encoder.config.num_layers)==(256,8,4)
        obs['masks'][:,1:]=False
        with torch.no_grad():
            expected=actor(obs);changed={k:v.clone() for k,v in obs.items()};changed['pursuers'].fill_(999);changed['evaders'].fill_(-999);changed['obstacles'].fill_(999)
            same(expected,actor(changed));obs['masks'].zero_();assert torch.isfinite(actor(obs)).all()
            action,logp,latent=actor.sample(obs);same(logp,actor.evaluate_indices(obs,latent)[0])
        assert active['architecture']['critic']['hidden_dim']==256 and active['architecture']['critic']['layers']==4
        checks['native_self4_information_only']=True;checks['masked_pooling_no_target_logprob']=True;checks['fixed_t0_critic']=active['architecture']['critic']
    if line=='R1':
        from .r1 import shaping,angle_delta
        bridge=runtime.hooks.reward_transform;pre=bridge.snapshot(runtime.adapter.env);post=copy.deepcopy(pre)
        assert not shaping(pre,post).any()
        absent=copy.deepcopy(pre);absent['evaders']=np.zeros((0,6));assert not shaping(absent,post).any()
        inactive=copy.deepcopy(pre);inactive['pursuers'][:,3]=0;post['pursuers'][:,:2]+=.1;assert not shaping(inactive,post).any()
        assert abs(float(angle_delta(-np.pi+1e-5,np.pi-1e-5))-2e-5)<1e-10
        checks['pre_target_post_recipient_geometry_inactive_no_target_wrap']=True
    if line in {'T1','C0'}:
        from .secondary_contracts import verify
        checks.update(verify(runtime,line,a.device))
    with isolated_rng():manifest=runtime_manifest(lock,delta,resolved,runtime,{'device':a.device,'max_gpu_allocated_mib':2048})
    first=time.monotonic();batch,_=runtime.collect(a.length);collect_seconds=time.monotonic()-first
    if line=='R1':
        same(batch['raw_rewards'],batch['reward_components']['raw'])
        same(batch['rewards'],batch['raw_rewards']-batch['reward_components']['native_distance']+batch['shaping_rewards'])
        preserved=sum(batch['reward_components'][key] for key in ('time','global','emergency','collision','goal'))
        np.testing.assert_allclose(preserved,batch['rewards']-batch['shaping_rewards'],atol=1e-6)
        assert runtime.trainer.update_count==0
        checks['raw_native_distance_shaping_shaped_separate']=True;checks['other_components_preserved']=True
    if line=='C0':
        from .secondary_contracts import rejected
        corrupt=copy.deepcopy(batch);corrupt['log_prob'][corrupt['active_mask']]+=1.
        before=copy.deepcopy(runtime.trainer.state_dict());before_rng=rng_state()
        rejected(runtime.update,corrupt);same(before,runtime.trainer.state_dict());same(before_rng,rng_state())
        checks['corrupt_behavior_density_rejected_before_optimizer']=True
    first=time.monotonic();metrics=runtime.update(batch);update_seconds=time.monotonic()-first
    assert metrics['minibatch_updates']>0 and all(np.isfinite(x) for x in metrics.values())
    path=out.parent/'resume_probe.pt'
    try:
        save_bound(path,runtime,manifest,a.length,int(np.asarray(batch['active_mask']).sum()),int(metrics['minibatch_updates']))
        expected,episodes=runtime.collect(8);expected_metrics=runtime.update(expected)
        trainer=copy.deepcopy(runtime.trainer.state_dict());env=runtime.adapter.state_dict();rng=rng_state()
        other,_,_,_,_,_=prepare(a.delta,a.device,out.parent/'source_failures',fork=False)
        load_bound(path,other,manifest)
        actual,actual_episodes=other.collect(8);actual_metrics=other.update(actual)
        same(expected,actual);same(episodes,actual_episodes);same(expected_metrics,actual_metrics)
        same(trainer,other.trainer.state_dict());same_graph(env,other.adapter.state_dict());same(rng,rng_state())
        checks['full_state_resume_exact']=True
    finally:path.unlink(missing_ok=True);path.with_suffix('.batch01.json').unlink(missing_ok=True)
    result={'status':'ARM_SELFTEST_PASS','independent_qa':False,'line':line,'device':a.device,'length':a.length,
            'base':delta['base'],'delta_hash':fingerprint(delta),'checks':checks,'active_values':active,
            'metrics':metrics,'collect_seconds':collect_seconds,'update_seconds':update_seconds,'duration_seconds':time.monotonic()-start,
            'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2 if a.device.startswith('cuda') else 0}
    atomic_json(out,result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
