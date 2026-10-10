"""Check operational migration and new bound resume with actual PPO updates."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from .runner import prepare
from .long_budget import authorization,migrate,AUTH_SHA
from .preflight import same,same_graph
from terl_mappo.batch01.contracts import fingerprint
from terl_mappo.batch01.provenance import runtime_manifest,active_values
from terl_mappo.batch01.checkpoints import save_bound,load_bound
from terl_mappo.batch01.evaluation import isolated_rng
from terl_mappo.run import atomic_json,rng_state

def main():
    p=argparse.ArgumentParser();p.add_argument('--delta',required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);p.add_argument('--length',type=int,default=8);a=p.parse_args()
    torch.set_num_threads(1);start=time.monotonic();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    runtime,lock,delta,resolved,_,warm=prepare(a.delta,a.device,out.parent/'source_failures',fork=False)
    grant=authorization(delta);ck,receipt=migrate(runtime,delta)
    if ck is None:
        runtime,lock,delta,resolved,_,warm=prepare(a.delta,a.device,out.parent/'source_failures',fork=True)
    else:
        same(runtime.trainer.state_dict(),ck['trainer']);same_graph(runtime.adapter.state_dict(),ck['runtime']);same({k:v for k,v in rng_state().items() if k!='cuda' or a.device.startswith('cuda')},{k:v for k,v in ck['rng'].items() if k!='cuda' or a.device.startswith('cuda')})
    with isolated_rng():manifest=runtime_manifest(lock,delta,resolved,runtime,{'device':a.device,'max_gpu_allocated_mib':2048})
    manifest.update(run_id=grant['run_id'],execution_mode='PROVISIONAL_LONG',budget_authorization_sha256=AUTH_SHA,max_additional_decisions=grant['authorized_end_step']-grant['start_step'],authorized_end_step=grant['authorized_end_step'],independent_qa_accepted=False,scientific_evidence='PROVISIONAL_NOT_FORMAL')
    checks={'authorization_pinned':True,'old_pilot_caps_preserved':True,'strict_full_state_parent':bool(ck),'scratch_or_stage_transfer':ck is None,'config_and_science_sources_unchanged':True}
    batch,_=runtime.collect(a.length);metrics=runtime.update(batch);assert metrics['minibatch_updates']>0
    cp=out.parent/('bound_'+a.device.replace(':','_')+'.pt')
    step=grant['start_step']+a.length
    save_bound(cp,runtime,manifest,step,int(np.asarray(batch['active_mask']).sum()),int(metrics['minibatch_updates']))
    atomic_json(out.parent/'delta.json',delta);atomic_json(out.parent/'manifest.json',manifest)
    expected,episodes=runtime.collect(8);em=runtime.update(expected);state=copy.deepcopy(runtime.trainer.state_dict());env=runtime.adapter.state_dict();rng=rng_state()
    other,_,_,_,_,_=prepare(a.delta,a.device,out.parent/'source_failures',fork=False);load_bound(cp,other,manifest)
    actual,ae=other.collect(8);am=other.update(actual)
    same(expected,actual);same(episodes,ae);same(em,am);same(state,other.trainer.state_dict());same_graph(env,other.adapter.state_dict());same(rng,rng_state())
    checks['new_bound_full_state_resume_exact']=True
    if ck:
        # Two independent strict restores of the legacy checkpoint must give
        # identical rollout, update, optimizer/ValueNorm/environment and RNG.
        migrate(runtime,delta);b1,e1=runtime.collect(8);m1=runtime.update(b1);s1=copy.deepcopy(runtime.trainer.state_dict());v1=runtime.adapter.state_dict();r1=rng_state()
        migrate(other,delta);b2,e2=other.collect(8);m2=other.update(b2)
        same(b1,b2);same(e1,e2);same(m1,m2);same(s1,other.trainer.state_dict());same_graph(v1,other.adapter.state_dict());same(r1,rng_state())
        checks['parent_next_rollout_update_exact']=True
    result={'status':'LONG_OPERATIONAL_PREFLIGHT_PASS','device':a.device,'length':a.length,'delta_hash':fingerprint(delta),'authorization_sha256':AUTH_SHA,'checks':checks,'receipt':receipt,'warm_start_receipt':warm,'active_values':active_values(runtime),'metrics':metrics,'duration_seconds':time.monotonic()-start,'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2 if a.device.startswith('cuda') else 0,'checkpoint':{'path':str(cp)},'formal_evidence':False}
    atomic_json(out,result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
