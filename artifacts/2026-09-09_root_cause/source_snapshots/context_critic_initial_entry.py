#!/usr/bin/env python3
"""Context-only fixed-BC MC test, paired to existing 100-update geometry diagnostic."""
import argparse,copy,hashlib,json,os,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from tools import fit_forward_final_mc_critic_20260909 as old
from tools.train_forward_final_ppo_20260909 import make_trainer,tensor_hash,BC_SHA
from tools.run_forward_final_bridge_20260908 import atomic_json
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file
from tools.summarize_forward_final_root_cause_20260909 import calibration,PHASES
from cocap_voradj.training.forward_final import preflight
from cocap_voradj.training.forward_final_value_context import value_context,augment,expand_value_input,NAMES,SCHEMA
from cocap_voradj.training.runtime_semantics import initial_state_fingerprint
from cocap_voradj.training.small_step_ac import tensor_tree


def content_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();out=a.out
    assert not (out/'launch.json').exists(),'fresh output required'
    atomic_json(out/'runtime_preflight.json',preflight());start=time.monotonic()
    trainer=make_trainer(a.device,2026097101);trainer.actor.eval();trainer.value.eval()
    for param in trainer.actor.parameters():param.requires_grad_(False)
    actor_sha=tensor_hash(trainer.actor.state_dict());base=ROOT/'artifacts/2026-09-09_forward_final_d/fixed_mc_critic'
    atomic_json(out/'launch.json',{'schema':SCHEMA,'names':NAMES,'pid':os.getpid(),'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'bc_parent_sha256':BC_SHA,'source_sha256':sha256_file(Path(__file__)),'context_source_sha256':sha256_file(ROOT/'src/cocap_voradj/training/forward_final_value_context.py'),'updates':100,'batch':64,'seed':2026099301,'change':'critic input context only; existing width/depth/target/batches/lr/ValueNorm retained; extra input columns start zero','actor_updates':0,'seeds':{'train':2026099201,'heldout':2027099201},'geometry_reference':str(base),'gate':'post/pure train AND heldout RMSE <= .75 original cold; EV>0; context heldout RMSE <= train-only phase/time baseline; no PPO unless separately approved gates. No failure samples => failure calibration unavailable.'})
    banks={};lineages={};source_pools={};original=old.collect_transition
    for split,seed,count in [('train',2026099201,30),('heldout',2027099201,10)]:
        context=[];lineage=[];snapshots=set();seen=set()
        def hook(t,stream):
            env=stream.env
            if stream.episode not in seen:
                seen.add(stream.episode);pool=list(stream.recovery_init_pool);hashes=[content_hash(s) for s in pool];snapshots.update(hashes)
                matching=[h for h,s in zip(hashes,pool) if np.array_equal(np.asarray(s['positions']),np.asarray([[p.x,p.y] for p in env.pursuers]))]
                source=stream.last_reset_source[stream.task]
                if source=='capture_snapshot':assert matching,'capture source must match actual reset positions'
                lineage.append({'episode':stream.episode,'task':stream.task,'initial_fingerprint':initial_state_fingerprint(env),'reset_source':source,'capture_source_hashes':matching,'pool_snapshot_hashes':hashes})
            context.append(value_context(env));assert env.reward_cfg['coverage_ce_speed_weight']==.0005
            return original(t,stream)
        old.collect_transition=hook
        def progress(n):
            elapsed=time.monotonic()-start
            atomic_json(out/'progress.json',{'status':'collecting_context','split':split,'completed':n+(30 if split=='heldout' else 0),'total':40,'elapsed_seconds':elapsed,'episodes_per_minute':60*(n+(30 if split=='heldout' else 0))/max(elapsed,1),'eta_seconds':elapsed/max(n+(30 if split=='heldout' else 0),1)*(40-n-(30 if split=='heldout' else 0))+15})
        try:central,data=old.collect(trainer,seed,count,out/split,progress)
        finally:old.collect_transition=original
        prior=dict(np.load(base/split/'fixed_bank.npz'));assert len(context)==len(data['episode'])
        for k,v in data.items():np.testing.assert_array_equal(v,prior[k])
        for k,v in central.items():np.testing.assert_array_equal(v,prior['global_'+k])
        banks[split]=(augment(central,np.asarray(context)),data);lineages[split]=lineage;source_pools[split]=snapshots
        np.savez_compressed(out/split/'context.npz',context=np.asarray(context))
        atomic_json(out/split/'context_manifest.json',{'context_sha256':sha256_file(out/split/'context.npz'),'old_bank_sha256':sha256_file(base/split/'fixed_bank.npz'),'all_geometry_target_phase_episode_bit_exact':True,'lineage':lineage})
    assert not source_pools['train']&source_pools['heldout']
    assert not {r['initial_fingerprint'] for r in lineages['train']}&{r['initial_fingerprint'] for r in lineages['heldout']}
    central,data=banks['train'];old_value=trainer.value;trainer.value=expand_value_input(old_value)
    trainer.value_optimizer=torch.optim.Adam(trainer.value.parameters(),lr=trainer.config.critic_lr,eps=1e-5)
    with torch.no_grad():
        batch=tensor_tree({k:v[:64] for k,v in central.items()},trainer.device);plain=dict(batch,self=batch['self'][...,:9])
        initial_error=float((trainer.value(batch)-old_value(plain)).abs().max());assert initial_error<1e-6
    trainer.value_normalizer.update(torch.as_tensor(data['target'],device=trainer.device),torch.as_tensor(data['active'],device=trainer.device));normalization=(float(trainer.value_normalizer.mean),float(trainer.value_normalizer.std))
    random.seed(2026099301);np.random.seed(2026099301);torch.manual_seed(2026099301);rng=np.random.default_rng(2026099301);losses=[];batch_hash=hashlib.sha256()
    for update in range(1,101):
        take=rng.choice(len(data['target']),size=min(64,len(data['target'])),replace=False);batch_hash.update(take.tobytes())
        batch=tensor_tree({k:v[take] for k,v in central.items()},trainer.device);active=torch.as_tensor(data['active'][take],device=trainer.device);target=trainer.value_normalizer.normalize(torch.as_tensor(data['target'][take],device=trainer.device))
        loss=.5*(trainer.value(batch)[active]-target[active]).square().mean();assert torch.isfinite(loss)
        trainer.value_optimizer.zero_grad(set_to_none=True);(trainer.config.value_coef*loss).backward();torch.nn.utils.clip_grad_norm_(trainer.value.parameters(),trainer.config.max_grad_norm);trainer.value_optimizer.step();losses.append(float(loss.detach()))
        if update%10==0:atomic_json(out/'progress.json',{'status':'fitting_context','update':update,'total_updates':100,'elapsed_seconds':time.monotonic()-start,'eta_seconds':(100-update)*.12})
    assert actor_sha==tensor_hash(trainer.actor.state_dict());assert normalization==(float(trainer.value_normalizer.mean),float(trainer.value_normalizer.std))
    scores={};checks={};baseline=json.loads((ROOT/'artifacts/2026-09-09_root_cause/cpu/existing_mc_enrichment.json').read_text())
    for split,(central,data) in banks.items():
        fitted=old.predict(trainer,central);np.savez_compressed(out/split/'predictions.npz',context_fitted=fitted);prior=dict(np.load(base/split/'predictions.npz'));groups={name:data['phase']==i for i,name in enumerate(PHASES)};closure=np.zeros(len(data['phase']),bool)
        for e in np.unique(data['episode']):closure[np.flatnonzero((data['episode']==e)&(data['phase']==0))[-10:]]=True
        groups['closure_last_10_pre_steps']=closure;scores[split]={}
        for name,m in groups.items():
            mask=data['active']&m[:,None];s=calibration(data['target'][mask],fitted[mask]);c=calibration(data['target'][mask],prior['cold'][mask]);g=calibration(data['target'][mask],prior['fitted'][mask]);bl=baseline['scores'][split][name]['safe_success']['phase_time_baseline']
            scores[split][name]={'safe_success':{'context':s,'geometry':g,'cold':c,'phase_time_baseline':bl},'failure':{'rows':0,'reason':'same original bank all safe; cannot assess failure calibration'},'episodes':len(np.unique(data['episode'][m]))}
            if name in PHASES[1:]:
                checks[split+'/'+name]=bool(s['rmse']<=.75*c['rmse'] and s['ev']>0)
                if split=='heldout':checks[split+'/'+name+'/baseline']=bool(s['rmse']<=bl['rmse'])
    checkpoint=out/'context_critic_100.pt';torch.save({'value':trainer.value.state_dict(),'optimizer':trainer.value_optimizer.state_dict(),'normalizer':trainer.value_normalizer.state_dict(),'schema':SCHEMA,'names':NAMES,'actor_sha256':actor_sha,'normalization':normalization},checkpoint)
    decision='CONTEXT_VALUE_SCREEN_PASS_FAILURE_UNTESTED' if all(checks.values()) else 'HOLD_CONTEXT_VALUE_CALIBRATION'
    atomic_json(out/'report.json',{'decision':decision,'checks':checks,'scores':scores,'actor_bit_exact':True,'all_bank_arrays_bit_exact_to_geometry_reference':True,'train_heldout_initial_and_capture_snapshot_disjoint':True,'initial_value_max_error':initial_error,'batch_indices_sha256':batch_hash.hexdigest(),'normalization':normalization,'losses':losses,'checkpoint_sha256':sha256_file(checkpoint),'elapsed_seconds':time.monotonic()-start,'limits':'100-update context-only screen; original data has no failure samples. No automatic PPO. Added context removes identified aliases but does not prove all policy/environment hidden state sufficient or causal PPO efficiency repair.'})
    atomic_json(out/'progress.json',{'status':'complete','decision':decision,'update':100,'elapsed_seconds':time.monotonic()-start,'eta_seconds':0})

if __name__=='__main__':main()
