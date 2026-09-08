#!/usr/bin/env python3
"""C2: frozen Final IQN decision feature, categorical head only, no AC env adapter."""
from __future__ import annotations
import argparse, dataclasses, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
import torch.nn.functional as F
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.models.small_step_ac import CategoricalGridActor
from cocap_voradj.models.continuous.local_entity_token_encoder import LegacyVorAdjFeatureBackbone,LegacyVorAdjFeatureBackboneConfig
from cocap_voradj.training.small_step_ac import tensor_tree
from cocap_voradj.training.forward_final import CONTRACT,TEACHER,TEACHER_SHA,preflight,make_env
from tools.run_forward_final_bridge_20260908 import atomic_json
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file
from tools.collect_forward_final_dataset_20260908 import PHASES,ROLES
SCHEMA='forward-final-categorical-bc-v1'


def make_actor(teacher,device):
    config=LegacyVorAdjFeatureBackboneConfig(**{f.name:getattr(teacher.config,f.name) for f in dataclasses.fields(LegacyVorAdjFeatureBackboneConfig) if hasattr(teacher.config,f.name)})
    backbone=LegacyVorAdjFeatureBackbone(config).to(device)
    backbone.load_legacy_iqn_state_dict(teacher.state_dict(),strict=True)
    for p in backbone.parameters():p.requires_grad_(False)
    env,_=make_env('mixed',2026090801)
    actor=CategoricalGridActor(backbone,np.asarray(env.pursuers[0].action_list),orthogonal_policy_head=True).to(device).eval()
    return actor


def load_actor(path,device):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    assert payload['schema']==SCHEMA and payload['contract']==CONTRACT and payload['teacher_sha256']==TEACHER_SHA
    teacher=CoCapIQN.load(str(TEACHER),device=device).eval()
    actor=make_actor(teacher,device);actor.load_state_dict(payload['actor_state_dict'],strict=True)
    assert_backbone(actor,teacher)
    return actor.eval(),payload


def assert_backbone(actor,teacher):
    original=actor.encoder.map_legacy_iqn_decision_keys(teacher.state_dict())
    assert len(original)==len(actor.encoder.state_dict())
    for k,v in actor.encoder.state_dict().items():assert torch.equal(v,original[k]),f'Backbone changed: {k}'
    assert not actor.training and not any(p.requires_grad for p in actor.encoder.parameters())
    return len(original)


def load_data(root):
    manifest=json.loads((root/'manifest.json').read_text());audit=json.loads((root/'distribution_audit.json').read_text())
    assert manifest['schema']=='forward-final-teacher-dataset-v1' and manifest['status']=='complete'
    assert manifest['contract']==CONTRACT and manifest['teacher_sha256']==TEACHER_SHA and audit['status']=='PASS'
    storage={}
    for shard in manifest['shards']:
        path=root/shard['path'];assert sha256_file(path)==shard['sha256']
        with np.load(path,allow_pickle=False) as x:
            for k in x.files:
                if k.startswith('local_obs.') or k in ('teacher_q','greedy_action','episode','phase_id','role_id'):
                    storage.setdefault(k,[]).append(x[k])
    arrays={k:np.concatenate(v) for k,v in storage.items()}
    assert len(arrays['teacher_q'])==manifest['row_count']
    assert np.isfinite(arrays['teacher_q']).all() and np.array_equal(arrays['teacher_q'].argmax(1),arrays['greedy_action'])
    return arrays,manifest


def split_pairs(episodes):
    # Keep every trajectory and its same-base mixed/coverage pair together.
    validation=(episodes//2)%5==0
    train=np.flatnonzero(~validation);valid=np.flatnonzero(validation)
    assert len(train) and len(valid)
    assert not set((episodes[train]//2).tolist())&set((episodes[valid]//2).tolist())
    return train,valid


def local_batch(arrays,indices,device):
    return tensor_tree({k.removeprefix('local_obs.'):v[indices] for k,v in arrays.items() if k.startswith('local_obs.')},torch.device(device))


def calibrate_temperature(q,phase,train,seed,target_entropy=.35):
    rng=np.random.default_rng(seed);selected=[]
    for i in PHASES.values():
        candidates=train[phase[train]==i];assert len(candidates)
        selected.extend(rng.choice(candidates,min(2048,len(candidates)),replace=False).tolist())
    values=torch.as_tensor(q[selected],dtype=torch.float64);values-=values.max(1,keepdim=True).values
    def entropy(t):
        lp=F.log_softmax(values/t,dim=1);return float(-(lp.exp()*lp).sum(1).mean())
    low=1e-8;high=max(float(values.abs().max()),1.)
    assert entropy(low)<target_entropy<entropy(high),'Entropy target cannot be calibrated'
    for _ in range(60):
        mid=(low*high)**.5
        if entropy(mid)<target_entropy:low=mid
        else:high=mid
    temperature=(low*high)**.5
    return temperature,{'temperature':temperature,'target_entropy_nats':target_entropy,'actual_entropy_nats':entropy(temperature),'calibration_rows':len(selected),'sampling':'train only; equal capped rows per phase; no temperature sweep','q_gap_percentiles':np.percentile(np.diff(np.sort(q[selected],axis=1)[:,-2:],axis=1),[10,50,90]).tolist()}


@torch.no_grad()
def evaluate_head(actor,features,q,arrays,indices,temperature,batch=4096):
    storage={'hits':[],'kl':[],'hard_ce':[],'entropy':[],'target_entropy':[],'q_regret':[]}
    for start in range(0,len(indices),batch):
        take=torch.as_tensor(indices[start:start+batch],device=features.device)
        lp=F.log_softmax(actor.logits_head(actor.policy(features[take])),dim=1)
        target_lp=F.log_softmax((q[take]-q[take].max(1,keepdim=True).values)/temperature,dim=1);target=target_lp.exp()
        greedy=q[take].argmax(1);pred=lp.argmax(1)
        rows={'hits':pred.eq(greedy).float(),'kl':(target*(target_lp-lp)).sum(1),'hard_ce':F.nll_loss(lp,greedy,reduction='none'),'entropy':-(lp.exp()*lp).sum(1),'target_entropy':-(target*target_lp).sum(1),'q_regret':q[take].max(1).values-q[take].gather(1,pred[:,None]).squeeze(1)}
        for k,v in rows.items():storage[k].append(v.cpu().numpy())
    storage={k:np.concatenate(v) for k,v in storage.items()}
    result={'rows':len(indices),**{k:float(v.mean()) for k,v in storage.items()}}
    for key,groups in (('phase_id',PHASES),('role_id',ROLES)):
        result[key]={name:{'rows':int((mask:=arrays[key][indices]==i).sum()),**{k:float(v[mask].mean()) if mask.any() else None for k,v in storage.items()}} for name,i in groups.items()}
    result['macro_phase_agreement']=float(np.mean([r['hits'] for r in result['phase_id'].values()]))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=2048)
    p.add_argument('--seed',type=int,default=2026095101);args=p.parse_args();out=args.output_root
    assert args.epochs>0 and args.batch_size>0
    if (out/'launch.json').exists():raise ValueError('Use fresh output; existing checkpoints are immutable')
    torch.manual_seed(args.seed);np.random.seed(args.seed);start=time.monotonic()
    atomic_json(out/'runtime_preflight.json',preflight());assert sha256_file(TEACHER)==TEACHER_SHA
    arrays,manifest=load_data(args.dataset_root);train,valid=split_pairs(arrays['episode'])
    temperature,calibration=calibrate_temperature(arrays['teacher_q'],arrays['phase_id'],train,args.seed)
    atomic_json(out/'temperature.json',calibration)
    teacher=CoCapIQN.load(str(TEACHER),device=args.device).eval();actor=make_actor(teacher,args.device)
    keys=assert_backbone(actor,teacher);optimizer=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=3e-4,eps=1e-5)
    launch={'schema':SCHEMA,'contract':CONTRACT,'teacher_sha256':TEACHER_SHA,'dataset_manifest_sha256':sha256_file(args.dataset_root/'manifest.json'),'dataset_root':str(args.dataset_root),'rows':manifest['row_count'],'train_rows':len(train),'validation_rows':len(valid),'temperature':temperature,'epochs':args.epochs,'batch_size':args.batch_size,'seed':args.seed,'backbone_keys':keys,'backbone_frozen':True,'forward_mode':'eval including during head gradient updates','selection':'maximum validation macro-phase action agreement; lower KL breaks ties','learning_rate':3e-4,'split':'episode-pair group %5==0 validation; no agent-row split','gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'ppo_updates':0}
    atomic_json(out/'launch.json',launch)
    cached=[]
    with torch.no_grad():
        for offset in range(0,len(arrays['teacher_q']),512):
            selected=np.arange(offset,min(offset+512,len(arrays['teacher_q'])))
            obs=local_batch(arrays,selected,args.device);feature=actor.encoder(obs)
            if offset==0:
                torch.testing.assert_close(feature,teacher.voradj_single_feature(obs,teacher.features(obs)),rtol=1e-5,atol=1e-6)
                torch.testing.assert_close(actor.distribution(obs).logits,F.log_softmax(actor.logits_head(actor.policy(feature)),dim=1),rtol=1e-5,atol=1e-6)
            cached.append(feature.cpu())
            if offset%10240==0:atomic_json(out/'progress.json',{'status':'feature_cache','rows_completed':offset+len(selected),'total_rows':len(arrays['teacher_q']),'elapsed_seconds':time.monotonic()-start})
    features=torch.cat(cached).to(args.device);del cached
    q=torch.as_tensor(arrays['teacher_q'],device=args.device,dtype=torch.float32)
    history=[];best=(-1.,float('-inf'));best_path=None
    for epoch in range(1,args.epochs+1):
        actor.eval();order=np.random.permutation(train)
        for offset in range(0,len(order),args.batch_size):
            take=torch.as_tensor(order[offset:offset+args.batch_size],device=args.device)
            lp=F.log_softmax(actor.logits_head(actor.policy(features[take])),dim=1)
            target=F.softmax((q[take]-q[take].max(1,keepdim=True).values)/temperature,dim=1)
            loss=F.kl_div(lp,target,reduction='batchmean');assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],1.);optimizer.step()
        assert_backbone(actor,teacher)
        validation=evaluate_head(actor,features,q,arrays,valid,temperature)
        history.append({'epoch':epoch,'validation':validation})
        key=(validation['macro_phase_agreement'],-validation['kl'])
        if key>best:
            best=key;best_path=out/f'actor_epoch_{epoch:03d}.pt'
            payload={**launch,'epoch':epoch,'validation':validation,'actor_state_dict':{k:v.detach().cpu() for k,v in actor.state_dict().items()}}
            torch.save(payload,best_path)
            assert torch.load(best_path,map_location='cpu',weights_only=True)['contract']==CONTRACT
        elapsed=time.monotonic()-start
        atomic_json(out/'progress.json',{'status':'training','epoch':epoch,'total_epochs':args.epochs,'best_checkpoint':str(best_path),'elapsed_seconds':elapsed,'eta_seconds':elapsed/epoch*(args.epochs-epoch)})
        atomic_json(out/'history.json',history)
    payload=torch.load(best_path,map_location=args.device,weights_only=True);actor.load_state_dict(payload['actor_state_dict'],strict=True);assert_backbone(actor,teacher)
    report={**launch,'status':'complete','best_epoch':payload['epoch'],'checkpoint':str(best_path),'checkpoint_sha256':sha256_file(best_path),'train':evaluate_head(actor,features,q,arrays,train,temperature),'validation':evaluate_head(actor,features,q,arrays,valid,temperature),'next_gate':'C3 frozen task evaluation required; supervised agreement is not full-policy migration PASS'}
    atomic_json(out/'report.json',report);atomic_json(out/'progress.json',{'status':'complete','best_checkpoint':str(best_path),'elapsed_seconds':time.monotonic()-start,'eta_seconds':0})
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
