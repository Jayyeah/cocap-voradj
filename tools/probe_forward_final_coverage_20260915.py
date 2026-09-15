#!/usr/bin/env python3
"""Only after baseline200k: random representation supervision and matched reward attribution."""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tools import forward_final_single_task_20260915 as s
from tools import analyze_forward_final_single_task_20260915 as a


def load_actor_checkpoint(path,device):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    assert payload['schema']==s.SCHEMA and payload['launch']['transition_semantics']==s.SEMANTICS
    assert payload['launch']['source_sha256']==s.source_hashes()
    actor=s.p.make_actor(s.contract()['scratch'],s.contract()['spec']['seed'],device)
    actor.load_state_dict(payload['trainer']['actor'])
    return actor.eval(),payload


def load_teacher(device):
    # Teacher is a separate reference object. No parameter ever enters the student.
    from tools.train_forward_final_ppo_20260909 import BC,BC_SHA
    from tools.distill_forward_final_actor_20260908 import load_actor
    assert s.old.sha(BC)==BC_SHA
    teacher,payload=load_actor(BC,device)
    return teacher,dict(path=str(BC),sha256=BC_SHA,initialization='Existing Final categorical BC; teacher only')


@torch.no_grad()
def collect_supervised(teacher,out):
    spec=s.contract()['spec']['probe_a'];device=next(teacher.parameters()).device
    storage=defaultdict(list);records=[];seed_base=2026291501
    for episode in range(spec['train_episodes']+spec['heldout_episodes']):
        seed=seed_base+episode;s.p.seed_all(seed)
        stream=s.SingleTaskStream('coverage',seed)
        for _ in range(stream.env.episode_max_length):
            ids=[i for i,o in enumerate(stream.observations) if o is not None]
            local={k:np.stack([stream.observations[i][k] for i in ids]) for k in stream.observations[ids[0]]}
            actions=teacher.distribution(s.tensor_tree(local,device)).logits.argmax(-1).cpu().numpy()
            for k,v in local.items():storage['local.'+k].append(v)
            storage['action'].append(actions);storage['episode'].append(np.full(len(ids),episode,np.int32))
            physical=np.full(4,4);physical[ids]=actions
            result=stream.step(physical)
            if all(result.dones):break
        records.append(dict(seed=seed,episode=episode,**stream.telemetry.finish(stream.env)))
        s.p.write_json(out/'progress.json',dict(stage='collect_supervised',pid=os.getpid(),
            episodes=len(records),target_episodes=spec['train_episodes']+spec['heldout_episodes']))
    arrays={k:np.concatenate(v) for k,v in storage.items()}
    train=arrays['episode']<spec['train_episodes']
    assert set(arrays['episode'][train]).isdisjoint(set(arrays['episode'][~train]))
    assert len({r['initial_state_fingerprint'] for r in records})==len(records)
    path=out/'teacher_data.npz';np.savez_compressed(path,**arrays)
    s.p.write_json(out/'teacher_data_manifest.json',dict(path=str(path),sha256=s.old.sha(path),
        teacher_only=True,current_reward=True,transition_semantics=s.SEMANTICS,seed_base=seed_base,
        train_episodes=spec['train_episodes'],heldout_episodes=spec['heldout_episodes'],rows=len(train),
        split='whole independent episodes; no teacher recovery pool',records=records))
    return arrays,records


@torch.no_grad()
def agreement(actor,arrays,indices):
    hits=0;loss=0.;device=next(actor.parameters()).device
    for j in range(0,len(indices),512):
        ids=indices[j:j+512]
        local=s.tensor_tree({k[6:]:v[ids] for k,v in arrays.items() if k.startswith('local.')},device)
        logits=actor.distribution(local).logits
        labels=torch.as_tensor(arrays['action'][ids],device=device)
        hits+=int((logits.argmax(-1)==labels).sum())
        loss+=float(F.cross_entropy(logits,labels,reduction='sum'))
    return dict(rows=len(indices),agreement=hits/len(indices),cross_entropy=loss/len(indices))


def representation(teacher,out,device,baseline):
    spec=s.contract()['spec']['probe_a'];g=s.contract()['spec']['gates']
    arrays,records=collect_supervised(teacher,out)
    quality=float(np.mean([r['ce_success'] for r in records]))
    if quality<g['teacher_quality_min_success_rate']:
        raise RuntimeError('Independent pure-coverage teacher data fail predeclared quality; cannot attribute student failure to representation')
    s.p.seed_all(spec['seed'])
    student=s.p.make_actor(s.contract()['scratch'],spec['seed'],device)
    assert all(q.requires_grad for q in student.parameters())
    initial_backbone=s.p.tensor_hash(student.encoder.state_dict())
    teacher_backbone=s.p.tensor_hash(teacher.encoder.state_dict())
    assert initial_backbone!=teacher_backbone
    train=np.flatnonzero(arrays['episode']<spec['train_episodes'])
    valid=np.flatnonzero(arrays['episode']>=spec['train_episodes'])
    optimizer=torch.optim.Adam(student.parameters(),lr=spec['learning_rate'])
    generator=np.random.default_rng(spec['seed']);history=[]
    # eval() disables dropout exactly as in scratch PPO; autograd remains enabled.
    student.eval()
    initial=dict(train=agreement(student,arrays,train),heldout=agreement(student,arrays,valid))
    for epoch in range(1,spec['epochs']+1):
        order=generator.permutation(train)
        for j in range(0,len(order),spec['batch_size']):
            ids=order[j:j+spec['batch_size']]
            local=s.tensor_tree({k[6:]:v[ids] for k,v in arrays.items() if k.startswith('local.')},torch.device(device))
            labels=torch.as_tensor(arrays['action'][ids],device=device)
            loss=F.cross_entropy(student.distribution(local).logits,labels)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(student.parameters(),.5)
            assert torch.isfinite(grad);optimizer.step()
        row=dict(epoch=epoch,train=agreement(student,arrays,train),heldout=agreement(student,arrays,valid))
        history.append(row)
        s.p.write_json(out/'supervised_learning.json',dict(initial=initial,history=history))
        s.p.write_json(out/'progress.json',dict(stage='supervised_actor_fit',pid=os.getpid(),epoch=epoch,target_epochs=spec['epochs']))
    final_backbone=s.p.tensor_hash(student.encoder.state_dict());assert final_backbone!=initial_backbone
    torch.save(dict(schema=s.SCHEMA,actor=student.state_dict(),optimizer=optimizer.state_dict(),
        initialization='entire_actor_random',initial_backbone_sha256=initial_backbone,
        teacher_weights_loaded_into_student=False,epoch=spec['epochs']),out/'supervised_actor_final.pt')
    ev=s.evaluate(student,'coverage',out/'supervised_eval',spec['epochs'])
    b=baseline[0]['summary']['argmax'];v=ev['summary']['argmax']
    geometric=all(v[k]['p50']<=(1-g['representation_geometry_relative_improvement'])*b[k]['p50'] for k in ('ce_rms','area_cv'))
    learned=(v['ce_success_rate']>=g['representation_rollout_success_rate'] or
        (geometric and v['strict_max_hold']['mean']>=b['strict_max_hold']['mean']+3)) and v['collision_rate']<=.2
    result=dict(decision='REPRESENTATION_SUFFICIENT_FOR_SUPERVISED_LEARNING' if learned else 'REPRESENTATION_LEARNABILITY_SUSPECT',
        initial=initial,terminal_agreement=history[-1],rollout=ev['summary'],teacher_data_success_rate=quality,
        initial_backbone_sha256=initial_backbone,final_backbone_sha256=final_backbone,
        teacher_backbone_sha256=teacher_backbone,student_loaded_iqn_feature_weights=False,
        student_initialization='all parameters random, bias/LayerNorm default constants allowed',
        supervision='AW9 teacher argmax action; all layers trainable; final epoch only, no test checkpoint selection',
        interpretation='Supervised acquisition tests representation sufficiency; failure is suspect, not proof of missing information.')
    s.p.write_json(out/'representation.json',result)
    return result


def distribution(xs):
    values=np.array(xs,float)
    return dict(n=len(values),mean=float(values.mean()),std=float(values.std()),
        quantiles=dict(zip(('p05','p25','p50','p75','p95'),np.quantile(values,[.05,.25,.5,.75,.95]).tolist())))


def rank(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return dict(n=len(x),rho=None,reason='constant_or_insufficient')
    return dict(n=len(x),rho=float(spearmanr(x,y).statistic))


def reward_separability(teacher,out,device,baseline):
    teacher_ev=s.evaluate(teacher,'coverage',out/'teacher_eval',0)
    actor,payload=load_actor_checkpoint(out.parent/'coverage'/'step_200000.pt',device)
    replay=s.evaluate(actor,'coverage',out/'scratch_replay',200000)
    prior=baseline[-1]
    for x,y in zip(prior['records'],replay['records']):
        assert (x['seed'],x['mode'],x['initial_state_fingerprint'])==(y['seed'],y['mode'],y['initial_state_fingerprint'])
        for k in ('discounted_return','length','ce_rms','area_cv','ce_success'):
            assert x[k]==y[k],f'Baseline replay differs: {k}'
    groups={}
    for mode in ('argmax','sample'):
        strong=[r for r in teacher_ev['records'] if r['mode']==mode]
        bad=[r for r in replay['records'] if r['mode']==mode]
        for x,y in zip(strong,bad):assert x['seed']==y['seed'] and x['initial_state_fingerprint']==y['initial_state_fingerprint']
        delta=np.array([x['discounted_return']-y['discounted_return'] for x,y in zip(strong,bad)])
        rng=np.random.default_rng(2026391501)
        boots=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(1)
        summaries={}
        for name,rows in (('strong',strong),('bad',bad)):
            summaries[name]=dict(summary=s.summarize(rows),
                returns=distribution([r['discounted_return'] for r in rows]),
                ce_pbrs=distribution([r['discounted_components']['reward_ce_pbrs'] for r in rows]),
                safety_control=distribution([r['discounted_components']['reward_safety']+r['discounted_components']['reward_ce_control'] for r in rows]))
        pooled=strong+bad
        correlations={}
        within_policy={}
        for name,rows in (('strong',strong),('bad',bad)):
            within_policy[name]={}
            for key in ('ce_rms_improvement','ce_success','mission_seconds','length'):
                eligible=[r for r in rows if r.get(key) is not None]
                within_policy[name][key]=rank([r['discounted_return'] for r in eligible],[r[key] for r in eligible])
        for key in ('ce_rms_improvement','ce_success','mission_seconds','length'):
            eligible=[r for r in pooled if r.get(key) is not None]
            correlations[key]=rank([r['discounted_return'] for r in eligible],[r[key] for r in eligible])
        quality=float(np.mean([r['ce_success'] for r in strong]))
        low,high=np.quantile(boots,[.025,.975])
        stable=quality>=.8 and float(low)>0 and np.mean(delta>0)>=.8 and np.median(delta)>0
        groups[mode]=dict(**summaries,paired_return_difference=distribution(delta),
            matched_pair_positive_fraction=float(np.mean(delta>0)),paired_mean_bootstrap95=[float(low),float(high)],
            high_quality_return_stably_higher=bool(stable),rank_correlation=correlations,within_policy_rank_correlation=within_policy,
            matched_seed_ids=[r['seed'] for r in strong],
            mission_time_scope='Success-only times; failed episodes are censored, length is separately reported.')
    allbad=replay['records']
    ratios=[]
    for r in allbad:
        comp=r['rewards'];ce=abs(comp['reward_ce_center'])+abs(comp['reward_ce_pbrs'])
        other=abs(comp['reward_safety'])+abs(comp['reward_ce_control'])
        ratios.append(ce/max(ce+other,1e-12))
    weak_scale=np.median(ratios)<.5
    ambiguous=not all(g['high_quality_return_stably_higher'] for g in groups.values())
    teacher_good=groups['argmax']['strong']['summary']['ce_success_rate']>=.8
    result=dict(groups=groups,current_reward_recomputed=True,baseline_replay_exact=True,
        transition_semantics=s.SEMANTICS,reward_source_sha256=s.old.sha(s.ROOT/'src/cocap_voradj/envs/voronoi_adjacency.py'),
        ce_fraction_distribution=distribution(ratios),reward_scale_candidate=bool(teacher_good and (weak_scale or ambiguous)),
        reason='Candidate only if high-quality teacher exists and CE absolute share <.5 or matched return separation is not stable.',
        rank_scope='Episode ranks; pooled policy association is not an independent causal estimate; matched seed deltas also reported.')
    s.p.write_json(out/'reward_separability.json',result)
    return result


def scale_probe(out,device):
    trainer,_,payload=s.load(out.parent/'coverage'/'step_200000.pt',device)
    s.p.seed_all(2026491501)
    stream=s.SingleTaskStream('coverage',2026491501)
    result=[];fixed=[]
    for chunk in range(8):
        rollout=s.p.empty_rollout();ce=[];control=[];safety=[]
        for _ in range(256):
            row,_=s.collect_transition(trainer,stream)
            s.p.append_transition(rollout,row)
            c=stream.last_components
            ce.append(c['reward_ce_center']+c['reward_ce_pbrs'])
            control.append(c['reward_ce_control']);safety.append(c['reward_safety'])
        batch=s.p.stack_rollout(rollout);shaping=np.asarray(ce)
        variants=[]
        for multiplier in (1.,2.):
            reward=batch['rewards']+(multiplier-1)*shaping
            t=lambda k:torch.as_tensor(batch[k],device=trainer.device)
            adv,ret=s.compute_gae(torch.as_tensor(reward,device=trainer.device),
                trainer._denormalize_values(t('values')),trainer._denormalize_values(t('next_values')),
                t('terminated'),t('active_mask'),gamma=.99,gae_lambda=.95,
                truncated=t('truncated'),episode_end=t('episode_end'))
            mask=t('active_mask');adv=adv[mask];ret=ret[mask]
            norm=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
            # Same actual ValueNorm updater, no optimizer step. It only affects value targets.
            value_norm=copy.deepcopy(trainer.value_norm);value_norm.update(ret)
            variants.append(dict(raw_adv=adv.cpu().numpy(),norm=norm.cpu().numpy(),ret=ret.cpu().numpy(),
                ce_abs_ratio=float(np.abs(multiplier*shaping).sum()/max(np.abs(reward).sum(),1e-12)),
                value_norm_mean=float(value_norm.mean),value_norm_std=float(value_norm.std)))
        x,y=variants;cos=float(np.dot(x['norm'],y['norm'])/(np.linalg.norm(x['norm'])*np.linalg.norm(y['norm'])+1e-12))
        rms=float(np.sqrt(np.mean((x['norm']-y['norm'])**2)));flip=float(np.mean((x['norm']>0)!=(y['norm']>0)))
        result.append(dict(chunk=chunk,normalized_advantage_cosine=cos,normalized_advantage_rms_delta=rms,
            advantage_sign_flip_fraction=flip,variants=[dict(multiplier=i+1,raw_return=distribution(v['ret']),
                raw_advantage=distribution(v['raw_adv']),normalized_advantage=distribution(v['norm']),
                ce_abs_to_total_abs_ratio=v['ce_abs_ratio'],value_norm_mean=v['value_norm_mean'],value_norm_std=v['value_norm_std'])
                for i,v in enumerate(variants)]))
        fixed.append(dict(batch=batch,ce=shaping,control=np.asarray(control),safety=np.asarray(safety)))
    path=out/'fixed_scale_rollouts.pt';torch.save(fixed,path)
    g=s.contract()['spec']['gates']
    null=all(r['normalized_advantage_rms_delta']<g['scale_null_normalized_advantage_rms_delta_max'] and
             r['advantage_sign_flip_fraction']<g['scale_null_advantage_sign_flip_max'] and
             r['normalized_advantage_cosine']>g['scale_null_advantage_cosine_min'] for r in result)
    ratio_changed=any(abs(r['variants'][1]['ce_abs_to_total_abs_ratio']-r['variants'][0]['ce_abs_to_total_abs_ratio'])>1e-3 for r in result)
    raw_changed=any(abs(r['variants'][1]['raw_return']['mean']-r['variants'][0]['raw_return']['mean'])>1e-3 for r in result)
    null=bool(null or not ratio_changed or not raw_changed)
    report=dict(effectively_null=null,decision='REWARD_SCALE_INTERVENTION_EFFECTIVELY_NULL' if null else 'NON_NULL_FIXED_ROLLOUT_INTERVENTION',
        ce_total_ratio_changed=ratio_changed,raw_return_changed=raw_changed,chunks=result,
        fixed_rollouts_sha256=s.old.sha(path),optimizer_updates=0,
        semantics='Same state/action/V/logp/terminal/active rows; only existing center+PBRS doubled; control/safety unchanged.')
    s.p.write_json(out/'scale_probe.json',report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    root=args.root.resolve();out=root/'attribution'
    assert not out.exists(),'Fresh attribution output required'
    baseline=a.reports(root/'coverage');gate=a.coverage_result(baseline)
    assert gate['decision']=='NO_MEANINGFUL_LEARNING','Reward modification forbidden after baseline learning'
    out.mkdir();torch.set_num_threads(1)
    s.p.write_json(out/'launch.json',dict(pid=os.getpid(),baseline_gate=gate,protocol=s.contract(),
        source_sha256=s.source_hashes(),probe_source_sha256=s.old.sha(__file__),transition_semantics=s.SEMANTICS))
    try:
        teacher,teacher_manifest=load_teacher(args.device)
        s.p.write_json(out/'teacher.json',teacher_manifest)
        rep=representation(teacher,out,args.device,baseline)
        reward=reward_separability(teacher,out,args.device,baseline)
        sufficient=rep['decision']=='REPRESENTATION_SUFFICIENT_FOR_SUPERVISED_LEARNING'
        candidate=reward['reward_scale_candidate']
        scale=scale_probe(out,args.device) if sufficient and candidate else None
        start=sufficient and candidate and not scale['effectively_null']
        r2=dict(decision='START_R2' if start else ('REWARD_SCALE_INTERVENTION_EFFECTIVELY_NULL' if scale and scale['effectively_null'] else 'DO_NOT_START_R2'),
            baseline_no_learning=True,representation_sufficient=sufficient,reward_scale_candidate=candidate,
            effectively_null=scale['effectively_null'] if scale else None,
            baseline_checkpoint_sha256=s.old.sha(root/'coverage'/'step_200000.pt'),
            common_contract_sha256=s.old.sha(s.CONFIG),source_sha256=s.source_hashes())
        s.p.write_json(out/'r2_gate.json',r2)
        s.p.write_json(out/'report.json',dict(status='COMPLETE',representation=rep,reward_separability=reward,
            scale_probe=scale,r2_gate=r2,teacher=teacher_manifest))
    except BaseException:
        s.p.write_json(out/'failure.json',dict(status='STOP_IMPLEMENTATION',traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
