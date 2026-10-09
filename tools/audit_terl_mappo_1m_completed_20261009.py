#!/usr/bin/env python3
"""Independent closeout audit of complete 1m evidence; never updates models."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from terl_mappo.run import source_hash, atomic_json, file_hash

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'artifacts/2026-10-08_terl_mappo_1m'
PPO_KEYS=('actor_loss','value_loss','entropy','explained_variance','approx_kl','approx_kl_max',
          'actor_grad_norm','value_grad_norm','clip_fraction','post_update_kl',
          'post_update_ratio_mean','post_update_ratio_min','post_update_ratio_max',
          'post_update_clip_fraction','actor_update_l2','value_norm_mean','value_norm_std')

def read(path): return json.loads(Path(path).read_text())

def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines()]

def require(condition,description):
    if not condition: raise AssertionError(description)

def close(actual,expected,description):
    if expected is None: require(actual is None,description)
    else: require(math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-6),description)

def quantiles(values):
    a=np.asarray(values,dtype=float)
    return {'min':float(a.min()),'median':float(np.median(a)),'mean':float(a.mean()),'max':float(a.max())}

def score(result):
    modes=list(result['modes'].values())
    return (sum(m['normal_capture_rate'] for m in modes)/2,
            -sum(m['collision_rate'] for m in modes)/2,
            sum(m['strict_geometry_rate'] for m in modes)/2,
            sum(m['ring3_rate'] for m in modes)/2,-result['steps'])

def audit(out):
    manifest=read(out/'manifest.json');cfg=manifest['config'];parent=Path(manifest['parent_run'])
    original=read(parent/'manifest.json');progress=read(out/'progress.json');control=read(out/'continuation_status.json')
    require(control['status']=='COMPLETE_1M_AND_FINAL_TEST','1m controller complete')
    require(progress['status']=='COMPLETE_BUDGET' and progress['steps']==cfg['budget']==1000000 and progress['finite'],'healthy cumulative 1m')
    require(original['config']['budget']==100000,'100k origin')
    require({k:v for k,v in cfg.items() if k!='budget'}=={k:v for k,v in original['config'].items() if k!='budget'},'only budget changed')
    require(manifest['sources']==original['sources']==source_hash(),'all frozen scientific source hashes')
    require(manifest['parent_manifest_sha256']==file_hash(parent/'manifest.json'),'parent manifest hash')
    require(manifest['resume_checkpoint_sha256']==file_hash(manifest['resume_checkpoint']),'immutable full-state 100k anchor')
    require(manifest['runtime_contract']==original['runtime_contract'],'native runtime contract unchanged')
    for name in ('failure.json','continuation_failure.json','evaluation_failures.json'):
        require(not (out/name).exists(),'no '+name)
    require(not list((out/'evaluations').glob('*.partial.json')),'no partial evaluation remains')

    metrics=rows(out/'metrics.jsonl');by_step={r['steps']:r for r in metrics}
    require(len(metrics)==len(by_step)==progress['updates']==3920,'one unique row per PPO rollout call')
    previous=0;optimizers=0
    for index,row in enumerate(metrics,1):
        step=row['steps'];count=step-previous
        require(count==min(256,25000-previous%25000),'unchanged rollout/candidate cuts')
        require(row['agent_transitions']==3*step and row['update_count']==index,'cumulative counters')
        require(all(not isinstance(v,(float,int)) or math.isfinite(v) for v in row.values()),'finite complete training telemetry')
        require(set(PPO_KEYS)<=row.keys(),'all PPO/critic diagnostics present')
        optimizers+=int(row['minibatch_updates'])
        require(optimizers==row['optimizer_steps'],'paired actor/critic optimizer count')
        previous=step
    require(previous==1000000 and optimizers==progress['optimizer_steps']==22994,'final counters')
    parent_metrics=rows(parent/'metrics.jsonl')
    require(metrics[:len(parent_metrics)]==parent_metrics,'parent metrics prefix preserved exactly')
    first_new=metrics[len(parent_metrics)]
    require(first_new['steps']==100256 and first_new['update_count']==393 and
            first_new['optimizer_steps']==2358 and first_new['agent_transitions']==300768,'first actual continued update')

    deletion={r['steps']:r for r in rows(out/'retention.jsonl')}
    evaluations=[read(p) for p in sorted((out/'evaluations').glob('*.json'))]
    screens=[r for r in evaluations if r['seed_domain']=='screen']
    selections=[r for r in evaluations if r['seed_domain']=='selection']
    finals=[r for r in evaluations if r['seed_domain']=='final']
    require({r['steps'] for r in screens}==set(range(0,1000001,25000)) and len(screens)==41,'all 41 screens')
    require(len(selections)==3 and len(finals)==1,'three held-out candidates and one isolated final')
    domains={};fingerprints={};evidence=[]
    for result in evaluations:
        step=result['steps'];domain=result['seed_domain']
        seed_base={'screen':cfg['screen_seed_base'],'selection':cfg['heldout_seed_base'],'final':2046101800}[domain]
        require(result['seed_base']==seed_base,'predeclared seed domain')
        key={0,25000,50000,100000,250000,500000,750000,1000000}
        n=50 if domain=='final' else 20 if domain=='selection' or step in key else 10
        require(result['episodes_per_mode']==n and set(result['modes'])=={'argmax','sample'},'episode protocol')
        checkpoint=Path(result['checkpoint'])
        if checkpoint.exists():
            require(file_hash(checkpoint)==result['checkpoint_sha256'],'retained immutable checkpoint hash')
            storage='present; independently hashed'
        else:
            require(domain=='screen' and step in deletion and deletion[step]['sha256']==result['checkpoint_sha256'],
                    'deleted checkpoint has evaluation hash and retention ledger')
            metadata=read(out/f'checkpoints/step_{step:09d}.json')
            require(metadata['sha256']==result['checkpoint_sha256'],'deleted checkpoint metadata hash')
            storage='deleted after completed screen; hash corroborated by checkpoint metadata/retention ledger'
        mode_evidence={}
        paired={}
        for mode,summary in result['modes'].items():
            ep=[r for r in result['episodes'] if r['mode']==mode]
            require(len(ep)==summary['episodes']==n and {r['seed'] for r in ep}==set(range(seed_base,seed_base+n)),'unique expected episode seeds')
            paired[mode]={r['seed']:r['initial_fingerprint'] for r in ep}
            domains.setdefault(domain,set()).update(r['seed'] for r in ep)
            for row in ep:
                require(row['normal_capture']==(row['capture'] and not row['collision']),'normal capture excludes simultaneous collision')
                require(row['censored']==(not row['capture']),'native all-noncapture censor convention')
                require(0<row['steps']<=3001,'native Stage1 horizon')
                close(row['capture_time'],row['steps']*.5 if row['capture'] else None,'capture time clock')
                close(sum(row['reward_components'].values()),row['episode_reward'],'per-episode raw reward decomposition')
                close(row['reward_components']['time'],-row['steps'],'native -1 clock reward')
                fp=fingerprints.setdefault(row['seed'],row['initial_fingerprint'])
                require(fp==row['initial_fingerprint'],'physical initial state paired across all checkpoints/modes')
                require(not row['normal_capture'] or row['strict_geometry'],'native strict encirclement in successful normal episodes')
            for field in ('capture','normal_capture','collision','ring2','ring3','strict_geometry','censored'):
                count=sum(bool(r[field]) for r in ep)
                require(count==summary[field+'_count'],'raw '+field+' count')
                close(summary[field+'_rate'],count/n,field+' rate')
            times=[r['capture_time'] for r in ep if r['capture']]
            require(summary['success_n']==len(times) and summary['censored_n']==n-len(times),'success/censor sample counts')
            for suffix,fn in (('mean',np.mean),('median',np.median),('p90',lambda a:np.quantile(a,.9))):
                close(summary['capture_time_'+suffix],float(fn(times)) if times else None,'conditional time '+suffix)
            collision_types={k:sum(k in r['collision_types'] for r in ep)
                for k in sorted({k for r in ep for k in r['collision_types']})}
            require(collision_types==summary['collision_types'],'collision type episode incidence')
            close(summary['episode_reward_mean'],float(np.mean([r['episode_reward'] for r in ep])),'mean raw return')
            close(summary['action_entropy'],float(np.mean([r['action_entropy'] for r in ep])),'mean entropy')
            for k in summary['reward_components']:
                close(summary['reward_components'][k],float(np.mean([r['reward_components'][k] for r in ep])),'mean '+k+' reward')
            per_decision={k:sum(r['reward_components'][k] for r in ep)/sum(r['steps'] for r in ep) for k in summary['reward_components']}
            mode_evidence[mode]={'summary':summary,'ppo_training_at_same_step':by_step.get(step),
                'collision_failure_n':sum(not r['capture'] and r['collision'] for r in ep),
                'time_limit_censored_n':sum(not r['capture'] and not r['collision'] for r in ep),
                'reward_per_agent_decision':per_decision,
                'boundary_agent_time_fraction_lower_bound':max(0,-per_decision['global']/5)}
        require(paired['argmax']==paired['sample'],'argmax/sample paired physical states')
        evidence.append({'steps':step,'domain':domain,'sha256':result['checkpoint_sha256'],
                         'storage_evidence':storage,'modes':mode_evidence})
    for a in domains:
        for b in domains:
            if a!=b: require(not domains[a]&domains[b],'screen/selection/final seed isolation')
    old_final=read(parent/'selection.json')['final']
    require(not domains['final']&set(range(old_final['seed_base'],old_final['seed_base']+50)),'1m final isolated from earlier 100k final')
    expected_candidates={r['steps'] for r in sorted(screens,key=score,reverse=True)[:3]}
    require({r['steps'] for r in selections}==expected_candidates,'predeclared top-three screen selection')
    selected=max(selections,key=score);choice=read(out/'selection.json');final=finals[0]
    require(selected['steps']==choice['selected']['steps']==final['steps']==775000,'heldout selected 775k')
    require(selected['checkpoint_sha256']==choice['selected']['checkpoint_sha256']==final['checkpoint_sha256']==file_hash(out/'checkpoints/best.pt'),'best/final/alias hash')
    positive=[r['steps'] for r in screens if r['steps']>0 and any(m['normal_capture_count']>=2 for m in r['modes'].values())]
    base=next(r for r in screens if r['steps']==0)
    require(len(positive)>=2 and any(m['normal_capture_rate']>=.2 and m['collision_rate']<.5 for m in final['modes'].values()) and
        max(m['normal_capture_rate'] for m in final['modes'].values())>max(m['normal_capture_rate'] for m in base['modes'].values())+.1,
        'independently recomputed strong-positive criteria')
    require(choice['decision']==control['decision']=='TERL_MAPPO_STAGE1_LEARNABLE','authoritative final classification')
    windows=[]
    for low,high in ((0,100000),(100000,250000),(250000,500000),(500000,775000),(775000,1000000)):
        chunk=[r for r in metrics if low<r['steps']<=high]
        windows.append({'from_exclusive':low,'to_inclusive':high,'rollout_calls':len(chunk),
            'kl_early_stop_fraction':float(np.mean([r['kl_early_stop'] for r in chunk])),
            'statistics':{k:quantiles([r[k] for r in chunk]) for k in PPO_KEYS}})
    retained=[]
    for path in sorted((out/'checkpoints').glob('*.pt')):
        ck=torch.load(path,map_location='cpu',weights_only=False)
        require(ck['sources']==manifest['sources'],'retained checkpoint frozen source hashes')
        for part in ('actor','value'):
            require(all(torch.isfinite(t).all().item() for t in ck['trainer'][part].values()),'retained finite model parameters')
        retained.append({'path':str(path),'steps':ck['steps'],'sha256':file_hash(path)})
    episodes=rows(out/'episodes.jsonl')
    return {'audited_at':datetime.now().astimezone().isoformat(),'all_checks_passed':True,
        'decision':choice['decision'],'completed_environment_decisions':progress['steps'],
        'active_agent_transitions':progress['agent_transitions'],'rollout_calls':len(metrics),
        'paired_actor_critic_minibatches':optimizers,'screen_checkpoints':41,
        'screen_episodes':sum(len(r['episodes']) for r in screens),'selection_episodes':sum(len(r['episodes']) for r in selections),
        'final_episodes':len(final['episodes']),'strong_positive_screen_steps':positive,
        'selected_steps':selected['steps'],'selected_sha256':selected['checkpoint_sha256'],
        'all_frozen_sources_match':True,'only_budget_changed':True,'seed_domains_disjoint':True,
        'resume_actual_first_update':first_new,'checkpoint_evidence':evidence,'training_windows':windows,
        'retained_checkpoints':retained,'deleted_own_checkpoint_count':len(deletion),
        'training_episode_count':len(episodes),'training_capture_count':sum(e['capture'] for e in episodes),
        'training_collision_count':sum(e['collision'] for e in episodes),
        'training_episode_ring_note':'terminal-step fields; independent evaluation records full-episode visitation',
        'limits':['one random-origin training seed9','screen/heldout budgets10–20 per mode; final50 per mode',
                  'argmax/sample share physical seed50; do not treat their 100 episodes as 100 independent physical seeds',
                  'best775k; last1m degraded; learnability does not establish optimization stability',
                  'native effective APF/current/padding quirks preserved; no representation/algorithm superiority inference across tasks']}

def plot_training(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=rows(out/'metrics.jsonl');x=[r['steps'] for r in data]
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for ax,keys in zip(axes.flat,(('reward_mean',),('entropy',),('explained_variance',),
                                 ('post_update_kl','approx_kl'),('actor_grad_norm','value_grad_norm'),
                                 ('post_update_clip_fraction',))):
        for key in keys: ax.plot(x,[r[key] for r in data],label=key,linewidth=.6)
        ax.axvline(775000,color='gray',linestyle='--',linewidth=.8)
        ax.set_xlabel('cumulative environment decisions');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ARCHIVE/'training_health_1m.png',dpi=150);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();out=Path(a.run).resolve()
    torch.set_num_threads(1);result=audit(out)
    atomic_json(ARCHIVE/'completion_audit_20261009.json',result);plot_training(out)
    print(json.dumps({k:v for k,v in result.items() if k not in ('checkpoint_evidence','training_windows','retained_checkpoints','resume_actual_first_update')},ensure_ascii=False))

if __name__=='__main__': main()
