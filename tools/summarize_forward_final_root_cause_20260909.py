#!/usr/bin/env python3
"""CPU-only reward alignment and existing fixed-MC-bank calibration enrichment."""
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
from scipy.stats import pearsonr,spearmanr
from tools.run_forward_final_bridge_20260908 import atomic_json
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file
PHASES=('pre_capture','post_capture','pure_coverage')


def corr(x,y):
    x,y=np.asarray(x),np.asarray(y)
    if len(x)<3 or np.std(x)<1e-10 or np.std(y)<1e-10:return {'pearson':None,'spearman':None}
    return {'pearson':float(pearsonr(x,y).statistic),'spearman':float(spearmanr(x,y).statistic)}


def calibration(y,p):
    y,p=np.asarray(y),np.asarray(p)
    if not len(y):return {'rows':0,'rmse':None,'mae':None,'ev':None,'slope':None,'intercept':None,'rank_correlation':None}
    varied=np.std(p)>1e-10
    slope,intercept=np.polyfit(p,y,1) if varied else (None,None)
    return {'rows':len(y),'rmse':float(np.mean((p-y)**2)**.5),'mae':float(np.mean(abs(p-y))),'ev':float(1-np.var(y-p)/np.var(y)) if np.var(y)>1e-10 else None,'slope':float(slope) if varied else None,'intercept':float(intercept) if varied else None,'rank_correlation':corr(y,p)['spearman'],'bias':float(np.mean(p-y))}


def alignment(rows):
    result={}
    for policy in sorted(set(r['policy'] for r in rows)):
        for scene in ('mixed','coverage'):
            all_rows=[r for r in rows if r['policy']==policy and r['scene']==scene]
            selected=[r for r in all_rows if r['safe_complete']]
            groups={'episode':('mission_seconds','total'), 'pure_quality':('ce_rms','total')} if scene=='coverage' else {'episode':('mission_seconds','total'),'pre_capture':('capture_seconds','pre_capture'),'post_capture':('recovery_seconds','post_capture')}
            summary={'episodes':len(all_rows),'safe_episodes':len(selected),'groups':{}}
            for label,(time_key,prefix) in groups.items():
                group={}
                usable=[r for r in selected if r.get(time_key) is not None]
                if not usable:
                    summary['groups'][label]={'unavailable':'No measured quality/time values in source artifact'}
                    continue
                for discount in ('discounted','undiscounted'):
                    key=prefix+'_'+discount
                    group[discount]=corr([r[time_key] for r in usable],[r[key] for r in usable])
                    group[discount]['mean_return']=float(np.mean([r[key] for r in usable])) if usable else None
                    ordered=sorted(usable,key=lambda r:r[time_key])
                    group[discount]['time_tertiles']={name:{'n':len(idx),'time_mean':float(np.mean([ordered[j][time_key] for j in idx])) if len(idx) else None,'return_quantiles':np.percentile([ordered[j][key] for j in idx],[10,50,90]).tolist() if len(idx) else [],'return_mean':float(np.mean([ordered[j][key] for j in idx])) if len(idx) else None} for name,idx in zip(('fast','medium','slow'),np.array_split(np.arange(len(ordered)),3))}
                summary['groups'][label]=group
            result[policy+'/'+scene]=summary
    return result


def returns(reward,phase):
    result={};r=np.asarray(reward,dtype=np.float64)
    for label,mask in [('total',np.ones(len(r),bool))]+[(name,phase==i) for i,name in enumerate(PHASES)]:
        v=r[mask]
        result[label+'_undiscounted']=float(v.sum())
        result[label+'_discounted']=float(np.dot(.99**np.arange(len(v)),v))
    return result


def fixed_bank(out):
    root=ROOT/'artifacts/2026-09-09_forward_final_d/fixed_mc_critic';banks={};sources=[];episode_rows=[];fingerprints={}
    for split in ('train','heldout'):
        data=dict(np.load(root/split/'fixed_bank.npz'));pred=dict(np.load(root/split/'predictions.npz'));manifest=json.loads((root/split/'bank_manifest.json').read_text());step=np.zeros(len(data['episode']),int);phase_age=step.copy();fingerprints[split]=[]
        reports={r['episode']:r for r in manifest['episodes']}
        for eid in np.unique(data['episode']):
            ix=np.flatnonzero(data['episode']==eid);step[ix]=np.arange(len(ix))
            for ph in np.unique(data['phase'][ix]):
                ids=ix[data['phase'][ix]==ph];phase_age[ids]=np.arange(len(ids))
            first=ix[0];fingerprints[split].append(hashlib.sha256(b''.join(data[k][first].tobytes() for k in sorted(data) if k.startswith('global_'))).hexdigest())
            target=data['target'][ix].astype(float);rew=target.copy();rew[:-1]-=.99*target[1:]
            report=reports[int(eid)];capture_steps=int((data['phase'][ix]==0).sum())
            episode_rows.append({'policy':'bc_mc_'+split,'scene':'coverage' if report['task']=='voradj_coverage' else 'mixed','safe_complete':bool(report['ce_success'] and not report['collision']),'mission_seconds':len(ix)*.5,'capture_seconds':capture_steps*.5,'recovery_seconds':(len(ix)-capture_steps)*.5,'ce_rms':None,**returns(rew.mean(1),data['phase'][ix])})
        # Predictors use elapsed phase time only, never realized future completion time.
        data['time_bin']=phase_age//50
        banks[split]=(data,pred,manifest)
        sources.append({'split':split,'bank_sha256':sha256_file(root/split/'fixed_bank.npz'),'prediction_sha256':sha256_file(root/split/'predictions.npz')})
    train=banks['train'][0];means={};phase_means={}
    for ph in range(3):
        mask=train['active']&(train['phase'][:,None]==ph);phase_means[ph]=float(train['target'][mask].mean())
        for b in np.unique(train['time_bin'][train['phase']==ph]):
            mask=train['active']&((train['phase']==ph)&(train['time_bin']==b))[:,None]
            means[ph,int(b)]=float(train['target'][mask].mean())
    scores={}
    for split,(data,pred,manifest) in banks.items():
        pred['phase_time_baseline']=np.broadcast_to(np.asarray([means.get((int(ph),int(b)),phase_means[int(ph)]) for ph,b in zip(data['phase'],data['time_bin'])])[:,None],data['target'].shape)
        groups={ph:data['phase']==i for i,ph in enumerate(PHASES)}
        closure=np.zeros(len(data['phase']),bool)
        for eid in np.unique(data['episode']):
            ix=np.flatnonzero((data['episode']==eid)&(data['phase']==0));closure[ix[-10:]]=True
        groups['closure_last_10_pre_steps']=closure
        scores[split]={}
        for ph,mask in groups.items():
            active=data['active']&mask[:,None]
            scores[split][ph]={'safe_success':{k:calibration(data['target'][active],v[active]) for k,v in pred.items()},'failure':{'rows':0,'reason':'No failures in the original 40-episode bank; cannot assess collision calibration'},'episodes':len(np.unique(data['episode'][mask]))}
    overlap=set(fingerprints['train'])&set(fingerprints['heldout']);assert not overlap
    atomic_json(out/'existing_mc_enrichment.json',{'classification':'EXPERIMENT RESULT','sources':sources,'scores':scores,'initial_global_fingerprint_overlap':list(overlap),'fingerprints':fingerprints,'baseline':'train-only phase x 50-step elapsed-phase bins; unseen bins fall back to train phase mean; no outcome/time-to-completion feature','limits':'Existing bank lacks CE hold/release counters, pool snapshot fingerprints and full context. Distinct initial global fingerprints do not alone prove disjoint source pools; independent pools verified in collector. All 40 episodes safe. Closure is retrospective diagnostic stratum only. No fitting of neural critic in this audit.'})
    atomic_json(out/'bc_mc_alignment.json',{'results':alignment(episode_rows),'limits':'Reward reconstructed from float32 full MC targets r_t=G_t-gamma G_(t+1), true terminal; roundoff; same training reward but native stream reset, not evaluation reset; pure CE completion quality absent from source bank','episode_rows':episode_rows})


def replay_summary(root,out):
    records=json.loads((root/'records.json').read_text());rows=[]
    for rec in records:
        p=root/rec['raw_path'];assert sha256_file(p)==rec['raw_sha256'];data=dict(np.load(p))
        r={**rec,**returns(data['reward'].mean(1),data['phase'])}
        r['components']={key:returns(data[key].mean(1),data['phase']) for key in data if key.startswith('reward_')}
        rows.append(r)
    paired={}
    for scene in ('mixed','coverage'):
        bc={r['seed']:r for r in rows if r['policy']=='bc' and r['scene']==scene and r['safe_complete']}
        pp={r['seed']:r for r in rows if r['policy']=='ppo_low_lr' and r['scene']==scene and r['safe_complete']}
        ids=sorted(set(bc)&set(pp));paired[scene]={'n':len(ids),'deltas':{k:float(np.mean([pp[i][k]-bc[i][k] for i in ids])) if ids else None for k in ('mission_seconds','total_discounted','total_undiscounted','post_capture_discounted','post_capture_undiscounted')},'slower_and_higher_discounted_return':sum(pp[i]['mission_seconds']>bc[i]['mission_seconds'] and pp[i]['total_discounted']>bc[i]['total_discounted'] for i in ids)}
    for scene,entry in paired.items():
        bc={r['seed']:r for r in rows if r['policy']=='bc' and r['scene']==scene and r['safe_complete']}
        pp={r['seed']:r for r in rows if r['policy']=='ppo_low_lr' and r['scene']==scene and r['safe_complete']}
        ids=sorted(set(bc)&set(pp));rng=np.random.default_rng(2026099501)
        resamples=rng.integers(0,len(ids),size=(10000,len(ids)))
        entry['exploratory_paired_bootstrap95']={k:np.percentile(np.asarray([pp[i][k]-bc[i][k] for i in ids])[resamples].mean(1),[2.5,97.5]).tolist() for k in entry['deltas']}
        prefix='post_capture_discounted' if scene=='mixed' else 'total_discounted'
        entry['phase_component_discounted_deltas']={k:float(np.mean([pp[i]['components'][k][prefix]-bc[i]['components'][k][prefix] for i in ids])) for k in next(iter(bc.values()))['components']}
        if scene=='mixed':
            entry['recovery_slower_pairs']=sum(pp[i]['recovery_seconds']>bc[i]['recovery_seconds'] for i in ids)
            entry['recovery_slower_and_higher_phase_return']=sum(pp[i]['recovery_seconds']>bc[i]['recovery_seconds'] and pp[i]['post_capture_discounted']>bc[i]['post_capture_discounted'] for i in ids)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,scene,xkey,ykey,title in zip(axes,('mixed','mixed','coverage'),('mission_seconds','recovery_seconds','mission_seconds'),('total_discounted','post_capture_discounted','total_discounted'),('Full mixed mission','Post-capture recovery','Pure coverage')):
        for policy,label in [('iqn','IQN'),('bc','Frozen BC'),('ppo_low_lr','Low-LR PPO512')]:
            subset=[r for r in rows if r['policy']==policy and r['scene']==scene and r['safe_complete']]
            ax.scatter([r[xkey] for r in subset],[r[ykey] for r in subset],s=24,alpha=.7,label=label)
        ax.set(title=title,xlabel='Completion time (s)',ylabel='Discounted return');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Safe episodes; gamma=.99; native training reward; phase discount restarts at entry',fontsize=10);fig.tight_layout()
    fig.savefig(out/'reward_efficiency.png',dpi=160);fig.savefig(out/'reward_efficiency.pdf');plt.close(fig)
    atomic_json(out/'p0_alignment.json',{'classification':'EXPERIMENT RESULT','complete':json.loads((root/'progress.json').read_text())['status']=='complete','results':alignment(rows),'paired_ppo_minus_bc':paired,'return_convention':'sum over time of mean reward across four pursuers, gamma .99; phase discount restarted at phase entry; reward CE speed .0005; separate components overlap and must not be summed blindly','limits':'Safe-conditioned correlations are descriptive, initial difficulty confounds cross-episode rank; paired comparison controls initial state, not intermediate state or action randomness. Existing diagnostic seed family; no new reliability sample count. IQN has same evaluation reset seeds; this is not historical training-pool replay.','episode_rows':rows})

def iqn_dataset(out):
    root=ROOT/'artifacts/2026-09-08_forward_final/c1_dataset';manifest=json.loads((root/'manifest.json').read_text());rows=[]
    for shard in manifest['shards']:
        path=root/shard['path'];assert sha256_file(path)==shard['sha256']
        data=dict(np.load(path));rec=json.loads((root/'episodes'/f"episode_{shard['episode']:05d}.json").read_text())['record']
        if not rec['safe_complete']:continue
        steps=np.unique(data['transition_index']);rewards=np.asarray([data['reward'][data['transition_index']==t].sum()/4 for t in steps]);phase=np.asarray([data['phase_id'][data['transition_index']==t][0] for t in steps])
        rows.append({**rec,'policy':'iqn_c1_reward0',**returns(rewards,phase)})
    atomic_json(out/'iqn_existing_alignment.json',{'classification':'EXPERIMENT RESULT','dataset_manifest_sha256':sha256_file(root/'manifest.json'),'results':alignment(rows),'episode_rows':rows,'limits':'Existing C1 IQN greedy data; reward CE speed 0 (frozen collection), NOT PPO training .0005. Independent collection seeds; not paired with low-LR eval. Only safe trajectories selected, source manifest retains all failures.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--replay',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    fixed_bank(a.out)
    iqn_dataset(a.out)
    if a.replay:replay_summary(a.replay,a.out)
