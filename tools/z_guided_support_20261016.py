#!/usr/bin/env python3
"""Bounded, isolated Z-guided v0 diagnostics and matched continuation.

No stage promotion, best-screen selection, historical writes or Git publishing.
"""
from __future__ import annotations
import argparse, copy, csv, gzip, hashlib, json, math, os, random, subprocess, sys, time
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
import yaml
from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.trainer import CoCapTrainer, set_global_config
from cocap_voradj.training.runtime_semantics import assert_runtime, initial_state_fingerprint
from cocap_voradj.envs.z_guided_support import ally_progress
from tools import iqn_token_matched_20260919 as matched
from tools import iqn_z_unified_decay_curriculum_20260919 as historical
from tools.collect_iqn_aw_teacher_dataset_20260903 import fixed_midpoint_q, stack_obs, sha256_file
from tools.rollout_voradj_visual import act_evaders
from tools.evaluate_vxy_stage2_cross_retention_20260903 import ring_count

CONFIG_DIR=ROOT/'configs/experiments/iqn_z_guided_support_20261016'
ART=ROOT/'artifacts/2026-10-16_z_guided_support'
RUNTIME=Path('/home/yjq/rl/CoCap1/iqn-z-guided-support-20261016-runtime')
PARENT=Path('/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage1/training/checkpoints/step_1300000.pt')
PARENT_SHA='d35b7984d8ca03fa7b536811794578ff402e9b8ad48589e3318f2cb1077d702b'
LEDGER=['reward_coverage','reward_capture','reward_safety','reward_terminal','reward_speed_repeat','reward_early_speed','reward_deceleration','reward_settled_terminal','reward_motion_penalty']
SCENES=('capture','mixed','coverage')
MODES=('argmax','sample')
SEED=2026101601
EVAL_SEED=2026102600


def clean(value):
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    return matched.json_finite(value)
def write(path,value): matched.atomic_json(Path(path),clean(value))
def now(): return time.strftime('%Y-%m-%dT%H:%M:%S%z')
def config(arm): return matched.resolved(CONFIG_DIR/f'{arm}.yaml')
def model_load(path,device): return CoCapIQN.load(str(path),device=device).eval()
def verify_parent():
    if sha256_file(PARENT)!=PARENT_SHA: raise AssertionError('historical selected parent hash mismatch')
def equal_obs(a,b):
    assert len(a)==len(b)
    for x,y in zip(a,b):
        if x is None: assert y is None;continue
        assert y is not None
        for k in x: np.testing.assert_array_equal(x[k],y[k])
def env_make(cfg,scene,seed):
    sc=matched.scene_config(cfg,scene);set_global_config(sc)
    env=VorAdjEnv(copy.deepcopy(sc),seed=seed);obs=env.reset();assert_runtime(env)
    assert env._z_guided_support_enabled==bool(cfg['voradj']['z_guided_support_enabled'])
    return env,obs

@torch.no_grad()
def actions(model,obs,device,mode='argmax'):
    ids=[i for i,o in enumerate(obs) if o is not None]
    local=[obs[i] for i in ids]
    if mode=='argmax': _,a=fixed_midpoint_q(model,local,device)
    else:
        # IQN has no categorical actor. Native stochastic inference samples
        # 32 quantile fractions, then takes argmax of their mean Q.
        a=model.act(stack_obs(local,device),mode='voradj',epsilon=0,deterministic_quantiles=False).cpu().numpy()
    result=[None]*len(obs)
    for i,v in zip(ids,a): result[i]=int(v)
    return result

def ledger_check(result):
    for reward,info in zip(result.rewards,result.infos):
        m=info['replay_metadata']
        values=[float(m[k]) for k in LEDGER]
        if not np.isfinite(values+[reward]).all(): raise AssertionError('non-finite reward ledger')
        if not np.isclose(reward,sum(values),atol=1e-8,rtol=1e-8): raise AssertionError('ledger mismatch')
        if not np.isclose(m['reward_coverage'],m['reward_ce_center']+m['reward_ce_control']+m['reward_ce_pbrs'],atol=1e-8): raise AssertionError('CE ledger mismatch')

def stats(values): return matched.numeric_summary(list(values))
def correlation(a,b):
    if len(a)<2 or np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])
def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]


class PilotTrainer(CoCapTrainer):
    """Observational guards around the unchanged IQN update/save methods."""
    def _update(self,task):
        loss=super()._update(task)
        if loss is not None and not math.isfinite(float(loss)):
            raise FloatingPointError('non-finite pilot learning loss')
        if self.global_step%1000==0:
            if shutil.disk_usage(self.run_dir).free<20*1024**3:
                raise OSError('pilot disk critical: less than 20 GiB free')
            write(self.run_dir/'heartbeat.json',dict(pid=os.getpid(),current_step=self.global_step,updates=self.update_steps,loss_ema=self.loss_ema,updated_at=now()))
        return loss

    def _save_checkpoint(self,name):
        if not all(torch.isfinite(p).all().item() for p in self.model.parameters()):
            raise FloatingPointError('non-finite pilot model; refusing checkpoint')
        return super()._save_checkpoint(name)

def offline(output,episodes,device,replay_source=None):
    verify_parent();output.mkdir(parents=True,exist_ok=False)
    model=model_load(PARENT,device);cfg0=config('control');cfg1=config('treatment')
    # Restore original Z05 training hyperparameters for this frozen-policy audit.
    source=historical.resolved('z05','stage1')
    source['voradj']['z_guided_support_enabled']=False
    cfg0=source;cfg1=copy.deepcopy(source);cfg1['voradj']['z_guided_support_enabled']=True
    count=Counter();rows=[];episode_rows=[];fingerprints=[]
    for scene_index,scene in enumerate(SCENES):
        for ep in range(episodes):
            seed=2026101700+scene_index*1000+ep
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            old,oo=env_make(cfg0,scene,seed);new,no=env_make(cfg1,scene,seed)
            equal_obs(oo,no)
            fingerprint=initial_state_fingerprint(old)
            assert fingerprint==initial_state_fingerprint(new)
            apf=[ApfAgent(e.a,e.w) for e in old.evaders]
            previous_w={};previous_target={};caps=[];collision=False;capture_step=None;ring2=False;ring3=False
            trace=output/f'{scene}_{seed}.jsonl.gz'
            archive=iter(gzip.open(replay_source/f'{scene}_{seed}.jsonl.gz','rt')) if replay_source else None
            with gzip.open(trace,'wt') as handle:
                for step in range(old.episode_max_length):
                    replay_row=json.loads(next(archive)) if archive is not None else None
                    pa=replay_row['actions'] if replay_row else actions(model,oo,device)
                    ea=replay_row['evader_actions'] if replay_row else act_evaders(old,apf)
                    before_p=np.asarray([[p.x,p.y] for p in old.pursuers]);before_e=np.asarray([[e.x,e.y] for e in old.evaders]).reshape(-1,2)
                    if replay_row:
                        np.testing.assert_array_equal(before_p,replay_row['before_p'])
                        np.testing.assert_array_equal(before_e,np.asarray(replay_row['before_e']).reshape(-1,2))
                    active_enemies=[j for j,e in enumerate(old.evaders) if not e.deactivated]
                    active_ids=[i for i,p in enumerate(old.pursuers) if not p.deactivated]
                    before_cov=old._voradj_coverage_potentials(before_p,before_e)
                    before_z=old.z_state.copy()
                    before_adjacency=old._capture_voronoi_map()['adjacency']
                    a=old.step(pa,ea);b=new.step(pa,ea)
                    equal_obs(a.observations,b.observations);assert a.dones==b.dones
                    np.testing.assert_array_equal(old.z_state,new.z_state)
                    assert old.last_capture_events==new.last_capture_events
                    ledger_check(a);ledger_check(b)
                    after_p=np.asarray([[p.x,p.y] for p in old.pursuers])
                    events=list(old.last_capture_events);caps+=events
                    collision|=bool(old.last_collision_events)
                    if old.evaders and all(e.deactivated and not e.collision for e in old.evaders) and capture_step is None:capture_step=step+1
                    ring=ring_count(old);ring2|=ring>=2;ring3|=ring>=3
                    trace_meta=[]
                    for i in active_ids:
                        x=a.infos[i]['replay_metadata'];y=b.infos[i]['replay_metadata'];sel=y['z_support_selection']
                        count['active']+=1
                        obs_nonzero=x['z_support_observed_nonzero_neighbor']
                        full_nonzero=any(before_z[int(k[1])]>0 for k in before_adjacency.get(('pursuer',i),set()) if k[0]=='pursuer' and int(k[1]) in active_ids)
                        count['full_nonzero_neighbor']+=full_nonzero
                        count['observed_nonzero_neighbor']+=obs_nonzero
                        count['observed_nonzero_not_support']+=obs_nonzero and not x['support_candidate']
                        count['observed_nonzero_coverage']+=obs_nonzero and x['reward_role']=='coverage'
                        count['nonzero_neighbor_not_support']+=x['z_support_nonzero_neighbor_not_support']
                        count['high_z_truncated']+=x['z_support_high_z_truncated_count']
                        if not y['support_reward_blend_active']:continue
                        count['support']+=1;count['self_tie']+=sel['self_tie'];count['friend_tie']+=sel['friend_tie'];count['id_fallback']+=sel['id_fallback']
                        target=sel['target_id'];alpha=sel['alpha'];count['effective_attraction']+=alpha>0
                        switch=i in previous_target and previous_target[i]!=target;count['target_switch']+=switch;previous_target[i]=target
                        enemy_progress=None
                        if active_enemies:
                            j=min(active_enemies,key=lambda j:float(np.linalg.norm(before_p[i]-before_e[j])))
                            enemy_progress=float(np.linalg.norm(before_p[i]-before_e[j])-np.linalg.norm(after_p[i]-before_e[j]))
                        ally=y['z_support_friend_progress']
                        count['ally_toward_enemy_away']+=ally>1e-8 and enemy_progress is not None and enemy_progress< -1e-8
                        count['ally_toward']+=ally>1e-8
                        w=y['support_reward_coverage_weight']
                        residual=(previous_w.get(i,w)-w)*.99*(-float(before_cov[i]))*10
                        previous_w[i]=w
                        ce=x['reward_support_blend_coverage']/x['support_reward_coverage_weight']
                        row=dict(scene=scene,seed=seed,step=step,agent=i,target=target,alpha=alpha,self_tie=sel['self_tie'],friend_tie=sel['friend_tie'],distance_tie=sel['distance_tie'],id_fallback=sel['id_fallback'],friend_distance=sel['selected_distance_normalized']*old._distance_scale(),ally_progress=ally,enemy_progress=enemy_progress,ally_reward=y['z_support_ally_reward'],old_capture=x['reward_support_blend_capture'],new_capture=y['reward_support_blend_capture'],old_ce=x['reward_support_blend_coverage'],new_ce=y['reward_support_blend_coverage'],full_ce=ce,weight_switch_pbrs_residual=residual,target_switch=switch,old_total=a.rewards[i],new_total=b.rewards[i])
                        rows.append(row);trace_meta.append(row)
                    handle.write(json.dumps(clean(dict(step=step,actions=pa,evader_actions=ea,before_p=before_p,before_e=before_e,after_p=after_p,z=old.z_state,old_rewards=a.rewards,new_rewards=b.rewards,events=events,support=trace_meta)),allow_nan=False)+'\n')
                    oo=a.observations;no=b.observations
                    if all(a.dones):break
            rec=old.episode_record();episode_rows.append(dict(scene=scene,seed=seed,initial_state_fingerprint=fingerprint,steps=step+1,captured=bool(rec['captured']),normal_capture=bool(rec['captured'] and caps and all(e['capture_type']=='loose' for e in caps)),collision=collision,ring2=ring2,ring3=ring3,capture_seconds=None if capture_step is None else capture_step*.5,trace=str(trace),sha256=sha256_file(trace)))
            print(json.dumps(dict(phase='offline',scene=scene,episode=ep+1,steps=step+1)),flush=True)
    if rows:
        with (output/'support_transitions.csv').open('w') as h:
            writer=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    report=dict(status='PASS',parent=str(PARENT),parent_sha256=PARENT_SHA,trajectory_source='fresh deterministic replays of frozen historical Z05 selected policy; not archived historical trajectories',episodes=episode_rows,counts=dict(count),self_tie_ratio=count['self_tie']/max(1,count['support']),friend_tie_ratio=count['friend_tie']/max(1,count['support']),full_nonzero_not_support_ratio=count['nonzero_neighbor_not_support']/max(1,count['full_nonzero_neighbor']),observed_nonzero_not_support_ratio=count['observed_nonzero_not_support']/max(1,count['observed_nonzero_neighbor']),observed_nonzero_coverage_ratio=count['observed_nonzero_coverage']/max(1,count['observed_nonzero_neighbor']),ally_toward_enemy_away_ratio=count['ally_toward_enemy_away']/max(1,count['ally_toward']),ally_enemy_progress_correlation=correlation([r['ally_progress'] for r in rows if r['enemy_progress'] is not None],[r['enemy_progress'] for r in rows if r['enemy_progress'] is not None]),distributions={k:stats([r[k] for r in rows if r[k] is not None]) for k in ('alpha','ally_reward','full_ce','old_capture','new_capture','old_ce','new_ce','weight_switch_pbrs_residual')},completed_at=now())
    if count['effective_attraction']<100: report['status']='INSUFFICIENT_SUPPORT_NO_TRAIN'
    report['fixed_replay_source']=str(replay_source) if replay_source else None
    write(output/'report.json',report);return report


def potential_audit(archive,output):
    """Replay saved actions; measure weight-boundary residual, not reward gain."""
    report=json.loads((archive/'report.json').read_text());rows=[]
    for episode in report['episodes']:
        cfg0=historical.resolved('z05','stage1');cfg0['voradj']['z_guided_support_enabled']=False
        cfg1=copy.deepcopy(cfg0);cfg1['voradj']['z_guided_support_enabled']=True
        old,_=env_make(cfg0,episode['scene'],episode['seed']);new,_=env_make(cfg1,episode['scene'],episode['seed'])
        previous={};previous_reset={}
        with gzip.open(episode['trace'],'rt') as h:
            for line in h:
                r=json.loads(line);p=np.asarray(r['before_p']);e=np.asarray(r['before_e']).reshape(-1,2)
                cov=new._voradj_coverage_potentials(p,e)
                a=old.step(r['actions'],r['evader_actions']);b=new.step(r['actions'],r['evader_actions'])
                ledger_check(a);ledger_check(b)
                for i,(x,y) in enumerate(zip(a.infos,b.infos)):
                    x=x['replay_metadata'];y=y['replay_metadata']
                    w0=x['coverage_ce_applied_weight'];w1=y['coverage_ce_applied_weight']
                    if i in previous and not previous_reset[i]:
                        v0,v1=previous[i];phi=-10*float(cov[i])
                        rows.append(dict(scene=episode['scene'],seed=episode['seed'],step=r['step'],agent=i,old_weight=w0,new_weight=w1,old_residual=(v0-w0)*.99*phi,new_residual=(v1-w1)*.99*phi,incremental_residual=((v1-w1)-(v0-w0))*.99*phi,old_total=float(a.rewards[i]),new_total=float(b.rewards[i])))
                    previous[i]=(w0,w1);previous_reset[i]=y['coverage_ce_pbrs_reset_reason']!='none' or b.dones[i]
    output.mkdir(parents=True,exist_ok=False)
    with (output/'potential_weights.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
    write(output/'report.json',dict(status='PASS_FINITE_LEDGER_RESET',rows=len(rows),weight_boundary_steps=sum(abs(r['incremental_residual'])>1e-12 for r in rows),distributions={k:dict(**stats([r[k] for r in rows]),max_abs=max(abs(r[k]) for r in rows)) for k in ('old_residual','new_residual','incremental_residual','old_total','new_total')},interpretation='weighted CE is not a globally telescoping fixed potential; residuals are diagnostic, not extra rewards',completed_at=now()))


def evaluate(checkpoint,cfg,output,device,episodes=50):
    output.mkdir(parents=True,exist_ok=True)
    records=[];model=model_load(checkpoint,device)
    for mode in MODES:
        for si,scene in enumerate(SCENES):
            for ep in range(episodes):
                path=output/f'{mode}_{scene}_{ep:03d}.json'
                if path.exists(): records.append(json.loads(path.read_text()));continue
                seed=EVAL_SEED+si*1000+ep
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                env,obs=env_make(cfg,scene,seed);fingerprint=initial_state_fingerprint(env)
                apf=[ApfAgent(e.a,e.w) for e in env.evaders]
                first_contact=0 if any(env._z_last_direct) else None
                engagement=None;capture_step=None;ce_step=None;events=[];collision=False;aa=False;ring2=False;ring3=False;support=Counter();alphas=[];friend=[];enemy=[];sources=Counter();depth=Counter();target_counts=Counter();support_rows=[]
                for step in range(1,env.episode_max_length+1):
                    result=env.step(actions(model,obs,device,mode),act_evaders(env,apf));ledger_check(result)
                    ring=ring_count(env);ring2|=ring>=2;ring3|=ring>=3
                    if ring>=2 and engagement is None:engagement=step
                    if first_contact is None and any(env._z_last_direct):first_contact=step
                    events.extend(env.last_capture_events);collision|=bool(env.last_collision_events)
                    for e in env.last_collision_events:
                        aa|='pursuer' in str(e.get('type','')) and ('pursuer-pursuer' in str(e) or 'pursuer_pursuer' in str(e))
                    for agent,info in enumerate(result.infos):
                        m=info['replay_metadata']
                        if m['task_label']=='inactive':continue
                        support['active']+=1
                        obsnonzero=m['z_support_observed_nonzero_neighbor'];support['observed_nonzero']+=obsnonzero
                        support['observed_nonzero_not_support']+=obsnonzero and not m['support_candidate']
                        support['high_z_truncated']+=m['z_support_high_z_truncated_count']
                        sources[m['z_dominant_source']]+=1;depth[str(m['z_lineage_hops'])]+=1
                        if m['support_reward_blend_active']:
                            sel=m['z_support_selection'];support['transitions']+=1
                            support['self_tie']+=sel['self_tie'];support['friend_tie']+=sel['friend_tie'];support['id_fallback']+=sel['id_fallback'];support['switch']+=m['z_support_target_switched']
                            support['effective_attraction']+=sel['alpha']>0
                            alphas.append(sel['alpha']);friend.append(m['z_support_friend_progress']);enemy.append(m['support_enemy_distance_progress'])
                            target_counts[str(sel['target_id'])]+=1
                            support_rows.append(dict(step=step,agent=agent,target=sel['target_id'],distance=sel['selected_distance_normalized']*env._distance_scale(),alpha=sel['alpha'],ally_progress=m['z_support_friend_progress'],enemy_progress=m['support_enemy_distance_progress']))
                    if env.evaders and all(e.deactivated and not e.collision for e in env.evaders) and capture_step is None:capture_step=step
                    if env.post_capture_coverage_success and ce_step is None:ce_step=step
                    obs=result.observations
                    if all(result.dones):break
                rec=env.episode_record();captured=bool(rec['captured'])
                # Agent-agent collisions are identified by the episode's typed robot ledger.
                aa=any('agent_agent' in (getattr(p,'collision_types',set()) or set()) for p in env.pursuers)
                record=dict(mode=mode,scene=scene,seed=seed,initial_state_fingerprint=fingerprint,checkpoint=str(checkpoint),checkpoint_sha256=sha256_file(checkpoint),captured=captured,normal_capture=bool(captured and events and all(e['capture_type']=='loose' for e in events)),stationary_capture=any(e['capture_type']=='stationary' for e in events),collision=collision,agent_agent_collision=aa,alive_agents=sum(not p.deactivated for p in env.pursuers),post_capture_ce=env.post_capture_coverage_success,ce_rms=rec['coverage_ce_center_rms'],steps=step,capture_seconds=None if capture_step is None else capture_step*.5,first_contact_seconds=None if first_contact is None else first_contact*.5,engagement_seconds=None if engagement is None else engagement*.5,ring2=ring2,ring3=ring3,unfinished=not (env.post_capture_coverage_success if scene=='coverage' else captured),support=dict(support),alpha=stats(alphas),friend_progress=stats(friend),enemy_progress=stats(enemy),z_sources=dict(sources),z_depths=dict(depth),chosen_neighbor_counts=dict(target_counts),completed_at=now())
                write(path,record)
                with gzip.open(output/f'{mode}_{scene}_{ep:03d}_support.json.gz','wt') as h: json.dump(support_rows,h)
                records.append(record)
                write(output/'progress.json',dict(mode=mode,scene=scene,episode=ep+1,total_episodes=episodes,pid=os.getpid(),updated_at=now()))
    summary={}
    for mode in MODES:
        summary[mode]={}
        for scene in SCENES:
            rr=[r for r in records if r['mode']==mode and r['scene']==scene];n=len(rr)
            result={'episodes':n}
            for k in ('captured','normal_capture','stationary_capture','collision','agent_agent_collision','post_capture_ce','ring2','ring3','unfinished'):
                hits=sum(r[k] for r in rr);result[k]={'count':hits,'rate':hits/n,'wilson95':wilson(hits,n)}
            for k in ('capture_seconds','first_contact_seconds','engagement_seconds','alive_agents','ce_rms'):result[k]=stats([r[k] for r in rr if r[k] is not None])
            result['support']=dict(sum((Counter(r['support']) for r in rr),Counter()))
            summary[mode][scene]=result
    report=dict(checkpoint=str(checkpoint),sha256=sha256_file(checkpoint),episodes_per_scene_mode=episodes,sample_definition='native IQN: sample 32 uniform quantile fractions then argmax mean Q; epsilon=0; not categorical softmax',summary=summary,records=[str(output/f"{r['mode']}_{r['scene']}_{(r['seed']-EVAL_SEED-SCENES.index(r['scene'])*1000):03d}.json") for r in records],completed_at=now())
    write(output/'report.json',report);return report


def compare(control,treatment):
    alerts=[];deltas={};support_count=0
    for mode in MODES:
        deltas[mode]={}
        for scene in SCENES:
            c=control['summary'][mode][scene];t=treatment['summary'][mode][scene]
            assert c['episodes']>=50 and t['episodes']==c['episodes']
            dc=t['normal_capture']['rate']-c['normal_capture']['rate']
            da=t['collision']['rate']-c['collision']['rate']
            deltas[mode][scene]={'normal_capture_delta':dc,'collision_delta':da}
            if scene!='coverage' and dc<-.10-1e-10:alerts.append(f'{mode}/{scene}: capture decline >10pp')
            if da>.10+1e-10:alerts.append(f'{mode}/{scene}: collision increase >10pp')
            support_count+=t['support'].get('effective_attraction',0)
    if support_count<100:alerts.append('insufficient effective support transitions')
    return dict(status='PAUSE_DIAGNOSE' if alerts else 'ENGINEERING_PASS_NOT_NONINFERIORITY',alerts=alerts,deltas=deltas,effective_support_transitions=support_count,policy='no automatic full curriculum or scratch launch; requires reported positive warm-start evidence',completed_at=now())


def export_results(runtime,destination=ART):
    destination.mkdir(parents=True,exist_ok=True)
    registry=[dict(arm='z05_readonly_parent',path=str(PARENT),sha256=PARENT_SHA,bytes=PARENT.stat().st_size,selection='historical Stage1 selected 1300000')]
    curves=[];index=[];statuses={};pairs=[]
    for arm in ('control','treatment'):
        out=runtime/arm
        for p in sorted((out/'checkpoints').glob('*.pt')):
            if p.name=='resume_latest.pt':continue
            registry.append(dict(arm=arm,path=str(p),sha256=sha256_file(p),bytes=p.stat().st_size,selection='registered milestone; no best screen selection'))
        metrics=out/'metrics.jsonl'
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                curves.append(dict(arm=arm,**{k:r.get(k) for k in ('global_step','task','loss','loss_ema','update_steps','epsilon','learning_rate','recent_capture_rate','recent_collision_rate','recent_post_capture_coverage_rate')}))
        for p in sorted((out/'evaluations').glob('step_*/*.json')):
            if p.name in ('report.json','progress.json'):continue
            r=json.loads(p.read_text());index.append(dict(arm=arm,path=str(p),sha256=sha256_file(p),scene=r['scene'],mode=r['mode'],seed=r['seed'],checkpoint_sha256=r['checkpoint_sha256'],captured=r['captured'],collision=r['collision']))
        statuses[arm]={}
        for name in ('status','heartbeat','launch','fresh_init_gate'):
            p=out/f'{name}.json'
            if p.exists():statuses[arm][name]=json.loads(p.read_text())
    for p in sorted(runtime.glob('paired_*.json')):pairs.append(dict(path=str(p),**json.loads(p.read_text())))
    for name,rows in [('checkpoint_registry.csv',registry),('training_curves.csv',curves),('rollout_index.csv',index)]:
        if rows:
            tmp=destination/(name+'.tmp')
            with tmp.open('w') as h:
                writer=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
            tmp.replace(destination/name)
    write(destination/'pilot_status.json',dict(updated_at=now(),runtime=str(runtime),statuses=statuses,paired_gates=pairs,formal_conclusion='pending until completed paired evaluation; engineering pass is not statistical noninferiority'))


def launch(runtime,limit):
    verify_parent()
    suite=ET.parse(ART/'unit_tests.xml').getroot()
    cases=list(suite.iter('testcase'))
    assert len(cases)>=64 and not list(suite.iter('failure')) and not list(suite.iter('error'))
    assert json.loads((ART/'optimizer_smoke.json').read_text())['status']=='PASS'
    assert json.loads((runtime/'offline_fixed_replay/report.json').read_text())['status']=='PASS'
    assert json.loads((runtime/'pbrs_audit/report.json').read_text())['status']=='PASS_FINITE_LEDGER_RESET'
    if shutil.disk_usage(runtime).free<50*1024**3:raise OSError('launch requires 50 GiB free')
    query=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if query:raise RuntimeError('GPU occupied; do not take or stop existing tasks')
    if any((runtime/a).exists() for a in ('control','treatment')):raise FileExistsError('pilot already registered')
    for path in ('src','tools/z_guided_support_20261016.py','configs/experiments/iqn_z_guided_support_20261016'):
        if matched.git('diff','HEAD','--',path):raise RuntimeError('commit implementation before pilot launch')
    launches=[]
    for arm,gpu in [('control','0'),('treatment','1')]:
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH='src:.')
        log=runtime/f'{arm}.log'
        with log.open('xb') as handle:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'train','--arm',arm,'--runtime',str(runtime),'--limit',str(limit),'--device','cuda:0'],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        launches.append(dict(arm=arm,pid=child.pid,gpu=gpu,log=str(log),limit=limit))
    write(runtime/'launch_registry.json',dict(head=matched.git('rev-parse','HEAD'),started_at=now(),launches=launches,gate_policy='25k -> matched evaluation -> warning gate -> 50k -> gate -> 100k; never auto-promote curriculum'))


def train_arm(arm,runtime,device,limit):
    assigned='0' if arm=='control' else '1'
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=assigned or device!='cuda:0' or not torch.cuda.is_available():
        raise RuntimeError('pilot requires assigned physical GPU and process-local cuda:0')
    verify_parent();cfg=config(arm);out=runtime/arm
    if out.exists(): raise FileExistsError('fresh pilot run already exists; use registered exact resume separately')
    cfg.update(device=device,output_root=str(runtime),run_name=arm)
    cfg['pretrained']={'path':str(PARENT),'compatibility_mode':'shape_compatible'}
    out.mkdir(parents=True)
    write(out/'launch.json',dict(pid=os.getpid(),arm=arm,seed=cfg['seed'],gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),head=matched.git('rev-parse','HEAD'),config_sha256=matched.stable_hash(cfg),parent_sha256=PARENT_SHA,maximum_steps=limit,started_at=now()))
    trainer=PilotTrainer(cfg)
    gate=historical._fresh_stage_gate(trainer,'warm-start-pilot',PARENT)
    reference=model_load(PARENT,device)
    assert matched.state_hash(trainer.model)==matched.state_hash(reference)
    del reference
    for env in trainer.envs.values():
        assert env._z_guided_support_enabled==(arm=='treatment')
    write(out/'fresh_init_gate.json',gate)
    for target in [n for n in (25000,50000,100000) if n<=limit]:
        start=time.time();start_step=trainer.global_step
        write(out/'status.json',dict(status='training',pid=os.getpid(),target=target,current_step=start_step,updated_at=now()))
        if trainer.metric_log.closed:
            trainer.metric_log=(out/'metrics.jsonl').open('a');trainer.episode_log=(out/'episodes.jsonl').open('a')
        trainer.total_timesteps=target
        checkpoint=trainer.train()
        write(out/f'milestone_{target}.json',dict(step=trainer.global_step,checkpoint=str(checkpoint),sha256=sha256_file(checkpoint),updates=trainer.update_steps,seconds=time.time()-start,steps_per_second=(target-start_step)/(time.time()-start),completed_at=now()))
        write(out/'status.json',dict(status='evaluation',pid=os.getpid(),current_step=target,updated_at=now()))
        # Evaluation must not perturb continuation's Python/NumPy/Torch RNG.
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        report=evaluate(checkpoint,cfg,out/'evaluations'/f'step_{target:09d}',device)
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
        # Train mode restored explicitly after using a separate evaluation model.
        trainer.model.train()
        set_global_config(trainer.envs[trainer.current_task].config)
        if arm=='treatment':
            other=runtime/'control'/'evaluations'/f'step_{target:09d}'/'report.json'
            deadline=time.time()+6*3600
            while not other.exists():
                if list(runtime.glob('failure_*.json')):raise RuntimeError('matched peer failed; stopping bounded pilot')
                if time.time()>deadline:raise RuntimeError('paired control evaluation deadline exceeded')
                write(out/'status.json',dict(status='waiting_for_matched_control',pid=os.getpid(),current_step=target,updated_at=now()))
                time.sleep(30)
            c=json.loads(other.read_text());paired=compare(c,report)
            # Initial scene equality across arms at each seed/mode.
            for mode in MODES:
                for scene in SCENES:
                    for ep in range(50):
                        left=json.loads((other.parent/f'{mode}_{scene}_{ep:03d}.json').read_text())
                        right=json.loads((out/'evaluations'/f'step_{target:09d}'/f'{mode}_{scene}_{ep:03d}.json').read_text())
                        assert left['initial_state_fingerprint']==right['initial_state_fingerprint']
            write(runtime/f'paired_{target}.json',paired)
        decision=runtime/f'paired_{target}.json';deadline=time.time()+6*3600
        while not decision.exists():
            if list(runtime.glob('failure_*.json')):raise RuntimeError('matched peer failed; stopping bounded pilot')
            if time.time()>deadline:raise RuntimeError('paired gate deadline exceeded')
            time.sleep(30)
        paired=json.loads(decision.read_text())
        export_results(runtime,out/'lightweight_delivery')
        if paired['status']!='ENGINEERING_PASS_NOT_NONINFERIORITY':
            write(out/'status.json',dict(status='paused_by_engineering_gate',current_step=target,pid=os.getpid(),gate=paired,updated_at=now()));return
    write(out/'status.json',dict(status='bounded_pilot_complete',pid=os.getpid(),current_step=trainer.global_step,updated_at=now()))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['offline','potential','evaluate','train','compare','export','launch']);p.add_argument('--output',type=Path);p.add_argument('--runtime',type=Path,default=RUNTIME);p.add_argument('--arm',choices=['control','treatment']);p.add_argument('--device',default='cuda:0');p.add_argument('--episodes',type=int,default=5);p.add_argument('--limit',type=int,choices=[25000,50000,100000],default=25000);p.add_argument('--checkpoint',type=Path);p.add_argument('--replay-source',type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    if a.mode=='offline':offline(a.output or RUNTIME/'offline',a.episodes,a.device,a.replay_source)
    elif a.mode=='potential':potential_audit(a.replay_source,a.output)
    elif a.mode=='evaluate':evaluate(a.checkpoint,config(a.arm),a.output,a.device,a.episodes)
    elif a.mode=='train':
        try:train_arm(a.arm,a.runtime,a.device,a.limit)
        except BaseException as error:
            write(a.runtime/f'failure_{a.arm}.json',dict(arm=a.arm,pid=os.getpid(),error=repr(error),failed_at=now()))
            raise
    elif a.mode=='export':export_results(a.runtime,a.output or ART)
    elif a.mode=='launch':launch(a.runtime,a.limit)
    else:
        c=json.loads((a.runtime/'control/report.json').read_text());t=json.loads((a.runtime/'treatment/report.json').read_text());write(a.output,compare(c,t))

if __name__=='__main__':main()
