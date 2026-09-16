#!/usr/bin/env python3
"""One corrected Final scratch contract, stream, optimizer and evaluator for both tasks."""
from __future__ import annotations
import argparse
import copy
import dataclasses
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
import numpy as np
import torch
import yaml
from tools import preflight_forward_final_scratch_20260914 as p
from tools import forward_final_overnight_20260914 as old
from cocap_voradj.training.forward_final import scene_config, flatten
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.small_step_ac import compute_gae, tensor_tree
from cocap_voradj.training.runtime_semantics import assert_runtime, initial_state_fingerprint
from tools.train_forward_final_ppo_20260909 import collect_transition
CONFIG = ROOT/'configs/experiments/forward_final_single_task_20260915/common.yaml'
SCHEMA = 'forward-final-single-task-v1'
SEMANTICS = old.SEMANTICS
REWARD_KEYS = (*p.COMPONENTS, 'reward_ce_center', 'reward_ce_control', 'reward_ce_pbrs',
               'reward_ce_terminal_correction', 'reward_support_blend_capture', 'reward_support_blend_coverage')


def contract():
    spec = yaml.safe_load(CONFIG.read_text())
    assert spec['schema'] == SCHEMA and spec['transition_semantics'] == SEMANTICS
    assert spec['budgets'] == dict(capture=500000, coverage=200000, coverage_r2=200000)
    assert spec['checkpoint_interval'] == 25000 and spec['eval_episodes'] == 20
    base = p.load_contract(ROOT/spec['parent'])
    return dict(spec=spec, scratch=base)


def task_config(task):
    assert task in ('capture', 'coverage')
    cfg = copy.deepcopy(scene_config('mixed'))
    # A single common spawn contract; do not inherit coverage's cluster/reset pool.
    cfg['env']['num_evaders'] = int(task == 'capture')
    cfg['voradj'].update(single_task_objective=task,
        capture_episode_ends_on_capture=task == 'capture',
        support_reward_blend_enabled=task == 'capture',
        support_reward_capture_weight=1.0, support_reward_coverage_weight=0.0,
        episode_schedule='single_task', scenes=[task],
        recovery=dict(capture_state_pool_capacity=0, captured_state_ratio=0.,
                      map_random_ratio_within_non_capture=1., spawn_mode='map_random'))
    # Same mature Final control coefficient; no teacher asset is read.
    cfg['reward']['coverage_ce_speed_weight'] = .0005
    return cfg


def check_env(env, task, actor=None):
    radius = 20.
    expected_config = task_config(task)
    sensing_metadata = None
    if 'onboard_sensing' in env.config.get('voradj', {}):
        from cocap_voradj.envs.density_sensing import runtime_metadata, enable_v2
        sensing_metadata = runtime_metadata(env)
        radius = sensing_metadata['resolved_onboard_radius']
        expected_config = enable_v2(expected_config)
    facts = assert_runtime(env, actor, categorical=actor is not None, expected={
        'pursuers':4, 'evaders':int(task=='capture'), 'topology':'friendly_voronoi_comm_v0',
        'enemy_token_rule':'surface_radius', 'global_enemy_flag':False,
        'support_capture_weight':1., 'support_coverage_weight':0.,
        'support_blend_enabled':task=='capture', 'capture_reward_mode':'ring_importance_ms_v0',
        'capture_radius':8., 'capture_k':3, 'enemy_radius':radius,
        'action_mode':'unicycle_discrete', 'decision_dt':.5, 'physics_dt':.05,
        'a_longitudinal_max':.4, 'omega_max':float(np.pi/6), 'v_max':3., 'drag':.4/3,
        'collision_semantics':'synchronized_swept_v1'})
    assert env.transition_semantics == SEMANTICS
    assert env.config == expected_config, 'Resolved runtime configuration drift'
    assert env.episode_max_length == 3000 and env.reward_cfg['min_active_pursuers'] == 4
    assert env._ce_coverage_enabled() and env._ring_importance_ms_enabled()
    assert not env._pure_capture_all_capture_enabled(), 'Historical reward override forbidden'
    assert env._support_reward_capture_component_mode() == 'approach_only'
    assert env._support_reward_capture_target_mode() == 'neighbor_visible'
    assert env._pursuing_release_delay_steps == 10 and env._vct_ls_apply_release_delay()
    assert env._vct_ls_sensing_radius('obstacle') == radius
    assert env.env_cfg['width'] == env.env_cfg['height'] == 120
    assert len(env.obstacles) == 1
    assert env.env_cfg['pursuer_spawn_mode'] == 'map_random'
    assert env.env_cfg['pursuer_spawn_min_sep'] == 15
    for mapping in (env._capture_voronoi_map(), env._coverage_voronoi_map()):
        assert all(k[0] == 'pursuer' for k in mapping['keys'])
    np.testing.assert_allclose(env.pursuers[0].action_list,
        [(a,w) for a in (-.4,0,.4) for w in (-np.pi/6,0,np.pi/6)])
    return dict(**facts, transition_semantics=SEMANTICS, task=task, recovery_pool=0,
                capture_terminal=task=='capture', ce_reward_enabled=task=='coverage')


def make_env(task, seed):
    cfg = task_config(task)
    p.set_global_config(cfg)
    env = VorAdjEnv(copy.deepcopy(cfg), seed=int(seed))
    observations = env.reset()
    check_env(env, task)
    return env, observations


def parity():
    configs = {t:task_config(t) for t in ('capture','coverage')}
    a,b = (flatten(configs[t]) for t in ('capture','coverage'))
    reasons = {
        'env.num_evaders':'Capture requires one enemy; Coverage requires none.',
        'voradj.single_task_objective':'Select current Final capture or coverage reward objective.',
        'voradj.capture_episode_ends_on_capture':'Capture success is true terminal.',
        'voradj.support_reward_blend_enabled':'No support role exists without an enemy.',
        'voradj.scenes':'Only the requested single task can reset.'}
    rows = [dict(field=k,capture=a.get(k),coverage=b.get(k),classification='TASK_REQUIRED',reason=reasons[k])
            for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)]
    assert {r['field'] for r in rows} == set(reasons)
    return dict(schema=SCHEMA, UNEXPLAINED=0, differences=rows,
        common_contract=contract(), resolved=configs,
        common_changes_from_full_task={
            'recovery':'No pool, alternating scenes or post-capture reset sources in either task.',
            'spawn':'Both use the Final mixed map_random/min_sep15 common root; coverage cluster overlay is intentionally not inherited.',
            'support':'Capture surviving .5/(.5)=1, removed coverage weight=0; direct CR-MS unchanged.',
            'coverage_reward':'Capture coverage reward returns zero before CE evaluation; geometry for tokens retained.',
            'budgets':'TASK_REQUIRED capture500k vs coverage200k; no metric-based early stop.',
            'initialization':'Same seed and random parameter constructor; independent processes/RNG/Adam/ValueNorm.'})


def shape_reward(outcome, multiplier):
    """R2 doubles current center/PBRS only, including weighted terminal correction."""
    assert multiplier in (1.,2.)
    for i, info in enumerate(outcome.infos):
        m = info['replay_metadata']
        if multiplier == 2.:
            delta = m['reward_ce_center'] + m['reward_ce_pbrs']
            outcome.rewards[i] += delta
            for key in ('reward_ce_center','reward_ce_pbrs','reward_ce_terminal_correction'):
                m[key] *= multiplier
            m['reward_coverage'] += delta
            m['reward_total'] += delta
        np.testing.assert_allclose(sum(m[k] for k in p.COMPONENTS),outcome.rewards[i],atol=1e-8)
    return outcome


class Telemetry:
    def __init__(self, env, task):
        self.task = task
        self.fingerprint = initial_state_fingerprint(env)
        self.initial_rms = env.episode_record(task='coverage')['coverage_ce_center_rms']
        self.count = Counter()
        self.reward_sum = Counter()
        self.reward_discounted = Counter()
        self.ce_rms = []
        self.ce_max = []
        self.cv = []
        self.capture_types = []
        self.capture_step = None
        self.collision_types = Counter()
        self.last_components = None
        self.terminal_transition_seen = False
        self.post_capture_terminal_metadata_transitions = 0

    def observe(self, env, outcome, active):
        if getattr(self, 'terminal_transition_seen', False):
            raise AssertionError('telemetry.observe called after the episode terminal transition')
        self.count['length'] += 1
        step = self.count['length']
        self.count['active_rows'] += len(active)
        events = list(env.last_capture_events)
        sizes = [sum(np.hypot(q.x-e.x,q.y-e.y)<=8 for q in env.pursuers if not q.deactivated)
                 for e in env.evaders if not e.deactivated]
        sizes += [len(e['participants']) for e in events]
        size = max(sizes,default=0)
        for k in (2,3): self.count[f'ring{k}_steps'] += int(size>=k)
        self.count['ring3_run'] = self.count['ring3_run']+1 if size>=3 else 0
        self.count['ring3_max_hold'] = max(self.count['ring3_max_hold'],self.count['ring3_run'])
        self.count['strict_max_hold'] = max(self.count['strict_max_hold'],env.distribution_hold_steps)
        self.capture_types.extend(e['capture_type'] for e in events)
        if events and self.capture_step is None: self.capture_step = step
        for event in env.last_collision_events:
            self.collision_types[event.get('type','unknown')] += 1
        metas = [o['replay_metadata'] for o in outcome.infos]
        post_capture_metadata = [m['phase'] == 'post_capture' for m in metas]
        if any(post_capture_metadata):
            # Pure-Capture must never roll into recovery. Runtime state can,
            # however, label the terminal successor as post_capture after the
            # final enemy is deactivated (capture or terminal enemy loss).
            # Accept that label only on the immediate terminal transition; a
            # non-terminal post-capture row remains a hard error.
            assert self.task == 'capture'
            assert all(post_capture_metadata)
            assert all(outcome.dones)
            assert all(i['terminated'] and not i['truncated'] for i in outcome.infos)
            self.post_capture_terminal_metadata_transitions = (
                getattr(self, 'post_capture_terminal_metadata_transitions', 0) + 1
            )
        self.count['support_rows'] += sum(metas[i]['support_candidate'] for i in active)
        if self.task == 'capture':
            assert all(m['reward_coverage']==m['reward_ce_pbrs']==m['reward_ce_control']==0 for m in metas)
            if events:
                assert all(outcome.dones) and all(i['terminated'] and not i['truncated'] for i in outcome.infos)
        else:
            assert not events and not env.evaders
            assert not any(m['support_candidate'] or m['reward_capture'] or m['reward_terminal'] for m in metas)
        if all(outcome.dones):
            self.terminal_transition_seen = True
        self.last_components = {k:np.array([m[k] for m in metas],np.float32) for k in REWARD_KEYS}
        for k in REWARD_KEYS:
            v = float(np.mean(self.last_components[k]))
            self.reward_sum[k] += v
            self.reward_discounted[k] += .99**(step-1)*v
        d = env.last_distribution_metrics
        for values,key in ((self.ce_rms,'ce_center_rms'),(self.ce_max,'ce_center_max'),(self.cv,'area_cv')):
            v=float(d.get(key,float('nan')))
            if np.isfinite(v): values.append(v)

    def finish(self, env):
        rec = env.episode_record(task='coverage' if self.task=='coverage' else 'mix')
        n = self.count['length']
        capture = bool(rec['captured']) if self.task=='capture' else False
        ce = bool(env.post_capture_coverage_success) if self.task=='coverage' else False
        collision = bool(self.collision_types) or bool(rec.get('collision_event'))
        rms = float(rec['coverage_ce_center_rms'])
        maximum = float(rec['coverage_ce_center_max'])
        cv = float(rec['coverage_strict_area_cv'])
        clean = lambda x: float(x) if np.isfinite(x) else None
        return dict(task=self.task,**dict(self.count),initial_state_fingerprint=self.fingerprint,
            captured=capture,normal_capture=capture and 'loose' in self.capture_types,
            stationary_capture=capture and 'stationary' in self.capture_types,
            ce_success=ce,collision=collision,boundary=bool(self.collision_types['boundary']),
            collision_types=dict(self.collision_types),capture_step=self.capture_step,
            capture_seconds=.5*self.capture_step if self.capture_step else None,
            mission_seconds=.5*n if (capture or ce) and not collision else None,
            time_to_ce=.5*n if ce else None,initial_ce_rms=clean(self.initial_rms),
            ce_rms=clean(rms),ce_max=clean(maximum),area_cv=clean(cv),
            ce_rms_improvement=clean(self.initial_rms-rms),
            ce_rms_trajectory=p.stats(self.ce_rms),ce_max_trajectory=p.stats(self.ce_max),
            area_cv_trajectory=p.stats(self.cv),rewards=dict(self.reward_sum),
            discounted_components=dict(self.reward_discounted),
            discounted_return=sum(self.reward_discounted[k] for k in p.COMPONENTS),
            total_return=sum(self.reward_sum[k] for k in p.COMPONENTS),
            reset_source='common_map_random',pool_size=0,
            post_capture_terminal_metadata_transitions=getattr(
                self, 'post_capture_terminal_metadata_transitions', 0),
            post_capture_transitions=0)


class SingleTaskStream:
    def __init__(self, task, seed, multiplier=1.):
        assert task=='coverage' or multiplier==1.
        self.task_name=task;self.multiplier=multiplier
        self.envs={task:make_env(task,seed)[0]};self.task=task
        self.apf_agents={task:[p.ApfAgent(e.a,e.w) for e in self.env.evaders]}
        self.observations=self.env.get_observations()
        self.episode=0;self.telemetry=Telemetry(self.env,task)
        self.episode_metrics=[];self.env_steps=0
    @property
    def env(self): return self.envs[self.task]
    def set_clock(self,step):
        self.env.total_steps=2000000+int(step)
        assert self.env.reward_cfg['coverage_ce_speed_weight']==.0005
    def step(self,indices):
        p.set_global_config(self.env.config)
        active=[i for i,o in enumerate(self.observations) if o is not None]
        result=self.env.step([int(a) if i in active else None for i,a in enumerate(indices)],
                             p.CoCapTrainer._evader_actions(self,self.task))
        shape_reward(result,self.multiplier)
        self.telemetry.observe(self.env,result,active)
        self.observations=result.observations;self.env_steps+=1
        self.last_components=self.telemetry.last_components
        return result
    def finish(self):
        row=dict(episode=self.episode,**self.telemetry.finish(self.env))
        self.episode_metrics.append(row);self.episode+=1
        p.set_global_config(self.env.config)
        self.observations=self.env.reset()
        self.apf_agents[self.task]=[p.ApfAgent(e.a,e.w) for e in self.env.evaders]
        check_env(self.env,self.task_name)
        self.telemetry=Telemetry(self.env,self.task_name)
        return row
    def summary(self):
        return dict(env_steps=self.env_steps,episodes=len(self.episode_metrics),pool_size=0,
            task=self.task_name,post_capture_transitions=0,completed=summarize(self.episode_metrics))


def summarize(rows):
    if not rows: return {'episodes':0}
    keys=('length','capture_seconds','mission_seconds','time_to_ce','ce_rms','ce_max','area_cv',
          'strict_max_hold','ring3_max_hold','discounted_return','total_return','ce_rms_improvement')
    return dict(episodes=len(rows),
        **{k+'_rate':float(np.mean([r[k] for r in rows])) for k in
           ('captured','normal_capture','stationary_capture','ce_success','collision','boundary')},
        **{k:p.stats([r.get(k) for r in rows if r.get(k) is not None]) for k in keys},
        **{f'ring{k}_visitation':float(np.mean([r.get(f'ring{k}_steps',0)>0 for r in rows])) for k in (2,3)},
        reward_components={k:p.stats([r['rewards'][k] for r in rows]) for k in REWARD_KEYS},
        discounted_components={k:p.stats([r['discounted_components'][k] for r in rows]) for k in REWARD_KEYS})


@torch.no_grad()
def evaluate(actor, task, out, step, episodes=20, max_steps=None, multiplier=1., seed_base=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    file=out/f'eval_step_{step:06d}.json'
    actor_hash=p.tensor_hash(actor.state_dict())
    if file.exists():
        result=json.loads(file.read_text());assert result['actor_sha256']==actor_hash
        return result
    device=next(actor.parameters()).device
    rng=p.rng_state(device);records=[];start=time.monotonic()
    seed_base=seed_base or contract()['spec']['eval_seed_base']
    try:
        actor.eval()
        for mode in ('argmax','sample'):
            for episode in range(episodes):
                seed=seed_base+episode;p.seed_all(seed)
                stream=SingleTaskStream(task,seed,multiplier)
                check_env(stream.env,task,actor)
                entropies=[];hist=np.zeros(9,dtype=np.int64)
                for t in range(max_steps or stream.env.episode_max_length):
                    active=[i for i,o in enumerate(stream.observations) if o is not None]
                    obs=tensor_tree({k:np.stack([stream.observations[i][k] for i in active])
                                    for k in stream.observations[active[0]]},device)
                    d=actor.distribution(obs);a=d.logits.argmax(-1) if mode=='argmax' else d.sample()
                    entropies.extend(d.entropy().cpu().tolist());hist+=np.bincount(a.cpu().numpy(),minlength=9)
                    action=np.full(4,4);action[active]=a.cpu().numpy()
                    result=stream.step(action)
                    if all(result.dones): break
                if max_steps is None: assert all(result.dones)
                row=stream.telemetry.finish(stream.env)
                row.update(mode=mode,seed=seed,episode=episode,native_done=all(result.dones),
                           entropy=float(np.mean(entropies)),action_histogram=hist.tolist())
                records.append(row)
                p.write_json(out/'eval_progress.json',dict(step=step,complete=len(records),total=2*episodes,
                    elapsed_seconds=time.monotonic()-start,eval_env_steps=sum(r['length'] for r in records)))
        assert p.tensor_hash(actor.state_dict())==actor_hash
        report=dict(schema=SCHEMA,task=task,step=step,actor_sha256=actor_hash,records=records,
            summary={m:summarize([r for r in records if r['mode']==m]) for m in ('argmax','sample')},
            transition_semantics=SEMANTICS,eval_env_steps=sum(r['length'] for r in records),
            elapsed_seconds=time.monotonic()-start,training_rng_preserved=True,reward_multiplier=multiplier,
            eval_seed_base=seed_base,source_sha256=source_hashes())
        p.write_json(file,report)
        return report
    finally: p.restore_rng(rng,device)


def advantage_diagnostics(trainer,rollout):
    batch=p.stack_rollout(rollout);t=lambda k:torch.as_tensor(batch[k],device=trainer.device)
    a,ret=compute_gae(t('rewards'),trainer._denormalize_values(t('values')),
        trainer._denormalize_values(t('next_values')),t('terminated'),t('active_mask'),
        gamma=.99,gae_lambda=.95,truncated=t('truncated'),episode_end=t('episode_end'))
    a=a[t('active_mask')];ret=ret[t('active_mask')]
    norm=(a-a.mean())/a.std(unbiased=False).clamp_min(1e-6)
    return dict(raw_advantage_mean=float(a.mean()),raw_advantage_std=float(a.std(unbiased=False)),
        normalized_advantage_mean=float(norm.mean()),normalized_advantage_std=float(norm.std(unbiased=False)),
        raw_return_mean=float(ret.mean()),raw_return_std=float(ret.std(unbiased=False)),
        raw_advantage_quantiles=torch.quantile(a,torch.tensor([.1,.5,.9],device=trainer.device)).cpu().tolist())


def update(trainer,rollout,step):
    diag=advantage_diagnostics(trainer,rollout)
    result=old.update(trainer,rollout,step);result.update(diag)
    return result


def source_hashes():
    files=[Path(__file__),CONFIG,p.CONFIG,Path(p.__file__),Path(old.__file__),
           ROOT/'tools/train_forward_final_ppo_20260909.py',
           ROOT/'tools/run_continuous_ctde_training.py',ROOT/'tools/run_small_step_ac_migration.py',
           ROOT/'tools/train_forward_final_single_task_20260915.py']
    files+=list((ROOT/'src/cocap_voradj').rglob('*.py'))
    files+=list((ROOT/'configs/experiments').rglob('*.yaml'))
    return {str(f.relative_to(ROOT)):old.sha(f) for f in sorted(set(files))}


def save(out,step,trainer,stream,launch,rollout,metrics):
    path=out/f'step_{step:06d}.pt'
    assert not path.exists(),'Immutable checkpoint exists'
    payload=dict(schema=SCHEMA,step=step,launch=launch,trainer=trainer.state_dict(),
                 stream_state=stream.__dict__,rollout=rollout,metrics=metrics,rng=p.rng_state(trainer.device))
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp)
    check=torch.load(tmp,map_location='cpu',weights_only=False)
    assert check['schema']==SCHEMA and check['step']==step
    os.replace(tmp,path)
    p.write_json(path.with_suffix('.json'),dict(step=step,sha256=old.sha(path),path=str(path),
        partial_rollout_steps=len(rollout['rewards']),transition_semantics=SEMANTICS))
    return path


def load(path,device):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    assert payload['schema']==SCHEMA
    launch=payload['launch'];assert launch['contract']==contract()
    assert launch['source_sha256']==source_hashes(),'Source changed since checkpoint'
    trainer=p.make_trainer(launch['contract']['scratch'],launch['seed'],device)
    trainer.load_state_dict(payload['trainer'])
    stream=SingleTaskStream.__new__(SingleTaskStream);stream.__dict__.update(payload['stream_state'])
    check_env(stream.env,stream.task_name,trainer.actor)
    p.restore_rng(payload['rng'],trainer.device)
    return trainer,stream,payload


def smoke(task,out,device):
    c=contract();seed=c['spec']['seed'];p.seed_all(seed)
    trainer=p.make_trainer(c['scratch'],seed,device);stream=SingleTaskStream(task,seed)
    launch=make_launch(task,c,device,smoke=True)
    p.write_json(out/'launch.json',launch)
    rollout=p.empty_rollout()
    for _ in range(256): p.append_transition(rollout,collect_transition(trainer,stream)[0])
    metrics=update(trainer,rollout,256)
    partial=p.empty_rollout()
    for _ in range(4):p.append_transition(partial,collect_transition(trainer,stream)[0])
    checkpoint=save(out,260,trainer,stream,launch,partial,metrics)
    expected=[collect_transition(trainer,stream)[0] for _ in range(4)]
    expected_rng=p.rng_state(trainer.device)
    clone,clone_stream,payload=load(checkpoint,device)
    actual=[collect_transition(clone,clone_stream)[0] for _ in range(4)]
    p.assert_tree_equal(expected,actual);p.assert_tree_equal(expected_rng,p.rng_state(clone.device))
    for row in expected:p.append_transition(partial,row)
    rng=p.rng_state(trainer.device);m1=update(trainer,partial,264)
    p.restore_rng(rng,clone.device);m2=update(clone,partial,264)
    p.assert_tree_equal(m1,m2);p.assert_tree_equal(trainer.state_dict(),clone.state_dict())
    rng=p.rng_state(clone.device);evaluate(clone.actor,task,out,264,episodes=1,max_steps=4)
    p.assert_tree_equal(rng,p.rng_state(clone.device))
    p.write_json(out/'report.json',dict(status='SMOKE_PASS',task=task,teacher_dependency=0,
        full_rollout256_update=True,resume_transition_rng_optimizer_bit_exact=True,
        eval_rng_preserved=True,metrics=metrics,resumed_update=m2))


def make_launch(kind,c,device,smoke=False):
    return dict(schema=SCHEMA,kind=kind,seed=c['spec']['seed'],contract=c,
        total_steps=c['spec']['budgets'][kind],transition_semantics=SEMANTICS,
        source_sha256=source_hashes(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        pid=os.getpid(),device=device,gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
        teacher_dependency=0,smoke=smoke,metric_early_stop=False)


def train(kind,out,device,resume=None):
    c=contract();task='coverage' if kind=='coverage_r2' else kind
    multiplier=2. if kind=='coverage_r2' else 1.
    budget=c['spec']['budgets'][kind];cadence=c['spec']['checkpoint_interval']
    if resume:
        trainer,stream,payload=load(resume,device)
        assert payload['launch']['kind']==kind and Path(resume).resolve().parent==out.resolve()
        step=payload['step'];rollout=payload['rollout'];metrics=payload['metrics'];launch=payload['launch']
    else:
        p.seed_all(c['spec']['seed']);trainer=p.make_trainer(c['scratch'],c['spec']['seed'],device)
        stream=SingleTaskStream(task,c['spec']['seed'],multiplier)
        launch=make_launch(kind,c,device)
        launch.update(runtime=check_env(stream.env,task,trainer.actor),
            actor_initial_sha256=p.tensor_hash(trainer.actor.state_dict()),
            critic_initial_sha256=p.tensor_hash(trainer.value.state_dict()),
            resolved_environment=task_config(task),reward_multiplier=multiplier)
        p.write_json(out/'launch.json',launch)
        step=0;rollout=p.empty_rollout();metrics={};save(out,0,trainer,stream,launch,rollout,metrics)
    start=time.monotonic();start_step=step
    def progress(status):
        elapsed=time.monotonic()-start;speed=(step-start_step)/elapsed
        p.write_json(out/'progress.json',dict(status=status,pid=os.getpid(),kind=kind,step=step,
            budget=budget,steps_per_second=speed,elapsed_seconds=elapsed,
            eta_seconds=(budget-step)/speed if speed else None,last_update=metrics,
            partial_rollout_steps=len(rollout['rewards']),training=stream.summary()))
    progress('checkpoint_evaluation')
    if step%cadence==0:evaluate(trainer.actor,task,out,step,multiplier=multiplier)
    while step<budget:
        stream.set_clock(step)
        row,episode=collect_transition(trainer,stream);old.finite_tree(row)
        p.append_transition(rollout,row);step+=1
        if episode:
            with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,**episode),allow_nan=False)+'\n')
        if len(rollout['rewards'])==c['scratch']['rollout_length']:
            metrics=update(trainer,rollout,step);rollout=p.empty_rollout()
            with (out/'learning.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
            print(json.dumps(dict(step=step,**{k:metrics[k] for k in ('entropy','explained_variance','exact_full_batch_kl_old_new')})),flush=True)
        if step%100==0:progress('training')
        if step%cadence==0:
            save(out,step,trainer,stream,launch,rollout,metrics)
            progress('checkpoint_evaluation')
            evaluate(trainer.actor,task,out,step,multiplier=multiplier)
            # Intermediate reviews are telemetry, never budget stop gates.
            p.write_json(out/f'review_{step:06d}.json',dict(step=step,continue_to=budget,
                review_only=True,metrics=metrics,training=stream.summary()))
    progress('COMPLETE_BUDGET')
    p.write_json(out/'report.json',dict(status='COMPLETE_BUDGET',step=step,task=task,kind=kind,
        teacher_dependency=0,metric_early_stop=False,final_checkpoint=str(out/f'step_{step:06d}.pt')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['capture','coverage','coverage_r2'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--r2-gate',type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if args.smoke and args.resume:parser.error('Smoke and resume are separate modes')
    if args.task=='coverage_r2':
        if not args.r2_gate:parser.error('R2 requires the attribution and non-null intervention gate')
        gate=json.loads(args.r2_gate.read_text())
        assert gate['decision']=='START_R2' and gate['baseline_no_learning'] and gate['representation_sufficient']
        assert gate['reward_scale_candidate'] and not gate['effectively_null']
        assert gate['common_contract_sha256']==old.sha(CONFIG)
        assert gate['source_sha256']==source_hashes()
        assert gate['baseline_checkpoint_sha256']==old.sha(args.r2_gate.parent.parent/'coverage'/'step_200000.pt')
    if not args.resume:
        if out.exists():parser.error('Fresh output required')
        out.mkdir(parents=True)
    torch.set_num_threads(1)
    p.forbid_teacher_dependencies(out)
    try:
        if args.smoke:smoke(args.task,out,args.device)
        else:train(args.task,out,args.device,args.resume)
    except BaseException:
        p.write_json(out/'failure.json',dict(status='STOP_IMPLEMENTATION',pid=os.getpid(),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
