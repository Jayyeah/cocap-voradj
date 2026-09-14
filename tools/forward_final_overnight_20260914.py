"""Bounded EXPLORATORY_NON_GATE orchestration; production PPO is unchanged."""
from __future__ import annotations
import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from collections import Counter
import numpy as np
import torch
from tools import preflight_forward_final_scratch_20260914 as p
from tools import train_forward_final_ppo_20260909 as production
from cocap_voradj.training.small_step_ac import compute_gae

ROOT = p.ROOT
SCHEMA = 'forward-final-overnight-diagnostic-v1'
LABEL = 'EXPLORATORY_NON_GATE'
SEMANTICS = 'terminal-priority-truncation-bootstrap-weighted-ce-v2'
BUDGETS = {'bc_ppo': 10000, 'scratch': 100000}
CADENCE = {'bc_ppo': [0, 5000, 10000], 'scratch': [0, 25000, 50000, 75000, 100000]}
SEEDS = {'bc_ppo': 2026097101, 'scratch': 2026091401}
EVAL_BASE = 2026092401


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sources():
    names = [Path(__file__), Path(p.__file__), Path(production.__file__), p.CONFIG,
             ROOT/'tools/train_forward_final_scratch_mappo_20260914.py',
             ROOT/'tools/train_forward_final_ppo_diagnostic_20260914.py']
    names += list((ROOT/'src/cocap_voradj').rglob('*.py'))
    names += list((ROOT/'configs/experiments/forward_final_mappo_20260908').glob('*.yaml'))
    return {str(q.relative_to(ROOT)): sha(q) for q in sorted(set(q.resolve() for q in names))}


def finite_tree(value):
    if isinstance(value, dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value, (list, tuple)):
        for v in value: finite_tree(v)
    elif torch.is_tensor(value):
        assert bool(torch.isfinite(value).all()), 'Nonfinite tensor'
    elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        assert np.isfinite(value).all(), 'Nonfinite array'
    elif isinstance(value, (float, np.floating)):
        assert np.isfinite(value), 'Nonfinite scalar'


class RecordedStream(p.FinalMissionStream):
    """Read-only telemetry around the exact native stream (no additional draws)."""
    def __init__(self, seed, out):
        self.telemetry = Counter()
        self.phase_steps = Counter()
        self.episode_metrics = []
        self.current = Counter()
        super().__init__(seed, out)

    def step(self, indices):
        phase = 'pure_coverage' if self.task == 'voradj_coverage' else (
            'post_capture' if self.current['capture_events'] else 'pre_capture')
        active = [i for i, o in enumerate(self.observations) if o is not None]
        outcome = super().step(indices)
        self.phase_steps[phase] += 1
        self.telemetry['env_steps'] += 1
        self.telemetry['active_rows'] += len(active)
        self.current['length'] += 1
        events = list(self.env.last_capture_events)
        sizes = [sum(np.hypot(q.x-e.x, q.y-e.y) <= 8. for q in self.env.pursuers if not q.deactivated)
                 for e in self.env.evaders if not e.deactivated]
        sizes += [len(e['participants']) for e in events]
        size = max(sizes, default=0)
        for k in (2, 3):
            self.telemetry[f'ring{k}_steps'] += int(size >= k)
            self.current[f'ring{k}_steps'] += int(size >= k)
        self.current['ring3_run'] = self.current['ring3_run']+1 if size >= 3 else 0
        self.current['ring3_max_hold'] = max(self.current['ring3_max_hold'], self.current['ring3_run'])
        for e in events:
            self.current['capture_events'] += 1
            self.telemetry[e['capture_type']+'_capture_events'] += 1
        for i in active:
            meta = outcome.infos[i]['replay_metadata']
            self.telemetry['support_rows'] += int(meta['support_candidate'])
        for event in self.env.last_collision_events:
            self.current['collision'] = 1
            if event.get('type') == 'boundary': self.current['boundary'] = 1
        return outcome

    def finish(self):
        record = self.env.episode_record(task='mix' if self.task == 'voradj' else 'coverage')
        extra = dict(self.current)
        extra.update(ce_rms=p.stats([record['coverage_ce_center_rms']])['mean'],
                     area_cv=p.stats([record['coverage_strict_area_cv']])['mean'])
        row = super().finish()
        row.update(extra)
        self.episode_metrics.append(row)
        self.current = Counter()
        return row

    def summary(self):
        return dict(cumulative=dict(self.telemetry), phase_steps=dict(self.phase_steps),
                    reset_source_distribution=dict(self.reset_counts),
                    real_capture_snapshots=len(self.recovery_init_pool),
                    episodes_completed=len(self.episode_metrics),
                    current_episode=dict(self.current),
                    scenes={task: dict(
                        episodes=sum(r['task']==task for r in self.episode_metrics),
                        capture=sum(r['captured'] for r in self.episode_metrics if r['task']==task),
                        ce_success=sum(r['ce_success'] for r in self.episode_metrics if r['task']==task),
                        collision=sum(r['collision'] for r in self.episode_metrics if r['task']==task),
                        boundary=sum(r.get('boundary',0) for r in self.episode_metrics if r['task']==task),
                        episode_length=p.stats([r['length'] for r in self.episode_metrics if r['task']==task]),
                        area_cv=p.stats([r['area_cv'] for r in self.episode_metrics if r['task']==task and r['area_cv'] is not None]),
                        ce_rms=p.stats([r['ce_rms'] for r in self.episode_metrics if r['task']==task and r['ce_rms'] is not None]),
                        ring3_max_hold=p.stats([r.get('ring3_max_hold',0) for r in self.episode_metrics if r['task']==task]))
                        for task in ('voradj','voradj_coverage')})


def factory(kind, c, device):
    seed = SEEDS[kind]
    trainer = p.make_trainer(c, seed, device) if kind == 'scratch' else production.make_trainer(device, seed)
    assert dataclasses.asdict(trainer.config) == c['ppo']
    assert not trainer.actor_optimizer.state and not trainer.value_optimizer.state
    assert trainer.update_count == 0 and float(trainer.value_norm.mean)==0 and float(trainer.value_norm.std)==1
    return trainer


def update(trainer, rollout, step):
    batch = p.stack_rollout(rollout)
    finite_tree(batch)
    error = trainer.assert_behavior_log_probs(batch)
    t = lambda k: torch.as_tensor(batch[k], device=trainer.device)
    advantages, returns = compute_gae(t('rewards'), trainer._denormalize_values(t('values')),
        trainer._denormalize_values(t('next_values')), t('terminated'), t('active_mask'),
        gamma=trainer.config.gamma, gae_lambda=trainer.config.gae_lambda,
        truncated=t('truncated'), episode_end=t('episode_end'))
    finite_tree([advantages, returns])
    before = p.rollout_log_probs(trainer, batch)
    metrics = trainer.update(batch, categorical=True)
    metrics.update(p.exact_update_diagnostics(before, p.rollout_log_probs(trainer, batch)))
    metrics.update(step=step, zero_update_max_log_prob_error=error,
                   raw_gae_min=float(advantages.min()), raw_gae_max=float(advantages.max()),
                   raw_return_min=float(returns.min()), raw_return_max=float(returns.max()),
                   rollout_steps=len(rollout['rewards']))
    finite_tree(metrics)
    finite_tree(trainer.state_dict())
    return metrics


def save(out, step, trainer, stream, launch, rollout, metrics):
    path = out/f'step_{step:06d}.pt'
    if path.exists(): raise ValueError('Immutable checkpoint already exists')
    payload = dict(schema=SCHEMA, classification=LABEL, kind=launch['kind'], step=step,
        transition_semantics=SEMANTICS, policy_contract=production.POLICY_CONTRACT,
        launch=launch, trainer=trainer.state_dict(), stream_state=stream.__dict__,
        rollout=rollout, metrics=metrics, rng=p.rng_state(trainer.device))
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    check = torch.load(tmp, map_location='cpu', weights_only=False)
    assert check['step']==step and check['transition_semantics']==SEMANTICS
    os.replace(tmp, path)
    manifest = dict(path=str(path), sha256=sha(path), step=step, classification=LABEL,
                    partial_rollout_steps=len(rollout['rewards']))
    p.write_json(out/f'step_{step:06d}.json', manifest)
    return path


def load(path, kind, c, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['schema']==SCHEMA and payload['classification']==LABEL
    assert payload['kind']==kind and payload['transition_semantics']==SEMANTICS
    assert payload['policy_contract']==production.POLICY_CONTRACT
    assert payload['launch']['source_sha256']==sources(), 'Resume source contract changed'
    assert payload['launch']['config']==c
    assert payload['launch']['total_steps']==BUDGETS[kind]
    assert payload['launch']['seed']==SEEDS[kind]
    trainer = factory(kind, c, device)
    trainer.load_state_dict(payload['trainer'])
    stream = RecordedStream.__new__(RecordedStream)
    stream.__dict__.update(payload['stream_state'])
    assert all(e.transition_semantics==SEMANTICS for e in stream.envs.values())
    p.restore_rng(payload['rng'], trainer.device)
    return trainer, stream, payload


def evaluate(trainer, out, step, *, episodes=20, max_steps=None):
    path = out/f'eval_step_{step:06d}.json'
    if path.exists():
        report=json.loads(path.read_text())
        assert report['actor_sha256']==p.tensor_hash(trainer.actor.state_dict())
        return report
    saved_rng = p.rng_state(trainer.device)
    actor_hash = p.tensor_hash(trainer.actor.state_dict())
    records=[]
    started=time.monotonic()
    try:
        for mode in ('argmax','sample'):
            for scene in ('mixed','coverage'):
                for i in range(episodes):
                    seed=EVAL_BASE+i+(100000 if scene=='coverage' else 0)
                    p.seed_all(seed)
                    env, obs=p.make_env(scene, seed)
                    assert env.transition_semantics==SEMANTICS
                    env.reward_cfg['coverage_ce_speed_weight']=.0005
                    host=type('APFHost',(),{})()
                    host.envs={'eval':env}
                    host.apf_agents={'eval':[p.ApfAgent(e.a,e.w) for e in env.evaders]}
                    discounted=0.; undiscounted=0.; count=0
                    def step_fn(actions):
                        nonlocal discounted, undiscounted, count
                        p.set_global_config(env.config)
                        outcome=env.step(actions,p.CoCapTrainer._evader_actions(host,'eval'))
                        reward=float(np.mean(outcome.rewards))
                        discounted += .99**count*reward
                        undiscounted += reward
                        count+=1
                        return outcome
                    row=p.observe_episode(trainer.actor,env,obs,step_fn,scene,mode,max_steps=max_steps)
                    if max_steps is None: assert row['native_done']
                    row.update(seed=seed, episode=i, discounted_return=discounted,
                               undiscounted_return=undiscounted,
                               reset_source='independent_scene_environment_default',
                               post_ce_success=bool(scene=='mixed' and row['captured'] and row['ce_success']))
                    records.append(row)
                    p.write_json(out/'eval_progress.json',dict(step=step,complete=len(records),total=4*episodes,
                        eval_env_steps=sum(r['length'] for r in records),elapsed_seconds=time.monotonic()-started))
        assert actor_hash==p.tensor_hash(trainer.actor.state_dict())
        report=dict(schema=SCHEMA,classification=LABEL,step=step,actor_sha256=actor_hash,
                    transition_semantics=SEMANTICS,records=records,
                    summary={mode:p.summarize([r for r in records if r['mode']==mode]) for mode in ('argmax','sample')},
                    eval_env_steps=sum(r['length'] for r in records),elapsed_seconds=time.monotonic()-started,
                    seed_base=EVAL_BASE,ce_speed_weight=.0005,
                    reset_protocol='independent fixed scene seeds; no training recovery pool; same initial states across checkpoints',
                    training_rng_preserved=True,formal_training='HOLD')
        p.write_json(path,report)
        return report
    finally:
        p.restore_rng(saved_rng,trainer.device)


def matched(base, current):
    result={}
    for mode in ('argmax','sample'):
        for scene in ('mixed','coverage'):
            a=[r for r in base['records'] if r['mode']==mode and r['scene']==scene]
            b=[r for r in current['records'] if r['mode']==mode and r['scene']==scene]
            assert len(a)==len(b)>0
            for x,y in zip(a,b):
                assert x['seed']==y['seed'] and x['initial_state_fingerprint']==y['initial_state_fingerprint'], 'Unmatched eval states'
            pairs=[(x,y) for x,y in zip(a,b) if x['safe_complete'] and y['safe_complete']]
            row={'episodes':len(a),'common_safe_n':len(pairs)}
            for key in ('safe_complete','captured','ce_success','collision','post_ce_success','discounted_return',
                        'ring2_steps','ring3_steps','ring3_max_hold'):
                row[key]={'baseline':float(np.mean([r[key] for r in a])),
                          'current':float(np.mean([r[key] for r in b])),
                          'delta':float(np.mean([y[key]-x[key] for x,y in zip(a,b)]))}
            for key in ('mission_seconds','recovery_seconds'):
                common=[(x[key],y[key]) for x,y in pairs if x[key] is not None and y[key] is not None]
                av=p.stats([x for x,y in common]); bv=p.stats([y for x,y in common])
                row['common_safe_'+key]={'baseline':av,'current':bv,
                    'relative_change':bv['mean']/av['mean']-1 if common and av['mean'] else None,
                    'paired_delta':p.stats([y-x for x,y in common])}
            dr=np.asarray([y['discounted_return']-x['discounted_return'] for x,y in zip(a,b)])
            row['return_negative_pair_fraction']=float(np.mean(dr<0))
            row['return_median_delta']=float(np.median(dr))
            row['phase_occupancy']={side:{k:sum(r['phase_steps'][k] for r in rows) for k in p.PHASES}
                                    for side,rows in [('baseline',a),('current',b)]}
            row['reset_sources']={side:dict(Counter(r['reset_source'] for r in rows)) for side,rows in [('baseline',a),('current',b)]}
            result[mode+'/'+scene]=row
    return result


def bc_decision(delta):
    failures=[]; missing=[]
    for group,d in delta.items():
        if d['safe_complete']['delta'] < -.05000001: failures.append(group+': safe decline >5pp')
        if d['collision']['delta'] > .05000001: failures.append(group+': collision rise >5pp')
        time_change=d['common_safe_mission_seconds']['relative_change']
        if time_change is None or d['common_safe_n']<10: missing.append(group+': common-safe n<10')
        elif time_change > .05000001: failures.append(group+': common-safe mission >5% slower')
        threshold=.05*max(abs(d['discounted_return']['baseline']),1.)
        if d['discounted_return']['delta'] < -threshold and d['return_negative_pair_fraction']>=.6 and d['return_median_delta']<0:
            failures.append(group+': consistently worse return')
        if group.endswith('/mixed'):
            recovery=d['common_safe_recovery_seconds']['relative_change']
            if d['post_ce_success']['delta']<-.05000001 or (recovery is not None and recovery>.10):
                failures.append(group+': post-recovery collapse screen')
    return dict(classification=LABEL,decision='REGRESSED' if failures else ('INCONCLUSIVE' if missing else 'SURVIVED'),
                continue_to_10k=not failures and not missing,failures=failures,missing=missing,
                scope='overnight operational screen only; not statistical significance or formal Gate PASS')


def scratch_decision(reports):
    # Predeclared sample-policy trend screen; report both modes separately.
    base=reports[0]['summary']['sample']; signals=[]
    for report in reports[1:]:
        s=report['summary']['sample']; m=s['mixed']; b=base['mixed']; cv=s['coverage']; bc=base['coverage']
        n=m['episodes']
        predecessor=(m['ring2_episode_count']-b['ring2_episode_count']>=.15*n and
                     m['ring3_episode_count']-b['ring3_episode_count']>=.10*n and
                     m['ring3_max_hold_steps']['p50']>=b['ring3_max_hold_steps']['p50'])
        capture_post=m['captured_count']>0 or m['phase_step_counts']['post_capture']>0
        support=m['phase_rewards']['support']['active_row_fraction']
        base_support=b['phase_rewards']['support']['active_row_fraction']
        coverage_ok=(cv['ce_success_count']>=bc['ce_success_count']-1 and
            cv['collision_count']<=bc['collision_count']+1 and
            cv['ce_rms']['p50']<=1.15*bc['ce_rms']['p50'] and
            cv['area_cv']['p50']<=1.15*bc['area_cv']['p50'])
        stable=coverage_ok and support>=.75*base_support and m['collision_count']<=b['collision_count']+1
        local=(m['ring2_episode_count']>b['ring2_episode_count'] or
               m['ring3_episode_count']>b['ring3_episode_count'] or
               cv['ce_success_count']>bc['ce_success_count'] or
               (cv['ce_rms']['p50']<.85*bc['ce_rms']['p50'] and cv['area_cv']['p50']<.85*bc['area_cv']['p50']))
        signals.append(dict(step=report['step'],predecessor_improved=predecessor,capture_or_post=capture_post,
                            coverage_support_stable=stable,local_improvement=local))
    sustained=any(a['predecessor_improved'] and b['predecessor_improved'] and a['coverage_support_stable'] and b['coverage_support_stable']
                  for a,b in zip(signals,signals[1:]))
    positive=sustained or any(s['capture_or_post'] and s['coverage_support_stable'] for s in signals)
    weak=any(s['local_improvement'] or s['capture_or_post'] or s['predecessor_improved'] for s in signals)
    return dict(classification=LABEL,decision='POSITIVE_LEARNABILITY_SIGNAL' if positive else (
        'WEAK_OR_EARLY_SIGNAL' if weak else 'NEGATIVE_DIAGNOSTIC_NOT_ALGORITHM_FAILURE'),signals=signals,
        scope='single seed bounded diagnostic; P2/P3 and phase visitation unresolved; no automatic 200k or more seeds')


def smoke(kind,c,out,device,launch):
    p.seed_all(SEEDS[kind]); trainer=factory(kind,c,device)
    stream=RecordedStream(SEEDS[kind],out/'stream')
    p.validate_stream(c,stream); p.assert_runtime(stream.env,trainer.actor,categorical=True)
    metrics=[]
    for j in range(2):
        rollout=p.empty_rollout()
        for _ in range(16): p.append_transition(rollout,p.collect_transition(trainer,stream)[0])
        metrics.append(update(trainer,rollout,(j+1)*16))
    partial=p.empty_rollout()
    for _ in range(4): p.append_transition(partial,p.collect_transition(trainer,stream)[0])
    path=save(out,36,trainer,stream,launch,partial,metrics[-1])
    expected=[p.collect_transition(trainer,stream)[0] for _ in range(4)]
    expected_rng=p.rng_state(trainer.device)
    clone,stream2,payload=load(path,kind,c,device)
    actual=[p.collect_transition(clone,stream2)[0] for _ in range(4)]
    p.assert_tree_equal(expected,actual); p.assert_tree_equal(expected_rng,p.rng_state(clone.device))
    p.assert_tree_equal(stream.summary(),stream2.summary())
    p.assert_tree_equal(payload['rollout'],partial)
    for row in expected: p.append_transition(partial,row)
    rng=p.rng_state(trainer.device)
    m1=update(trainer,partial,40)
    p.restore_rng(rng,clone.device)
    m2=update(clone,partial,40)
    p.assert_tree_equal(m1,m2); p.assert_tree_equal(trainer.state_dict(),clone.state_dict())
    rng=p.rng_state(clone.device)
    evaluate(clone,out,40,episodes=1,max_steps=4)
    p.assert_tree_equal(rng,p.rng_state(clone.device))
    report=dict(status='SMOKE_PASS',classification=LABEL,formal_training='HOLD',kind=kind,device=device,
        resume_transition_rng_optimizer_bit_exact=True,eval_rng_preserved=True,
        zero_update_log_prob_contract=True,finite=True,teacher_dependency=0 if kind=='scratch' else 'BC_INITIALIZATION_ONLY',
        metrics=metrics,resumed_update_metrics=m1,training_steps=40,replay_steps=4)
    p.write_json(out/'report.json',report)
    return report


def run(kind):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args(); out=args.output.resolve()
    if args.resume and args.smoke: parser.error('Smoke and resume are separate modes')
    if args.resume:
        if args.resume.resolve().parent != out: parser.error('Resume only this same run directory')
    elif out.exists(): parser.error('Fresh output directory required')
    else: out.mkdir(parents=True)
    torch.set_num_threads(1)
    c=p.load_contract(); assert c['transition_semantics']==SEMANTICS
    assert c['future_gate_design_only']['seed']==SEEDS['scratch']
    assert c['future_gate_design_only']['baseline_seed_base']==EVAL_BASE
    assert c['future_gate_design_only']['eval_episodes_per_scene_per_mode']==20
    if kind=='scratch': p.forbid_teacher_dependencies(out)
    launch=dict(schema=SCHEMA,classification=LABEL,formal_training='HOLD',kind=kind,seed=SEEDS[kind],
        total_steps=BUDGETS[kind],checkpoints=CADENCE[kind],config=c,transition_semantics=SEMANTICS,
        source_sha256=sources(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        pid=os.getpid(),gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),device=args.device,smoke=args.smoke,
        teacher_dependency=0 if kind=='scratch' else 'BC_INITIALIZATION_ONLY',
        operational_rules=dict(safe_decline_pp=5,collision_rise_pp=5,common_safe_time_max_regression=.05,
            return_rule='mean decline >5% of max(abs(baseline mean),1), >=60% pairs negative and median negative',
            post_recovery_rule='post CE decline >5pp or common-safe recovery mean >10% slower',
            common_safe_min_pairs=10,all_four_scene_mode_groups_required=True),
        no_automatic_extension=True)
    try:
        if args.smoke:
            p.write_json(out/'launch.json',launch)
            smoke(kind,c,out,args.device,launch)
            return
        if args.resume:
            trainer,stream,payload=load(args.resume,kind,c,args.device)
            launch=payload['launch']; step=payload['step']; rollout=payload['rollout']; metrics=payload['metrics']
            assert step<BUDGETS[kind], 'Completed budget cannot resume'
            p.write_json(out/'resume.json',dict(pid=os.getpid(),checkpoint=str(args.resume),step=step))
        else:
            p.seed_all(SEEDS[kind]); trainer=factory(kind,c,args.device)
            stream=RecordedStream(SEEDS[kind],out/'stream')
            launch['runtime']=p.validate_stream(c,stream)
            launch['zero_update_probe']=p.assert_runtime(stream.env,trainer.actor,categorical=True)
            launch['actor_initial_sha256']=p.tensor_hash(trainer.actor.state_dict())
            launch['critic_initial_sha256']=p.tensor_hash(trainer.value.state_dict())
            p.write_json(out/'launch.json',launch)
            p.write_json(out/'provenance.json',dict(teacher_dependency=launch['teacher_dependency'],
                initialization='random' if kind=='scratch' else 'canonical BC actor; fresh critic Adam ValueNorm',
                external_artifact_reads_blocked=kind=='scratch',formal_training='HOLD'))
            step=0; rollout=p.empty_rollout(); metrics={}
            save(out,step,trainer,stream,launch,rollout,metrics)
        started=time.monotonic(); start_step=step
        def progress(status):
            elapsed=time.monotonic()-started; speed=(step-start_step)/elapsed
            p.write_json(out/'progress.json',dict(status=status,classification=LABEL,pid=os.getpid(),step=step,
                total_steps=BUDGETS[kind],steps_per_second=speed,elapsed_seconds=elapsed,
                eta_seconds=(BUDGETS[kind]-step)/speed if speed else None,last_update=metrics,
                partial_rollout_steps=len(rollout['rewards']),training=stream.summary()))
        progress('step0_evaluation' if step==0 else 'resumed')
        # Resume at a checkpoint repeats only missing eval, never any saved update.
        if step in CADENCE[kind]:
            current=evaluate(trainer,out,step)
            if kind=='bc_ppo' and step==5000:
                delta=matched(json.loads((out/'eval_step_000000.json').read_text()),current)
                decision=bc_decision(delta); p.write_json(out/'gate_005000.json',dict(**decision,matched_delta=delta))
                if not decision['continue_to_10k']:
                    p.write_json(out/'report.json',dict(status='STOP_BC_PPO_AT_5K',step=step,**decision)); progress('STOP_BC_PPO_AT_5K'); return
        while step < BUDGETS[kind]:
            stream.set_clock(step)
            row,episode=p.collect_transition(trainer,stream); finite_tree(row)
            p.append_transition(rollout,row); step+=1
            if episode:
                with (out/'episodes.jsonl').open('a') as f: f.write(json.dumps(dict(step=step,**episode))+'\n')
            if len(rollout['rewards'])==c['rollout_length']:
                metrics=update(trainer,rollout,step); rollout=p.empty_rollout()
                with (out/'learning.jsonl').open('a') as f: f.write(json.dumps(metrics,allow_nan=False)+'\n')
                print(json.dumps(metrics,allow_nan=False),flush=True)
            if step%100==0: progress('training')
            if step in CADENCE[kind]:
                save(out,step,trainer,stream,launch,rollout,metrics)
                p.write_json(out/f'training_step_{step:06d}.json',dict(step=step,metrics=metrics,training=stream.summary()))
                progress('checkpoint_evaluation')
                current=evaluate(trainer,out,step)
                base=json.loads((out/'eval_step_000000.json').read_text())
                delta=matched(base,current)
                p.write_json(out/f'matched_step_{step:06d}.json',delta)
                if kind=='bc_ppo':
                    decision=bc_decision(delta)
                    p.write_json(out/f'gate_{step:06d}.json',dict(**decision,matched_delta=delta))
                    if step==5000 and not decision['continue_to_10k']:
                        p.write_json(out/'report.json',dict(status='STOP_BC_PPO_AT_5K',step=step,**decision)); progress('STOP_BC_PPO_AT_5K'); return
        if kind=='scratch':
            reports=[json.loads((out/f'eval_step_{s:06d}.json').read_text()) for s in CADENCE[kind]]
            decision=scratch_decision(reports)
        p.write_json(out/'report.json',dict(status='COMPLETE_BUDGET_STOP',step=step,**decision,formal_training='HOLD'))
        progress('COMPLETE_BUDGET_STOP')
    except BaseException:
        p.write_json(out/'failure.json',dict(status='STOP_IMPLEMENTATION',classification=LABEL,traceback=traceback.format_exc(),pid=os.getpid()))
        p.write_json(out/'progress.json',dict(status='STOP_IMPLEMENTATION',classification=LABEL,pid=os.getpid(),step=locals().get('step',0)))
        raise
