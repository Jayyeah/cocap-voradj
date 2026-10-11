#!/usr/bin/env python3
"""Fixed1M N3/R3/V3 capability runs, periodic evaluation, bounded local closeout.

No performance early-stop, tuning, automatic retry, budget extension or next
batch launch. Each process owns one physical GPU and one run directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import numpy as np
import torch

from terl_mappo.batch02.runtime import sources as frozen_sources
from terl_mappo.batch02_native.runtime import NativeAdapter as CapabilityAdapter, build, collect, sources, save_checkpoint, load_checkpoint
from terl_mappo.batch02_native.evaluation import evaluate, selection_key, EVALUATOR_VERSION
from terl_mappo.batch02_native.reporting import refresh, closeout
from terl_mappo.run import atomic_json, file_hash

CONFIGS = ROOT / 'configs/experiments/terl_mappo_batch02_native_20261011'
CENTRAL = Path('/home/yjq/rl/CoCap1/ac-master-dag-20260921')


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def append(path, row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def record_checkpoint(run, trainer, adapter, config, step, agent_steps, optimizer_steps, entries):
    path = run / f'checkpoints/step_{step:09d}.pt'
    digest = save_checkpoint(path, trainer, adapter, config, step, agent_steps, optimizer_steps)
    entry = dict(decisions=step, path=str(path), sha256=digest, created_at=now())
    entries.append(entry)
    atomic_json(run / 'checkpoint_list.json', entries)
    atomic_json(run / 'last_checkpoint.json', entry)
    return path


def run(args):
    torch.set_num_threads(1)
    config_path = CONFIGS / f'{args.arm.lower()}.json'
    config = json.loads(config_path.read_text())
    if config['budget'] != 1_000_000 or config['checkpoint_interval'] != 25_000 or config['arm'] != args.arm:
        raise ValueError('fixed user-authorized budget/config mismatch')
    if args.device.startswith('cuda') and os.environ.get('CUDA_VISIBLE_DEVICES') != str(config['physical_gpu']):
        raise ValueError('physical GPU binding mismatch')
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True)
    if dirty:
        raise ValueError('tracked experiment sources/configs must be committed before launch')
    freeze = json.loads((CENTRAL / 'artifacts/2026-10-11_batch02_e1_independent/freeze.json').read_text())
    if not freeze['E1_PASS'] or freeze['frozen_bridge_SHA'] != config['frozen_E1_SHA'] or frozen_sources() != freeze['source_manifest']:
        raise ValueError('frozen E1 mismatch')
    run = args.output_root.resolve() / args.arm
    if run.exists() and not args.resume:
        raise ValueError('existing run is immutable; explicit resume required')
    run.mkdir(parents=True, exist_ok=True)
    locked_sources = sources()
    atomic_json(run / 'progress.json', dict(status='INITIALIZING', arm=args.arm, pid=os.getpid(),
                experiment_SHA=sha, updated_at=now(), classification='PROVISIONAL'))
    adapter = CapabilityAdapter(config['learner']['seed'], args.arm)
    if adapter.resolved_hash != config['task_reference']['resolved_hash'] or adapter.resolved_hash != freeze['config_hash']:
        raise ValueError('successful CoCap resolved configuration changed')
    trainer = build(config, adapter, args.device)
    step = agent_steps = optimizer_steps = 0
    entries = []
    if args.resume:
        payload = load_checkpoint(args.resume, trainer, adapter, config)
        step, agent_steps, optimizer_steps = (payload[key] for key in ('decisions', 'agent_transitions', 'optimizer_steps'))
        entries = json.loads((run / 'checkpoint_list.json').read_text())
        manifest = json.loads((run / 'manifest.json').read_text())
        if manifest['experiment_SHA'] != sha:
            raise ValueError('experiment SHA changed during resume')
    else:
        manifest = dict(arm=args.arm, name=config['name'], parent_SHA=config['parent_SHA'],
                        frozen_E1_SHA=config['frozen_E1_SHA'], experiment_SHA=sha,
                        config=config, config_sha256=file_hash(config_path), sources=locked_sources,
                        evaluator_version=EVALUATOR_VERSION,
                        evaluator_sha256=file_hash(ROOT / 'src/terl_mappo/batch02_native/evaluation.py'),
                        seed_manifest=config['seed_manifest'], checkpoint_directory=str(run / 'checkpoints'),
                        initialization='scratch; fresh actor/V/Adam/ValueNorm/env/RNG; no IQN weights',
                        task_contract=adapter.contract(), initial_fingerprint=adapter.fingerprint(),
                        started_at=now(), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), pid=os.getpid(),
                        classification='PROVISIONAL', independent_validation_of_E1=True,
                        counter_units=dict(decisions='joint environment decisions', agent_transitions='active policy rows'),
                        relative_E1_delta=config['relative_E1_delta'],
                        relative_successful_IQN_delta=config['relative_successful_IQN_delta'])
        atomic_json(run / 'manifest.json', manifest)
        record_checkpoint(run, trainer, adapter, config, 0, 0, 0, entries)
    started, initial_step = time.monotonic(), step
    phase = 'initial_evaluation'
    try:
        current_checkpoint = Path(entries[-1]['path'])
        if step % config['checkpoint_interval'] == 0:
            atomic_json(run / 'progress.json', dict(status='EVALUATING', phase=phase, decisions=step,
                        budget=config['budget'], arm=args.arm, pid=os.getpid(), updated_at=now(), classification='PROVISIONAL'))
            evaluate(current_checkpoint, run / f'evaluations/screen_{step:09d}.json', config, 'screen', args.device)
            refresh(run)
        while step < config['budget']:
            phase = 'training'
            boundary = ((step // config['checkpoint_interval']) + 1) * config['checkpoint_interval']
            # Same exact-boundary rollout policy as the original TERL runner.
            count = min(config['learner']['rollout_length'], config['budget'] - step, boundary - step)
            batch, episodes = collect(trainer, adapter, count)
            metrics = trainer.update(batch, categorical=True)
            finite = all(np.isfinite(v) for v in metrics.values()) and all(torch.isfinite(p).all().item()
                         for p in list(trainer.actor.parameters()) + list(trainer.value.parameters()))
            grad_finite = all(p.grad is None or torch.isfinite(p.grad).all().item()
                             for p in list(trainer.actor.parameters()) + list(trainer.value.parameters()))
            if not finite or not grad_finite:
                raise FloatingPointError('nonfinite PPO metric/parameter/gradient')
            step += count
            agent_steps += int(batch['active_mask'].sum())
            optimizer_steps += int(metrics['minibatch_updates'])
            active_rewards = batch['rewards'][batch['active_mask']]
            row = dict(decisions=step, active_agent_transitions=agent_steps, optimizer_steps=optimizer_steps,
                       reward_mean_active=float(active_rewards.mean()) if active_rewards.size else None,
                       training_episodes=len(episodes), training_capture_completions=sum(e['capture_completion'] for e in episodes),
                       training_collision_episodes=sum(e['collision_event'] for e in episodes),
                       training_boundary_episodes=sum(e['boundary_collision_event'] for e in episodes),
                       gradient_finite=grad_finite, parameter_finite=finite,
                       reward_clock=int(adapter.env.total_steps),
                       coverage_ce_speed_weight=float(adapter.env.reward_cfg['coverage_ce_speed_weight']), **metrics)
            row.update(reward_decomposition_total=adapter.ledger_total, reward_decomposition_residual_max=adapter.ledger_residual_max)
            append(run / 'metrics.jsonl', row)
            for episode in episodes:
                append(run / 'episodes.jsonl', dict(recorded_at_decisions=step, **episode))
            speed = (step - initial_step) / max(time.monotonic() - started, 1e-9)
            status = dict(status='TRAINING', arm=args.arm, pid=os.getpid(), decisions=step, budget=config['budget'],
                          agent_transitions=agent_steps, optimizer_steps=optimizer_steps,
                          training_and_evaluation_decisions_per_second=speed,
                          eta_seconds=(config['budget'] - step) / max(speed, 1e-9),
                          last_checkpoint=entries[-1], last_metrics=row, classification='PROVISIONAL', updated_at=now())
            atomic_json(run / 'progress.json', status)
            print(json.dumps({k: v for k, v in status.items() if k not in ('last_metrics', 'last_checkpoint')}), flush=True)
            if step % 2048 == 0:
                save_checkpoint(run / 'checkpoints/resume_latest.pt', trainer, adapter, config, step, agent_steps, optimizer_steps)
            if step % config['checkpoint_interval'] == 0 or step == config['budget']:
                if sources() != locked_sources:
                    raise ValueError('frozen runtime/evaluator source changed')
                current_checkpoint = record_checkpoint(run, trainer, adapter, config, step, agent_steps, optimizer_steps, entries)
                atomic_json(run / 'schedule_events.json', adapter.schedule_events)
                phase = 'checkpoint_evaluation'
                atomic_json(run / 'progress.json', {**status, 'status': 'EVALUATING', 'phase': phase, 'updated_at': now()})
                evaluate(current_checkpoint, run / f'evaluations/screen_{step:09d}.json', config, 'screen', args.device)
                refresh(run)
        phase = 'selection_and_final'
        atomic_json(run / 'progress.json', dict(status='SELECTION_AND_FINAL', arm=args.arm, pid=os.getpid(),
                    decisions=step, budget=config['budget'], classification='PROVISIONAL', updated_at=now()))
        screens = refresh(run)
        ordered = sorted(screens, key=selection_key, reverse=True)
        candidates = {r['decisions']: r for r in ordered[:config['evaluation']['top_screen_candidates']]}
        candidates[0] = min(screens, key=lambda r: r['decisions'])
        candidates[step] = max(screens, key=lambda r: r['decisions'])
        selections = [evaluate(r['checkpoint'], run / f'evaluations/selection_{s:09d}.json', config,
                              'selection', args.device) for s, r in sorted(candidates.items())]
        selected = max(selections, key=selection_key)
        initial, last = min(screens, key=lambda r: r['decisions']), max(screens, key=lambda r: r['decisions'])
        finals = {}
        for name, reference in [('initial', initial), ('selected', selected), ('last', last)]:
            finals[name] = evaluate(reference['checkpoint'], run / f'evaluations/final_{reference["decisions"]:09d}.json',
                                    config, 'final', args.device)
        result = closeout(run, config, selected, finals)
        atomic_json(run / 'progress.json', dict(status='COMPLETE_BUDGET_AND_EVALUATION', arm=args.arm,
                    pid=None, decisions=step, budget=config['budget'], classification='PROVISIONAL',
                    last_checkpoint=entries[-1], selected_checkpoint_decisions=selected['decisions'],
                    learning_assessment=result['learning_assessment'], updated_at=now(), next_stage_started=False))
    except BaseException as error:
        atomic_json(run / 'failure.json', dict(status='STOPPED_FOR_BLOCKING_BUG', arm=args.arm, pid=os.getpid(),
                    decisions=step, phase=phase, error=repr(error), traceback=traceback.format_exc(),
                    last_checkpoint=entries[-1] if entries else None,
                    runtime_fingerprint=adapter.fingerprint(), automatic_retry=False, updated_at=now()))
        atomic_json(run / 'progress.json', dict(status='STOPPED_FOR_BLOCKING_BUG', arm=args.arm, pid=None,
                    decisions=step, budget=config['budget'], classification='PROVISIONAL', updated_at=now()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['N3', 'R3', 'V3'], required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', type=Path)
    run(parser.parse_args())
