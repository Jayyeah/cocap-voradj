#!/usr/bin/env python3
"""500k -> 1M P1 setting continuation, retaining/evaluating every25k."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
import torch
from terl_mappo.batch02_stability.runtime import (config_and_plan, fork_parent, profile,
                                                save, restore, wiring_sources, CONFIGS)
from terl_mappo.batch02_stability.reporting import refresh
from terl_mappo.batch01.evaluation import isolated_rng
from terl_mappo.run import atomic_json, file_hash


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def append(path, data):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(data,allow_nan=False)+'\n')


def report(run):
    # Plotting/imports are diagnostics, never an extra optimization RNG variable.
    with isolated_rng():
        return refresh(run)


def run(args):
    torch.set_num_threads(1)
    config, _, plan = config_and_plan(args.arm)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(plan['physical_gpu']) or args.device != 'cuda:0':
        raise ValueError('use original parent physical GPU with process-local cuda:0')
    sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True):
        raise ValueError('commit source/config before launching')
    verification = json.loads((ROOT / 'artifacts/2026-10-12_batch02_stability' / f'{args.arm}_engineering_verification.json').read_text())
    if verification['status'] != 'ENGINEERING_PASS' or verification['parent_checkpoint']['sha256'] != plan['parent_checkpoint']['sha256']:
        raise ValueError('parent strict resume validation required')
    runtime,evaluator = profile(args.arm)
    run = args.output_root.resolve() / (args.arm+'-stability')
    if run.exists() and not args.resume:
        raise ValueError('no duplicate/overwrite; explicit strict resume required')
    run.mkdir(parents=True,exist_ok=True)
    locked_sources = wiring_sources()
    baseline_sources = runtime.sources()
    phase = 'strict_parent_fork'
    step = 500000
    entries = []
    try:
        if args.resume:
            manifest = json.loads((run/'manifest.json').read_text())
            if manifest['experiment_SHA'] != sha or manifest['wiring_sources'] != locked_sources:
                raise ValueError('continuation implementation changed')
            trainer,adapter,config,payload = restore(args.arm,args.resume,args.device)
            entries = json.loads((run/'checkpoint_list.json').read_text())
            # A rollback may keep later immutable checkpoints; do not silently reuse them.
            if any(e['decisions'] > payload['decisions'] for e in entries):
                raise ValueError('resume precedes retained checkpoints; use a separate continuation directory')
        else:
            trainer,adapter,config,payload,receipt = fork_parent(args.arm,args.device)
            atomic_json(run/'fork_receipt.json',receipt)
            manifest = dict(line_id=args.arm+'-stability', arm=args.arm, experiment_SHA=sha,
                parent_SHA=plan['baseline_SHA'], parent_checkpoint=plan['parent_checkpoint'],
                baseline_run=plan['baseline_run'], config=config,
                config_sha256=file_hash(CONFIGS/f'{args.arm.lower()}.json'),
                scientific_delta=plan['scientific_delta'], task_contract=adapter.contract(),
                evaluator_version=evaluator.EVALUATOR_VERSION, evaluator_sha256=plan['evaluator_sha256'],
                wiring_sources=locked_sources, baseline_sources=baseline_sources,
                seed_manifest=plan['seed_manifest'], initial_step=500000, end_step=1000000,
                additional_budget=500000, checkpoint_interval=25000, screen_interval=25000,
                checkpoint_directory=str(run/'checkpoints'), started_at=now(),
                pid=os.getpid(),physical_gpu=plan['physical_gpu'],classification='PROVISIONAL',
                initialization=plan['initialization'], no_auto_extension=True, next_stage_started=False)
            atomic_json(run/'manifest.json',manifest)
        step,agent_steps,optimizer_steps = (payload[k] for k in ['decisions','agent_transitions','optimizer_steps'])

        def retain():
            path = run/f'checkpoints/step_{step:09d}.pt'
            if path.exists():
                raise ValueError('immutable checkpoint already exists')
            digest = save(args.arm,path,trainer,adapter,config,step,agent_steps,optimizer_steps)
            entry = dict(decisions=step,path=str(path),sha256=digest,created_at=now())
            entries.append(entry)
            atomic_json(run/'checkpoint_list.json',entries)
            atomic_json(run/'last_checkpoint.json',entry)
            return path

        if not args.resume:
            retain()
            # Common500k policy was proven identical; retain its original full screen.
            destination = run/'evaluations/screen_000500000.json'
            destination.parent.mkdir(parents=True,exist_ok=True)
            original_screen = Path(plan['baseline_run'])/'evaluations/screen_000500000.json'
            reference = json.loads(original_screen.read_text())
            if reference['checkpoint_sha256'] != plan['parent_checkpoint']['sha256']:
                raise ValueError('common500k evaluation parent mismatch')
            shutil.copyfile(original_screen,destination)
            atomic_json(run/'initial_screen_reference.json',dict(source=str(original_screen),sha256=file_hash(original_screen),
                same_parent_policy=True,reused_complete_original20_episode_screen=True,
                reason='exact state fork changes only target_kl; no actor/environment/evaluator difference'))
            report(run)
        elif step % 25000 == 0:
            phase = 'resumed_checkpoint_evaluation'
            entry = next(row for row in entries if row['decisions']==step)
            evaluator.evaluate(entry['path'],run/f'evaluations/screen_{step:09d}.json',config,'screen',args.device)
            report(run)
        started,initial_step = time.monotonic(),step
        while step < 1000000:
            phase = 'training'
            count = min(config['learner']['rollout_length'],1000000-step,25000-step%25000)
            batch,episodes = runtime.collect(trainer,adapter,count)
            metrics = trainer.update(batch,categorical=True)
            finite = all(np.isfinite(v) for v in metrics.values()) and all(torch.isfinite(p).all().item()
                       for p in [*trainer.actor.parameters(),*trainer.value.parameters()])
            grad_finite = all(p.grad is None or torch.isfinite(p.grad).all().item()
                              for p in [*trainer.actor.parameters(),*trainer.value.parameters()])
            if not finite or not grad_finite:
                raise FloatingPointError('nonfinite PPO update')
            step += count
            agent_steps += int(batch['active_mask'].sum())
            optimizer_steps += int(metrics['minibatch_updates'])
            active_rewards = batch['rewards'][batch['active_mask']]
            row = dict(decisions=step, active_agent_transitions=agent_steps, optimizer_steps=optimizer_steps,
                reward_mean_active=float(active_rewards.mean()) if active_rewards.size else None,
                training_episodes=len(episodes),training_capture_completions=sum(e['capture_completion'] for e in episodes),
                training_collision_episodes=sum(e['collision_event'] for e in episodes),
                training_boundary_episodes=sum(e['boundary_collision_event'] for e in episodes),
                gradient_finite=grad_finite,parameter_finite=finite, live_target_kl=trainer.config.target_kl,
                reward_clock=int(adapter.env.total_steps),coverage_ce_speed_weight=float(adapter.env.reward_cfg['coverage_ce_speed_weight']),**metrics)
            if args.arm in ('R3','V3'):
                row.update(reward_decomposition_total=adapter.ledger_total,reward_decomposition_residual_max=adapter.ledger_residual_max)
            append(run/'metrics.jsonl',row)
            for episode in episodes:
                append(run/'episodes.jsonl',dict(recorded_at_decisions=step,**episode))
            speed = (step-initial_step)/max(time.monotonic()-started,1e-9)
            status = dict(status='TRAINING',line_id=args.arm+'-stability',pid=os.getpid(),decisions=step,
                continuation_decisions=step-500000,end_step=1000000,additional_budget=500000,
                last_checkpoint=entries[-1],last_metrics=row,classification='PROVISIONAL',updated_at=now(),
                throughput=speed,eta_seconds=(1000000-step)/max(speed,1e-9))
            atomic_json(run/'progress.json',status)
            print(json.dumps({k:v for k,v in status.items() if k not in ('last_checkpoint','last_metrics')}),flush=True)
            if step % 2048 == 0:
                save(args.arm,run/'checkpoints/resume_latest.pt',trainer,adapter,config,step,agent_steps,optimizer_steps)
            if step % 25000 == 0:
                if wiring_sources() != locked_sources or runtime.sources() != baseline_sources:
                    raise ValueError('frozen continuation or task/evaluator source changed')
                checkpoint = retain()
                phase = 'checkpoint_evaluation'
                atomic_json(run/'progress.json',{**status,'status':'EVALUATING','phase':phase,'last_checkpoint':entries[-1]})
                evaluator.evaluate(checkpoint,run/f'evaluations/screen_{step:09d}.json',config,'screen',args.device)
                report(run)
        result = report(run)
        atomic_json(run/'progress.json',dict(status='COMPLETE_BUDGET_AND_SCREEN',line_id=args.arm+'-stability',pid=None,
            decisions=step,continuation_decisions=500000,end_step=1000000,last_checkpoint=entries[-1],
            classification='PROVISIONAL',updated_at=now(),report_status=result['status'],next_stage_started=False))
    except BaseException as error:
        atomic_json(run/'failure.json',dict(status='STOPPED_FOR_BLOCKING_BUG',line_id=args.arm+'-stability',
            decisions=step,phase=phase,error=repr(error),traceback=traceback.format_exc(),
            last_checkpoint=entries[-1] if entries else None,automatic_retry=False,updated_at=now()))
        atomic_json(run/'progress.json',dict(status='STOPPED_FOR_BLOCKING_BUG',pid=None,decisions=step,
                    classification='PROVISIONAL',updated_at=now()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['R2','R3','V3'],required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--resume',type=Path)
    run(parser.parse_args())
