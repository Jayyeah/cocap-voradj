#!/usr/bin/env python3
"""Fixed1024 CPU pilots, leaving GPUs and existing long-run worktrees alone."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from terl_mappo.batch02_native.runtime import NativeAdapter,build,collect,sources,save_checkpoint
from terl_mappo.batch02_native.evaluation import evaluate,EVALUATOR_VERSION
from terl_mappo.run import atomic_json,file_hash
from verify_batch02_native_20261011 import continuation


def main(args):
    torch.set_num_threads(1)
    config_path=ROOT/f'configs/experiments/terl_mappo_batch02_native_20261011/{args.arm.lower()}.json'
    config=json.loads(config_path.read_text())
    assert config['pilot_budget']==1024 and config['budget']==1000000 and config['checkpoint_interval']==25000
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    run=args.output_root.resolve()/args.arm
    if run.exists(): raise ValueError('existing pilot is immutable; no automatic retry')
    run.mkdir(parents=True)
    step=0
    try:
        adapter=NativeAdapter(config['learner']['seed'],args.arm)
        trainer=build(config,adapter,'cpu')
        atomic_json(run/'manifest.json',dict(arm=args.arm,pid=os.getpid(),experiment_SHA=sha,config=config,
            config_sha256=file_hash(config_path),sources=sources(),contract=adapter.contract(),
            evaluator_version=EVALUATOR_VERSION,budget=1024,device='cpu',initialization='scratch; pilot weights never seed long run',
            classification='PROVISIONAL',source_checkpoint=None))
        checkpoints=[]; metrics=[]; episodes=[]; agent_steps=optimizer_steps=0
        first=run/'checkpoints/step_000000000.pt'
        checkpoints.append(dict(decisions=0,path=str(first),sha256=save_checkpoint(first,trainer,adapter,config,0,0,0)))
        atomic_json(run/'progress.json',dict(status='PILOT_TRAINING',decisions=0,budget=1024,pid=os.getpid()))
        for _ in range(4):
            batch,completed=collect(trainer,adapter,256)
            update=trainer.update(batch,categorical=True)
            assert all(np.isfinite(v) for v in update.values())
            parameters=list(trainer.actor.parameters())+list(trainer.value.parameters())
            assert all(torch.isfinite(p).all() for p in parameters)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
            step+=256;agent_steps+=int(batch['active_mask'].sum());optimizer_steps+=int(update['minibatch_updates'])
            row=dict(decisions=step,reward_mean=float(batch['rewards'].mean()),**update)
            metrics.append(row);episodes.extend(completed)
            atomic_json(run/'progress.json',dict(status='PILOT_TRAINING',decisions=step,budget=1024,pid=os.getpid(),last_metrics=row))
        final=run/'checkpoints/step_000001024.pt'
        final_hash=save_checkpoint(final,trainer,adapter,config,step,agent_steps,optimizer_steps)
        checkpoints.append(dict(decisions=step,path=str(final),sha256=final_hash))
        atomic_json(run/'checkpoint_list.json',checkpoints)
        # Additional16 decisions are diagnostic, not pilot capability evidence.
        save_checkpoint(run/'resume.pt',trainer,adapter,config,step,agent_steps,optimizer_steps)
        atomic_json(run/'expected.json',continuation(trainer,adapter))
        subprocess.run([sys.executable,str(ROOT/'tools/verify_batch02_native_20261011.py'),
            '--arm',args.arm,'--output',str(run),'--restore'],cwd=ROOT,check=True)
        atomic_json(run/'progress.json',dict(status='PILOT_EVALUATOR_SMOKE',decisions=step,budget=1024,pid=os.getpid()))
        evaluation=evaluate(final,run/'pilot_evaluator_smoke.json',config,'screen','cpu',smoke_episodes=1)
        report=dict(status='PILOT_ENGINEERING_PASS',arm=args.arm,classification='PROVISIONAL',experiment_SHA=sha,
            config_sha256=file_hash(config_path),pilot_decisions=1024,diagnostic_unique_decisions=16,resume_replay_decisions=16,
            PPO_calls=4,minibatch_updates=optimizer_steps,active_agent_transitions=agent_steps,metrics=metrics,episodes=episodes,
            native_reward_decomposition=adapter.ledger_total,reconstruction_residual_max=adapter.ledger_residual_max,
            evaluator_compatibility=True,evaluator_smoke_episodes=len(evaluation['episodes']),fresh_process_resume_exact=True,
            checkpoint_list=checkpoints,gradient_finite=True,blocking_bugs=[],learning_claim=False,
            long_run_eligible=True,long_initialization='fresh scratch; not pilot resume',
            warning='1024 pilot and2 evaluator episodes cannot establish learning or success')
        atomic_json(run/'pilot_report.json',report)
        atomic_json(run/'progress.json',dict(status='PILOT_ENGINEERING_PASS',decisions=step,budget=1024,pid=None,classification='PROVISIONAL'))
        print(json.dumps({k:v for k,v in report.items() if k not in ('metrics','episodes','checkpoint_list')},ensure_ascii=False))
    except BaseException as error:
        atomic_json(run/'failure.json',dict(status='STOPPED_FOR_BLOCKING_BUG',decisions=step,error=repr(error),
            traceback=traceback.format_exc(),automatic_retry=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['N3','R3','V3'],required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    main(parser.parse_args())
