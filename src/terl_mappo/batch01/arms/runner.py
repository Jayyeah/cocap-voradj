"""Recoverable single-run process. Pilot25k; long provisional requires an explicit pinned user grant."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
from datetime import datetime,timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

from terl_mappo.batch01.contracts import ROOT,LOCK_PATH,fingerprint,verify_base,validate_resources
from terl_mappo.batch01.source_guard import SourceGuard
from terl_mappo.batch01.interfaces import Hooks,assemble
from terl_mappo.batch01.provenance import runtime_manifest,active_values
from terl_mappo.batch01.checkpoints import save_bound,load_bound
from terl_mappo.batch01.evaluation import isolated_rng
from terl_mappo.run import atomic_json,file_hash,post_update_policy_stats

CENTRAL=Path('/home/yjq/rl/CoCap1/ac-master-dag-20260921/artifacts/2026-10-09_terl_mappo_batch01')

def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()

def prepare(delta_path,device='cpu',diagnostics=None,fork=False):
    lock=read(ROOT/LOCK_PATH);delta=read(delta_path);pin=delta['base']['lock_sha256']
    resolved=verify_base(lock,delta,pin)
    guard=SourceGuard.from_base(lock,delta,pin,diagnostics=diagnostics)
    hooks=Hooks();extension=delta.get('extensions')
    if extension:
        name,fn=extension['entrypoint'].split(':')
        with guard.scope():hooks=guard.call(getattr(importlib.import_module(name),fn),extension.get('parameters',{}))
    config=copy.deepcopy(lock['anchor_config']) if fork and delta['line']=='P1' else resolved
    runtime=assemble(config,device,hooks,delta['line'],source_guard=guard)
    if hooks.reward_transform is not None and hasattr(hooks.reward_transform,'bind'):
        guard.call(hooks.reward_transform.bind,runtime.adapter)
    parent=None;receipt=None
    if fork and delta['line']=='P1':
        from .p1 import fork as controlled_fork
        parent,receipt=controlled_fork(runtime,resolved)
    if fork and delta['line']=='T1':
        from .t1 import initialize_stage
        receipt=guard.call(initialize_stage,runtime,delta['extensions']['parameters'])
    return runtime,lock,delta,resolved,parent,receipt


def launch_eval(output,manifest,checkpoint,step):
    target=output/'evaluations'/f'screen_{step:09d}.json'
    cmd=[sys.executable,'-m','terl_mappo.batch01.arms.evaluator','--manifest',str(output/'manifest.json'),
         '--checkpoint',str(checkpoint),'--output',str(target),'--domain','screen']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    log=(output/'logs'/f'eval_screen_{step:09d}.log').open('a')
    p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
    return {'pid':p.pid,'checkpoint':str(checkpoint),'output':str(target),'status':'PENDING_SCREEN','step':step}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--delta',required=True);ap.add_argument('--run-id',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--mode',choices=['PROVISIONAL','PROVISIONAL_LONG','FORMAL','SMOKE','BENCHMARK'],required=True)
    ap.add_argument('--decisions',type=int,required=True);ap.add_argument('--resume');ap.add_argument('--no-evaluation',action='store_true')
    args=ap.parse_args();output=Path(args.output).resolve();validate_resources(output,args.device,args.threads,workers=1)
    if args.decisions<=0 or (args.mode=='PROVISIONAL' and args.decisions>25000):raise ValueError('PROVISIONAL maximum25k additional decisions')
    if args.mode=='FORMAL':
        state=read(CENTRAL/'batch_state.json')
        if state['base_freeze_status']!='BASE_FROZEN' or state.get('QA_SAME_SHA') not in {'PASS','A3_INDEPENDENT_QA_PASS'}:
            raise ValueError('formal training requires independent frozen BASE')
    grant=None
    if args.mode=='PROVISIONAL_LONG':
        from .long_budget import validate_launch,AUTH_SHA
        grant=validate_launch(read(args.delta),args.run_id,args.decisions)
    existed=output.exists();output.mkdir(parents=True,exist_ok=True)
    if existed and not args.resume and (output/'progress.json').exists():raise ValueError('refuse duplicate run; explicit resume required')
    descriptor=(output/'process.lock').open('w');fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for folder in ('logs','checkpoints','evaluations','source_failures'): (output/folder).mkdir(exist_ok=True)
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    runtime,lock,delta,resolved,parent,fork_receipt=prepare(args.delta,args.device,output/'source_failures',fork=not args.resume and (args.mode!='PROVISIONAL_LONG' or read(args.delta)['line']=='T1'))
    if args.mode=='PROVISIONAL_LONG' and not args.resume:
        from .long_budget import migrate
        restored,continuation_receipt=migrate(runtime,delta)
        if restored is not None:parent=restored;fork_receipt=continuation_receipt
    if args.mode=='FORMAL':
        if state['BATCH01_BASE_SHA']!=delta['base']['candidate_sha'] or state['core']['canonical_lock_sha256']!=delta['base']['lock_sha256']:
            raise ValueError('formal ARM parent differs from independently frozen BASE')
        maximum=225000 if delta['line']=='P1' else resolved['budget']
        if args.decisions>maximum:raise ValueError('formal budget exceeds registered window')
    resources=validate_resources(output,args.device,args.threads,workers=1)
    steps=int(parent['steps']) if parent else 0;agent_steps=int(parent['agent_transitions']) if parent else 0
    optimizers=int(parent['optimizer_steps']) if parent else 0
    start_steps=steps;budget=args.decisions
    with isolated_rng():manifest=runtime_manifest(lock,delta,resolved,runtime,resources)
    manifest.update(run_id=args.run_id,execution_mode=args.mode,max_additional_decisions=budget,
                    authorized_end_step=steps+budget,independent_qa_accepted=False,scientific_evidence='PROVISIONAL_NOT_FORMAL' if args.mode in {'PROVISIONAL','PROVISIONAL_LONG'} else args.mode)
    if grant:manifest.update(budget_authorization_sha256=AUTH_SHA,authorization_id='USER-PROVISIONAL-LONG-20261010')
    if args.resume:
        stored=read(output/'manifest.json')
        if stored['run_id']!=args.run_id or stored['execution_mode']!=args.mode or stored['delta_manifest']!=delta:
            raise ValueError('resume run/mode/source delta mismatch')
        manifest=stored
        if grant:validate_launch(delta,args.run_id,args.decisions,stored)
        loaded=load_bound(args.resume,runtime,manifest);steps=loaded['steps'];agent_steps=loaded['agent_transitions'];optimizers=loaded['optimizer_steps']
        start_steps=manifest['authorized_end_step']-manifest['max_additional_decisions']
    else:
        atomic_json(output/'delta.json',delta)
        atomic_json(output/'manifest.json',manifest)
        if fork_receipt:atomic_json(output/'fork_receipt.json',fork_receipt)
    end_step=manifest['authorized_end_step']
    if args.mode=='PROVISIONAL' and end_step-start_steps>25000:raise ValueError('resume cannot extend pilot authorization')
    if grant and (start_steps!=grant['start_step'] or end_step!=grant['authorized_end_step']):raise ValueError('long manifest endpoint differs from pinned authorization')
    session_start_steps=steps;session_start_update=runtime.trainer.update_count
    gpu_uuid=None
    if args.device.startswith('cuda'):
        gpu_uuid=subprocess.check_output(['nvidia-smi',f'--id={os.environ["CUDA_VISIBLE_DEVICES"]}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    status={'run_id':args.run_id,'mode':args.mode,'status':'STARTING','pid':os.getpid(),'pgid':os.getpgrp(),
            'base':delta['base'],'branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
            'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'delta_hash':fingerprint(delta),
            'resolved_config_hash':fingerprint(resolved),'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu_uuid':gpu_uuid,'step':steps,'start_step':start_steps,'authorized_end_step':end_step,
            'update':runtime.trainer.update_count,'last_exception':None,'evaluation':[]}
    start=time.monotonic();latest=None
    def heartbeat(state):
        status.update(status=state,step=steps,update=runtime.trainer.update_count,agent_transitions=agent_steps,
                      optimizer_steps=optimizers,heartbeat=now(),elapsed_seconds=time.monotonic()-start,
                      decisions_per_second=(steps-session_start_steps)/(time.monotonic()-start),
                      disk_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
                      checkpoint=latest,torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if args.device.startswith('cuda') else 0)
        atomic_json(output/'progress.json',status)
    def save(step):
        nonlocal latest
        path=output/'checkpoints'/f'step_{step:09d}.pt';digest=save_bound(path,runtime,manifest,steps,agent_steps,optimizers)
        latest={'path':str(path),'sha256':digest,'step':steps,'bytes':path.stat().st_size}
        atomic_json(output/'checkpoints/latest.json',latest)
        return path
    try:
        if not args.resume:
            path=save(steps)
            if args.mode in {'FORMAL','PROVISIONAL','PROVISIONAL_LONG'} and not args.no_evaluation:status['evaluation'].append(launch_eval(output,manifest,path,steps))
        else:latest=read(output/'checkpoints/latest.json')
        heartbeat('STARTING')
        with (output/'metrics.jsonl').open('a') as metrics_log:
            while steps<end_step:
                length=min(256,end_step-steps,25000-steps%25000)
                t=time.monotonic();batch,episodes=runtime.collect(length);collect_seconds=time.monotonic()-t
                t=time.monotonic();metrics=runtime.update(batch);update_seconds=time.monotonic()-t
                if not all(np.isfinite(value) for value in metrics.values()):raise ValueError('nonfinite PPO metrics')
                steps+=length;agent_steps+=int(np.asarray(batch['active_mask']).sum());optimizers+=int(metrics['minibatch_updates'])
                row={'time':now(),'step':steps,'update':runtime.trainer.update_count,'collect_seconds':collect_seconds,
                     'update_seconds':update_seconds,'live_target_kl':runtime.trainer.config.target_kl,'metrics':metrics,'episodes':episodes}
                if 'reward_components' in batch:
                    row['reward_components_mean']={k:float(np.mean(v)) for k,v in batch['reward_components'].items()}
                if runtime.hooks.backend.categorical:row['post_update_policy']=post_update_policy_stats(runtime.trainer,batch)
                metrics_log.write(json.dumps(row,allow_nan=False)+'\n');metrics_log.flush()
                if steps%25000==0 or steps==end_step or runtime.trainer.update_count==session_start_update+1:
                    path=save(steps)
                    if (steps%25000==0 or steps==end_step) and args.mode in {'PROVISIONAL','PROVISIONAL_LONG','FORMAL'} and not args.no_evaluation:
                        status['evaluation'].append(launch_eval(output,manifest,path,steps))
                heartbeat(args.mode+'_RUNNING' if args.mode in {'PROVISIONAL','PROVISIONAL_LONG'} else 'RUNNING_'+args.mode)
                if args.device.startswith('cuda') and torch.cuda.max_memory_allocated()/1024**2>2048:raise ValueError('GPU allocation exceeded2GiB lease')
                if status['disk_bytes']>8*1024**3 or os.statvfs(output).f_bavail*os.statvfs(output).f_frsize<30*1024**3:
                    raise ValueError('run storage lease/reserve exceeded')
        heartbeat(args.mode+'_COMPLETE_PENDING_SCREEN' if args.mode in {'PROVISIONAL','PROVISIONAL_LONG'} else 'COMPLETE_'+args.mode)
    except BaseException:
        status['last_exception']=traceback.format_exc();heartbeat('FAILED_SOURCE_CONTRACT' if runtime.source_guard.failure else 'FAILED_ENGINEERING')
        raise
    print(json.dumps(status,allow_nan=False),flush=True)

if __name__=='__main__':main()
