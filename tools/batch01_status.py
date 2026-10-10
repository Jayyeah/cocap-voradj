#!/usr/bin/env python3
"""Inspect persistent registry, actual processes and bounded pilot progress."""
import argparse
import fcntl
from datetime import datetime,timezone
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];DIRECTORY=ROOT/'artifacts/2026-10-09_terl_mappo_batch01'
def read(path):return json.loads(Path(path).read_text())
def write(path,doc):
 p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def alive(pid,run_id):
 if not pid:return False
 try:
  p=Path(f'/proc/{pid}');cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  return p.stat().st_uid==os.getuid() and 'terl_mappo.batch01.arms.runner' in cmd and run_id in cmd
 except FileNotFoundError:return False

def evaluator_alive(pid,output):
 try:
  p=Path(f'/proc/{pid}');cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  return p.stat().st_uid==os.getuid() and 'terl_mappo.batch01.arms.evaluator' in cmd and str(output) in cmd
 except FileNotFoundError:return False

def inspect_runs():
 registry=read(DIRECTORY/'run_registry.json');rows=[]
 for registered in registry.get('pilot_runs',[]):
  row=dict(registered);out=Path(row['output_root']);progress=out/'progress.json'
  if progress.exists():
   live=read(progress)
   if live['run_id']!=row['run_id']:raise ValueError('registry/progress ID mismatch')
   if live['base']['candidate_sha']!=row['source_candidate_sha'] or live['delta_hash']!=row['delta_hash']:raise ValueError('registry/live source pin mismatch')
   prior_evaluations=row.get('evaluation',[])
   row.update({k:v for k,v in live.items() if k not in ('base','branch','head','delta_hash','run_id')})
   row['evaluation']=list({e['output']:dict(e) for e in prior_evaluations+live.get('evaluation',[])}.values())
   row['actual_process_head']=live['head'];row['alive']=alive(row.get('pid'),row['run_id'])
   if row['status'] in {'PROVISIONAL_RUNNING','PROVISIONAL_LONG_RUNNING','STARTING','PROVISIONAL_LONG_STARTING'} and not row['alive']:row['status']='STOPPED_BY_SUPERVISOR' if row.get('supervisor_event') else 'STOPPED_UNEXPECTEDLY'
   if row['status'] in {'PROVISIONAL_RUNNING','PROVISIONAL_LONG_RUNNING'} and (row['step']<=row['start_step'] or row['update']<=row['start_update']):raise ValueError('unverified RUNNING')
   for evaluation in row.get('evaluation',[]):
    p=Path(evaluation['output'])
    if p.exists():
     e=read(p);evaluation.update(status=e['status'],episodes=len(e['episodes']),checkpoint_sha256=e['checkpoint_sha256'])
    elif p.with_suffix('.partial.json').exists():evaluation.update(status='IN_PROGRESS',episodes=read(p.with_suffix('.partial.json'))['completed'])
    else:evaluation['status']='PENDING_SCREEN'
    if not p.exists() and not evaluator_alive(evaluation.get('pid'),p):
     evaluation['status']='EVALUATION_FAILED_PARTIAL' if p.with_suffix('.partial.json').exists() else 'EVALUATION_STOPPED'
     log=out/'logs'/f"eval_screen_{evaluation['step']:09d}.log"
     if log.exists():evaluation['last_exception']='\n'.join(log.read_text(errors='replace').splitlines()[-15:])
   if row['status'] in {'PROVISIONAL_COMPLETE_PENDING_SCREEN','PROVISIONAL_LONG_COMPLETE_PENDING_SCREEN'} and row.get('evaluation') and all(e['status']=='PROVISIONAL_SCREEN' for e in row['evaluation']):row['status']='PROVISIONAL_LONG_COMPLETE' if row.get('execution_mode')=='PROVISIONAL_LONG' else 'PROVISIONAL_COMPLETE'
  else:row['alive']=False
  if row.get('scientific_quarantine'):
   row['pre_quarantine_status']=row['status'];row['status']='QUARANTINED_PROVISIONAL';row['formal_evidence']=False
  rows.append(row)
 return registry,rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');p.add_argument('--json',action='store_true');p.add_argument('--current',action='store_true');args=p.parse_args()
 if args.write:
  lockdir=Path('/home/yjq/rl/CoCap1/batch01-commander-runtime');lockdir.mkdir(parents=True,exist_ok=True)
  descriptor=(lockdir/'registry.lock').open('w');fcntl.flock(descriptor,fcntl.LOCK_EX)
 registry,rows=inspect_runs();now=datetime.now(timezone.utc).isoformat()
 if args.write:
  registry['pilot_runs']=rows;registry['updated_at']=now;write(DIRECTORY/'run_registry.json',registry)
  state=read(DIRECTORY/'batch_state.json');central=read(ROOT/'artifacts/2026-09-21_ac_master_dag/state.json')
  launched=any(r.get('step',r['start_step'])>r['start_step'] for r in rows)
  state['provisional_training_launched']=launched;state['pilot_arm_states']={r['variant']:r['status'] for r in rows};state['updated_at']=now
  for row in rows:
   if row.get('scientific_quarantine') or row['source_candidate_sha']!=state['core']['candidate_sha']:continue
   key='P1-'+row['variant'].split('-')[1] if row['variant'].startswith('p1-') else row['variant'].upper()
   state.setdefault('engineering_arm_states',{})[key]=row['status']
  queue_path=Path('/home/yjq/rl/CoCap1/batch01-commander-runtime/v3r2_long_queue.json')
  if queue_path.exists():
   queue=read(queue_path);state['provisional_long_queue']=queue
   for label,item in queue.get('items',{}).items():
    if not any(r['variant']==label and r.get('execution_mode')=='PROVISIONAL_LONG' for r in rows):state['engineering_arm_states'][label.upper()]=item['status']
   central['batch01']['provisional_long_queue']=queue
  central['batch01']['engineering_arm_states']=state.get('engineering_arm_states',{})
  central['batch01']['provisional_training_launched']=launched;central['batch01']['pilot_gpu_leases']=state['pilot_gpu_leases'];central['batch01']['pilot_runs']=[{'run_id':r['run_id'],'status':r['status'],'pid':r.get('pid'),'step':r.get('step'),'checkpoint':r.get('checkpoint')} for r in rows]
  central['updated_at']=now
  write(DIRECTORY/'batch_state.json',state);write(ROOT/'artifacts/2026-09-21_ac_master_dag/state.json',central)
 state=read(DIRECTORY/'batch_state.json')
 if args.json:print(json.dumps({'time':now,'BASE_FROZEN':state['base_freeze_status']=='BASE_FROZEN','QA':state['qa']['status'],'core_status':state['core']['status'],'runs':rows},ensure_ascii=False,indent=2))
 else:
  print('BASE:',state['base_freeze_status'],'|',state['core']['status'],'|',state['qa']['status'])
  if args.current:
   latest={}
   for r in rows:
    if r['source_candidate_sha']==state['core']['candidate_sha'] and not r.get('scientific_quarantine'):latest[r['variant']]=r
   queued={k for k,v in state.get('provisional_long_queue',{}).get('items',{}).items() if v['status']!='PROVISIONAL_LONG_LAUNCHED'}
   rows=[r for r in latest.values() if r['variant'] not in queued or r.get('execution_mode')=='PROVISIONAL_LONG']
  for r in rows:
   cp=r.get('checkpoint') or {};print(r['run_id'],r['status'],'PID',r.get('pid'),'GPU',r.get('physical_gpu'),'step',r.get('step',r['start_step']),'update',r.get('update'),'checkpoint',cp.get('path'),'alive',r['alive'])
  if args.current:
   for label,item in state.get('provisional_long_queue',{}).get('items',{}).items():
    if not any(r['variant']==label and r.get('execution_mode')=='PROVISIONAL_LONG' for r in rows):print(label.upper(),'LONG',item['status'],'PID None; previous pilot status is historical')
if __name__=='__main__':main()
