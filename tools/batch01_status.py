#!/usr/bin/env python3
"""Inspect persistent registry, actual processes and bounded pilot progress."""
import argparse
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

def inspect_runs():
 registry=read(DIRECTORY/'run_registry.json');rows=[]
 for registered in registry.get('pilot_runs',[]):
  row=dict(registered);out=Path(row['output_root']);progress=out/'progress.json'
  if progress.exists():
   live=read(progress)
   if live['run_id']!=row['run_id']:raise ValueError('registry/progress ID mismatch')
   if live['base']['candidate_sha']!=row['source_candidate_sha'] or live['delta_hash']!=row['delta_hash']:raise ValueError('registry/live source pin mismatch')
   row.update({k:v for k,v in live.items() if k not in ('base','branch','head','delta_hash','run_id')})
   row['actual_process_head']=live['head'];row['alive']=alive(row.get('pid'),row['run_id'])
   if row['status'] in {'PROVISIONAL_RUNNING','STARTING'} and not row['alive']:row['status']='STOPPED_UNEXPECTEDLY'
   if row['status']=='PROVISIONAL_RUNNING' and (row['step']<=row['start_step'] or row['update']<=row['start_update']):raise ValueError('unverified RUNNING')
   for evaluation in row.get('evaluation',[]):
    p=Path(evaluation['output'])
    if p.exists():
     e=read(p);evaluation.update(status=e['status'],episodes=len(e['episodes']),checkpoint_sha256=e['checkpoint_sha256'])
    elif p.with_suffix('.partial.json').exists():evaluation.update(status='IN_PROGRESS',episodes=read(p.with_suffix('.partial.json'))['completed'])
    else:evaluation['status']='PENDING_SCREEN'
  else:row['alive']=False
  rows.append(row)
 return registry,rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');p.add_argument('--json',action='store_true');args=p.parse_args()
 registry,rows=inspect_runs();now=datetime.now(timezone.utc).isoformat()
 if args.write:
  registry['pilot_runs']=rows;registry['updated_at']=now;write(DIRECTORY/'run_registry.json',registry)
  state=read(DIRECTORY/'batch_state.json');central=read(ROOT/'artifacts/2026-09-21_ac_master_dag/state.json')
  launched=any(r.get('step',r['start_step'])>r['start_step'] for r in rows)
  state['provisional_training_launched']=launched;state['pilot_arm_states']={r['variant']:r['status'] for r in rows};state['updated_at']=now
  central['batch01']['provisional_training_launched']=launched;central['batch01']['pilot_gpu_leases']=state['pilot_gpu_leases'];central['batch01']['pilot_runs']=[{'run_id':r['run_id'],'status':r['status'],'pid':r.get('pid'),'step':r.get('step'),'checkpoint':r.get('checkpoint')} for r in rows]
  central['updated_at']=now
  write(DIRECTORY/'batch_state.json',state);write(ROOT/'artifacts/2026-09-21_ac_master_dag/state.json',central)
 if args.json:print(json.dumps({'time':now,'BASE_FROZEN':False,'QA':'QA_PENDING','runs':rows},ensure_ascii=False,indent=2))
 else:
  print('BASE: UNFROZEN | CORE_V3_SELFTEST_PASS | QA_PENDING')
  for r in rows:
   cp=r.get('checkpoint') or {};print(r['run_id'],r['status'],'PID',r.get('pid'),'GPU',r.get('physical_gpu'),'step',r.get('step',r['start_step']),'update',r.get('update'),'checkpoint',cp.get('path'),'alive',r['alive'])
if __name__=='__main__':main()
