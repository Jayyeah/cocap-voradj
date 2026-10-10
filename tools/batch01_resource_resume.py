#!/usr/bin/env python3
"""Strict resume of an authorized run after a verified resource stop only.
No scientific-failure restart, budget extension, source mutation or formal promotion.
"""
import argparse,datetime,fcntl,os,shlex,subprocess,sys,time
from pathlib import Path
import batch01_long_commander as m
from batch01_status import inspect_runs,alive
REASONS={'RAM reserve below8GiB','GPU dynamic free headroom below4GiB'}

def eligible(row):
 if row.get('execution_mode')!='PROVISIONAL_LONG' or row.get('alive') or row.get('scientific_quarantine') or row.get('last_exception'):return False
 event=row.get('supervisor_event')
 return bool(event and event.get('reason') in REASONS and row.get('checkpoint') and row['checkpoint']['step']<row['authorized_end_step'])

def choose(label):
 _,rows=inspect_runs();row=next(r for r in rows if r['variant']==label and r.get('execution_mode')=='PROVISIONAL_LONG')
 options=[]
 for gpu in (0,1):
  try:
   gate=m.gate_gpu(gpu)
   if gate['available_RAM_bytes']<24*1024**3:continue
   options.append(gate)
  except ValueError:pass
 if not options:return None
 preferred=[g for g in options if str(g['gpu'])==row.get('physical_gpu')]
 return (preferred or sorted(options,key=lambda x:-x['free_gpu_MiB']))[0]['gpu']

def resume(label,gpu):
 d=m.read(m.delta(label));grant=m.read(m.AUTH)['runs'][label];run_id=grant['run_id'];root=m.root(label)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()
 if subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=root,text=True).split()[0]!=head:raise ValueError('remote source HEAD mismatch')
 for item in d['source_changes']:
  if m.hf(root/item['path'])!=item['after']:raise ValueError('source delta changed before resource resume')
 with (m.R/'registry.lock').open('w') as fd:
  fcntl.flock(fd,fcntl.LOCK_EX);registry,rows=inspect_runs();row=next(r for r in rows if r['run_id']==run_id);state=m.read(m.D/'batch_state.json')
  if not eligible(row):raise ValueError('only a resource-stopped, nonfailed, nonquarantined run is resumable')
  if state.get('new_run_hold') is not None or state['core']['candidate_sha']!=m.CANDIDATE or state['core']['canonical_lock_sha256']!=m.PIN:raise ValueError('BASE hold/pin changed')
  if head!=row['head'] or row['delta_hash']!=m.fp(d) or row['authorization_sha256']!=m.hf(m.AUTH) or row['authorized_end_step']!=grant['authorized_end_step']:raise ValueError('resume source/budget binding mismatch')
  out=Path(row['output_root']);cp=row['checkpoint'];manifest=m.read(out/'manifest.json');side=m.read(Path(cp['path']).with_suffix('.batch01.json'))
  if m.hf(cp['path'])!=cp['sha256'] or side['checkpoint_sha256']!=cp['sha256'] or side['manifest_fingerprint']!=m.fp(manifest) or side['base']!=d['base'] or side['steps']!=cp['step']:raise ValueError('checkpoint/sidecar/manifest binding invalid')
  if manifest['execution_mode']!='PROVISIONAL_LONG' or manifest['delta_manifest']!=d or manifest['budget_authorization_sha256']!=m.hf(m.AUTH):raise ValueError('stored source/grant mismatch')
  session=row['tmux']
  if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('existing session: refuse duplicate resume')
  gate=m.gate_gpu(gpu)
  if gate['available_RAM_bytes']<24*1024**3:raise ValueError('RAM recovery must have24GiB headroom')
  attempt=len(row.get('resource_resume_history',[]))+1
  receipt={'attempt':attempt,'time':m.now(),'reason':row['supervisor_event']['reason'],'previous_pid':row.get('pid'),'last_observed_step':row.get('step'),'restore_step':cp['step'],'checkpoint':cp,'unsaved_decisions_discarded':row.get('step',cp['step'])-cp['step'],'metrics_previous_byte_offset':(out/'metrics.jsonl').stat().st_size,'authorized_end_step_unchanged':row['authorized_end_step'],'source_HEAD_unchanged':head,'strict_full_state_load_required':True,'resource_gate':gate,'formal_evidence':False}
  row.setdefault('resource_resume_history',[]).append(receipt);row.setdefault('supervisor_event_history',[]).append(row.pop('supervisor_event'));row.update(status='PROVISIONAL_LONG_RESTARTING',physical_gpu=str(gpu),gpu_uuid=gate['uuid'],resource_resume_pending=True)
  for i,x in enumerate(registry['pilot_runs']):
   if x['run_id']==run_id:registry['pilot_runs'][i]=row;break
  lease=next(x for x in state['pilot_gpu_leases'] if x['id']==row['lease_id']);lease.update(gpu=gpu,gpu_uuid=gate['uuid'],external_sharing_authorized=gate['external_sharing_authorized'],maximum_compute_processes=gate['maximum_own_lines'])
  if datetime.datetime.fromisoformat(lease['expires_at'])<=datetime.datetime.now(datetime.timezone.utc):raise ValueError('lease expired; explicit resource requalification required')
  m.write(m.D/'run_registry.json',registry);m.write(m.D/'batch_state.json',state);m.write(m.D/'commander_v3r2'/f'{label}_resource_resume_{attempt:02d}_request.json',receipt)
 args=m.command(label,'terl_mappo.batch01.arms.runner',['--delta',str(m.delta(label)),'--run-id',run_id,'--output',str(out),'--device','cuda:0','--threads','1','--mode','PROVISIONAL_LONG','--decisions',str(grant['authorized_end_step']-grant['start_step']),'--resume',cp['path']])
 environment=m.env(label,gpu);environment['TMPDIR']=str(out/'tmp');keys=('CUDA_VISIBLE_DEVICES','PYTHONPATH','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TMPDIR')
 log=out/'logs'/f'resource_resume_{attempt:02d}.log';script=out/f'resource_resume_{attempt:02d}.sh'
 script.write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(root))+'\n'+'\n'.join('export '+k+'='+shlex.quote(environment[k]) for k in keys)+'\nexec '+shlex.join(args)+' >> '+shlex.quote(str(log))+' 2>&1\n');script.chmod(0o755)
 m.gate_gpu(gpu);subprocess.run(['tmux','new-session','-d','-s',session,'bash '+shlex.quote(str(script))],check=True)
 deadline=time.monotonic()+300
 while time.monotonic()<deadline:
  live=m.read(out/'progress.json')
  if live['pid']!=receipt['previous_pid']:
   if live.get('last_exception'):raise ValueError('resource resume failed: '+live['last_exception'])
   if live['step']>cp['step'] and live.get('checkpoint') and live['checkpoint']['step']>cp['step'] and alive(live['pid'],run_id):
    receipt.update(actual_pid=live['pid'],actual_step=live['step'],actual_update=live['update'],actual_checkpoint=live['checkpoint'],status='STRICT_RESOURCE_RESUME_FIRST_UPDATE_PASS')
    with (m.R/'registry.lock').open('w') as fd:
     fcntl.flock(fd,fcntl.LOCK_EX);registry=m.read(m.D/'run_registry.json')
     for row in registry['pilot_runs']:
      if row['run_id']==run_id:row['resource_resume_history'][-1]=receipt;row['resource_resume_pending']=False
     m.write(m.D/'run_registry.json',registry)
    m.write(m.D/'commander_v3r2'/f'{label}_resource_resume_{attempt:02d}_pass.json',receipt)
    subprocess.run([sys.executable,str(m.C/'tools/batch01_status.py'),'--write'],check=True,stdout=subprocess.DEVNULL)
    print(label,'RESUMED',live['pid'],'GPU',gpu,'step',live['step'],flush=True);return
  if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode:raise ValueError('resume process exited: '+str(log))
  time.sleep(1)
 raise TimeoutError('resume has no confirmed new update')

def main():
 p=argparse.ArgumentParser();p.add_argument('--labels',nargs='+',required=True);p.add_argument('--gpu',type=int,choices=[0,1],required=True);a=p.parse_args()
 for label in a.labels:resume(label,a.gpu)
if __name__=='__main__':main()
