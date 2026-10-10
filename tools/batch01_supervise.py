#!/usr/bin/env python3
"""Science/budget supervision; running-job resources are telemetry only.

User override 2026-10-10: resource gates apply before launch, never as a
supervisor reason to terminate an already running training process.
No launch, restart, budget extension or mutation of source-locked runners.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from batch01_status import ROOT,DIRECTORY,read,write,inspect_runs,alive
RUNTIME=Path('/home/yjq/rl/CoCap1/batch01-commander-runtime')
RUNTIME_RESOURCE_STOP_ENABLED=False

def now():return datetime.now(timezone.utc).isoformat()
def memory():
 return {x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:','MemTotal:'))}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def process_rss(pid):
 try:
  return next(int(line.split()[1])*1024 for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('VmRSS:'))
 except (OSError,StopIteration):return None
def resource_alerts(row,rows,leases,free,ram,free_gpu,compute,own):
 """Resource observations cannot enter the scientific stop-reason path."""
 alerts=[]
 if free<30*1024**3:alerts.append('disk reserve below30GiB')
 if row.get('disk_bytes',0)>8*1024**3:alerts.append('run output above8GiB')
 if row.get('torch_peak_allocated_mib',0)>2048:alerts.append('GPU lease above2GiB')
 if free_gpu.get(row.get('gpu_uuid'),0)<4096:alerts.append('GPU dynamic free headroom below4GiB')
 if ram['MemAvailable']<8*1024**3:alerts.append('RAM reserve below8GiB')
 lease=leases.get(row.get('lease_id'))
 try:
  expired=not lease or datetime.fromisoformat(lease['expires_at'])<datetime.now(timezone.utc)
 except (KeyError,ValueError,TypeError):expired=True
 if expired:alerts.append('GPU lease missing/expired')
 if any(row.get('gpu_uuid') and row['gpu_uuid'] in line and int(line.split(',')[0]) not in own for line in compute.splitlines()) and not (lease and lease.get('external_sharing_authorized')):
  alerts.append('external process on originally exclusive leased GPU')
 if lease and lease.get('external_sharing_authorized') and sum(1 for r in rows if r.get('alive') and r.get('gpu_uuid')==row.get('gpu_uuid'))>2:
  alerts.append('shared GPU above startup maximum2 own lines')
 return alerts
def supervise(cache):
 registry,rows=inspect_runs();state=read(DIRECTORY/'batch_state.json');leases={l['id']:l for l in state.get('pilot_gpu_leases',[])}
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True).strip()
 free_gpu={}
 for line in subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines():
  uuid,amount=line.split(',');free_gpu[uuid.strip()]=float(amount)
 stat=os.statvfs('/home/yjq');free=stat.f_bavail*stat.f_frsize;ram=memory();events=[];alerts=[];own={r.get('pid') for r in rows if r.get('alive')}
 for row in rows:
  if row.get('mode',row.get('execution_mode')) not in {'PROVISIONAL','PROVISIONAL_LONG'}:continue
  reason=None;cp=row.get('checkpoint')
  if cp:
   try:
    path=Path(cp['path']).resolve();root=Path(row['output_root']).resolve();s=path.stat();key=(str(path),s.st_size,s.st_mtime_ns)
    if not path.is_relative_to(root/'checkpoints'):reason='checkpoint escapes registered run'
    elif key not in cache:cache[key]=digest(path)
    if not reason and cache[key]!=cp['sha256']:reason='checkpoint hash mismatch; isolate run, no promotion'
   except OSError as error:reason='checkpoint unavailable; isolate this run: '+str(error)
  if row.get('alive'):
   cap=25000
   if row.get('execution_mode')=='PROVISIONAL_LONG':
    try:
     auth=RUNTIME.parent/'ac-master-dag-20260921/artifacts/2026-10-09_terl_mappo_batch01/commander_v3r2/provisional_long_authorization_20261010.json'
     grant=read(auth)['runs'][row['variant']]
     if digest(auth)!=row['authorization_sha256'] or grant['run_id']!=row['run_id'] or grant['authorized_end_step']!=row['authorized_end_step'] or grant['start_step']!=row['start_step']:raise ValueError('long authorization pin mismatch')
     cap=grant['authorized_end_step']-grant['start_step']
    except (OSError,KeyError,ValueError) as error:reason='invalid explicit long authorization: '+str(error)
   if row['step']>row['authorized_end_step'] or row['authorized_end_step']-row['start_step']>cap:reason='provisional budget exceeded'
   observed=resource_alerts(row,rows,leases,free,ram,free_gpu,compute,own)
   if observed:alerts.append({'time':now(),'run_id':row['run_id'],'pid':row['pid'],'action':'RESOURCE_ALERT_ONLY','reasons':observed})
   if row.get('scientific_quarantine'):reason='scientific provenance quarantined'
   if row.get('last_exception') or row.get('status','').startswith('FAILED'):reason='failed contract must stop'
   if reason and alive(row['pid'],row['run_id']):
    os.kill(row['pid'],signal.SIGTERM)
    events.append({'time':now(),'run_id':row['run_id'],'pid':row['pid'],'action':'SIGTERM_OWN_RUN','reason':reason,'observed_RAM':ram,'own_process_RSS_bytes':{str(p):process_rss(p) for p in own},'disk_free_bytes':free,'GPU_free_MiB':free_gpu})
  elif reason:events.append({'time':now(),'run_id':row['run_id'],'action':'ISOLATE_CHECKPOINT','reason':reason})
 if events:
  fd=(RUNTIME/'registry.lock').open('w');fcntl.flock(fd,fcntl.LOCK_EX)
  registry=read(DIRECTORY/'run_registry.json')
  for event in events:
   for row in registry.get('pilot_runs',[]):
    if row['run_id']==event['run_id']:row['supervisor_event']=event
  write(DIRECTORY/'run_registry.json',registry)
  with (RUNTIME/'supervisor_events.jsonl').open('a') as f:
   for event in events:f.write(json.dumps(event)+'\n')
  fcntl.flock(fd,fcntl.LOCK_UN);fd.close()
 subprocess.run([sys.executable,str(ROOT/'tools/batch01_status.py'),'--write'],check=True,stdout=subprocess.DEVNULL)
 _,rows=inspect_runs();snapshot={'time':now(),'pid':os.getpid(),'status':'MONITORING','automatic_restart':False,'automatic_budget_extension':False,
   'runtime_resource_stop_enabled':RUNTIME_RESOURCE_STOP_ENABLED,'startup_resource_gates_enabled':True,'resource_alerts':alerts,
   'independent_QA':state['QA_SAME_SHA'],'BASE_FROZEN':state['base_freeze_status']=='BASE_FROZEN','disk_free_bytes':free,'memory':ram,
   'GPU_processes':compute,'events':events,'own_process_RSS_bytes':{str(p):process_rss(p) for p in own},'attention_required':[{'run_id':r['run_id'],'status':r['status']} for r in rows if not r.get('scientific_quarantine') and (r.get('status','').startswith(('FAILED','STOPPED','PROVISIONAL_LONG_STARTING')) or any(e.get('status','').startswith('EVALUATION_') for e in r.get('evaluation',[])))],'runs':[{'run_id':r['run_id'],'status':r['status'],'step':r.get('step'),'pid':r.get('pid'),'alive':r.get('alive')} for r in rows]}
 queue_path=RUNTIME/'v3r2_long_queue.json'
 if queue_path.exists():
  queue=read(queue_path);snapshot['long_queue']=queue
  for label,item in queue.get('items',{}).items():
   if item.get('attention_required'):snapshot['attention_required'].append({'variant':label,'queue_status':item['status'],'last_exception':item.get('last_exception')})
  if queue.get('status')!='QUEUE_FINISHED' and (datetime.now(timezone.utc)-datetime.fromisoformat(queue['heartbeat'])).total_seconds()>90:snapshot['attention_required'].append({'queue_status':'HEARTBEAT_STALE','last_heartbeat':queue['heartbeat']})
 recovery_path=RUNTIME/'resource_recovery_queue.json'
 if recovery_path.exists():
  recovery=read(recovery_path);snapshot['resource_recovery_queue']=recovery
  for label,item in recovery.get('items',{}).items():
   if item.get('manual_review_required'):snapshot['attention_required'].append({'variant':label,'resource_recovery':item})
  if (datetime.now(timezone.utc)-datetime.fromisoformat(recovery['heartbeat'])).total_seconds()>90:snapshot['attention_required'].append({'resource_recovery':'HEARTBEAT_STALE'})
 with (RUNTIME/'resource_samples_20261010.jsonl').open('a') as f:f.write(json.dumps({k:snapshot[k] for k in ['time','memory','own_process_RSS_bytes','disk_free_bytes','events','runtime_resource_stop_enabled','resource_alerts']})+'\n')
 write(RUNTIME/'supervisor.json',snapshot);return snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--interval',type=int,default=30);a=p.parse_args()
 if not 10<=a.interval<=60:raise ValueError('supervision interval10..60 seconds')
 RUNTIME.mkdir(parents=True,exist_ok=True);fd=(RUNTIME/'supervisor.lock').open('w');fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);cache={}
 while True:
  try:supervise(cache)
  except Exception as error:
   write(RUNTIME/'supervisor_error.json',{'time':now(),'error':repr(error),'automatic_restart':False});raise
  if a.once:break
  time.sleep(a.interval)
if __name__=='__main__':main()
