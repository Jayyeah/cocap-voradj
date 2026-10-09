#!/usr/bin/env python3
"""Persistent bounded-run supervision. No launch, restart or budget extension."""
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

def now():return datetime.now(timezone.utc).isoformat()
def memory():
 return {x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:','MemTotal:'))}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def supervise(cache):
 registry,rows=inspect_runs();state=read(DIRECTORY/'batch_state.json');leases={l['id']:l for l in state.get('pilot_gpu_leases',[])}
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True).strip()
 stat=os.statvfs('/home/yjq');free=stat.f_bavail*stat.f_frsize;ram=memory();events=[];own={r.get('pid') for r in rows if r.get('alive')}
 for row in rows:
  if row.get('mode',row.get('execution_mode'))!='PROVISIONAL':continue
  reason=None;cp=row.get('checkpoint')
  if cp:
   try:
    path=Path(cp['path']).resolve();root=Path(row['output_root']).resolve();s=path.stat();key=(str(path),s.st_size,s.st_mtime_ns)
    if not path.is_relative_to(root/'checkpoints'):reason='checkpoint escapes registered run'
    elif key not in cache:cache[key]=digest(path)
    if not reason and cache[key]!=cp['sha256']:reason='checkpoint hash mismatch; isolate run, no promotion'
   except OSError as error:reason='checkpoint unavailable; isolate this run: '+str(error)
  if row.get('alive'):
   if row['step']>row['authorized_end_step'] or row['authorized_end_step']-row['start_step']>25000:reason='provisional budget exceeded'
   if free<30*1024**3:reason='disk reserve below30GiB'
   if row.get('disk_bytes',0)>8*1024**3:reason='run output above8GiB'
   if row.get('torch_peak_allocated_mib',0)>2048:reason='GPU lease above2GiB'
   if ram['MemAvailable']<8*1024**3:reason='RAM reserve below8GiB'
   if row.get('scientific_quarantine'):reason='scientific provenance quarantined'
   if row.get('last_exception') or row.get('status','').startswith('FAILED'):reason='failed contract must stop'
   lease=leases.get(row['lease_id'])
   if not lease or datetime.fromisoformat(lease['expires_at'])<datetime.now(timezone.utc):reason='GPU lease missing/expired'
   for line in compute.splitlines():
    if row['gpu_uuid'] in line and int(line.split(',')[0]) not in own and not (lease and lease.get('external_sharing_authorized')):reason='new external process on exclusive leased GPU; yield own training'
   if lease and lease.get('external_sharing_authorized') and sum(1 for r in rows if r.get('alive') and r.get('gpu_uuid')==row['gpu_uuid'])>2:reason='shared GPU exceeds user maximum2 own lines'
   if reason and alive(row['pid'],row['run_id']):
    os.kill(row['pid'],signal.SIGTERM)
    events.append({'time':now(),'run_id':row['run_id'],'pid':row['pid'],'action':'SIGTERM_OWN_RUN','reason':reason})
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
   'independent_QA':state['QA_SAME_SHA'],'BASE_FROZEN':state['base_freeze_status']=='BASE_FROZEN','disk_free_bytes':free,'memory':ram,
   'GPU_processes':compute,'events':events,'runs':[{'run_id':r['run_id'],'status':r['status'],'step':r.get('step'),'pid':r.get('pid'),'alive':r.get('alive')} for r in rows]}
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
