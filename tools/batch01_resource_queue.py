#!/usr/bin/env python3
"""Recover only resource-paused runs within their existing explicit budgets."""
import datetime,fcntl,os,time,traceback
from pathlib import Path
import batch01_resource_resume as recovery
m=recovery.m
QUEUE=m.R/'resource_recovery_queue.json'
ORDER=('p1-control','p1-treatment','r1','n1','t1','c0')
def ram_available():
 return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
def main():
 lock=(m.R/'resource_recovery_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 doc=m.read(QUEUE) if QUEUE.exists() else {'schema':'batch01.resource_only_recovery.v1','items':{},'scientific_failure_restart':False,'budget_extension':False,'formal_evidence':False,'minimum_RAM_available_bytes':24*1024**3,'required_stable_samples':3,'minimum_seconds_after_resource_stop':90}
 stable=0
 while True:
  registry,rows=recovery.inspect_runs();current={r['variant']:r for r in rows if r.get('execution_mode')=='PROVISIONAL_LONG'}
  available=ram_available();stable=stable+1 if available>=doc['minimum_RAM_available_bytes'] else 0
  doc.update(pid=os.getpid(),heartbeat=m.now(),status='RESOURCE_ONLY_MONITORING',RAM_available_bytes=available,stable_RAM_samples=stable)
  for label in ORDER:
   row=current.get(label)
   if row is None:continue
   prior=doc['items'].get(label,{})
   if row.get('alive'):doc['items'][label]={'status':row['status'],'PID':row['pid'],'GPU':row.get('physical_gpu'),'step':row.get('step')};continue
   if row.get('step',0)>=row['authorized_end_step'] and not row.get('last_exception'):doc['items'][label]={'status':row['status'],'step':row.get('step')};continue
   if not recovery.eligible(row):doc['items'][label]={'status':'BLOCKED_NO_SCIENTIFIC_FAILURE_RESTART','run_status':row['status']};continue
   if prior.get('manual_review_required'):continue
   event=row['supervisor_event'];elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(event['time'])).total_seconds()
   gpu=recovery.choose(label) if stable>=3 and elapsed>=90 else None
   if gpu is None:
    doc['items'][label]={'status':'WAITING_RESOURCE_RECOVERY_SLOT','restore_checkpoint_step':row['checkpoint']['step'],'reason':'RAM needs24GiB sustained headroom; shared GPU maximum2 own lines'};continue
   attempts=row.get('resource_resume_history',[])
   recent=[x for x in attempts if (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(x['time'])).total_seconds()<3600]
   if len(recent)>=3:doc['items'][label]={'status':'RESOURCE_FLAPPING_REVIEW_REQUIRED','manual_review_required':True};continue
   doc['items'][label]={'status':'STRICT_RESOURCE_RESUME_IN_PROGRESS','GPU':gpu};m.write(QUEUE,doc)
   try:
    recovery.resume(label,gpu);doc['items'][label]={'status':'STRICT_RESOURCE_RESUME_FIRST_UPDATE_PASS','GPU':gpu,'time':m.now()}
    # One launch per cycle; fresh RAM and GPU samples precede the next launch.
    m.write(QUEUE,doc);break
   except Exception:
    doc['items'][label]={'status':'RESUME_FAILED_NO_AUTOMATIC_RETRY','last_exception':traceback.format_exc(),'manual_review_required':True}
  m.write(QUEUE,doc);time.sleep(30)
if __name__=='__main__':main()
