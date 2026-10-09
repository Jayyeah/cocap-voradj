#!/usr/bin/env python3
"""One durable manager queues T1/C0 source-first gates and a single25k pilot.

No Codex agent, restarts, extended budgets, formal promotion or Stage3 training.
"""
import datetime,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
import batch01_v3r2_commander as m
R=m.R;QUEUE=R/'v3r2_secondary_queue.json';CENTRAL_QUEUE=m.D/'commander_v3r2/secondary_queue.json'
def snapshot(doc):
 doc['heartbeat']=m.now();doc['pid']=os.getpid();m.write(QUEUE,doc);m.write(CENTRAL_QUEUE,doc)
def state_gate():
 s=m.read(m.D/'batch_state.json')
 if s['core']['candidate_sha']!=m.CANDIDATE or s['core']['canonical_lock_sha256']!=m.PIN or s['core']['status']!='CORE_V3_SELFTEST_PASS' or s.get('new_run_hold') is not None:raise ValueError('current candidate no longer eligible')
 if s['QA_SAME_SHA']!='PENDING' or s['base_freeze_status']!='BASE_FREEZE_BLOCKED':raise ValueError('QA/freeze state changed; no automatic formal conversion')
def available():
 rows=m.compute();choices=[]
 uuids={int(x.split(',')[0]):x.split(',')[1].strip() for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
 for g,u in uuids.items():
  try:resource=m.gate_gpu(g);choices.append((sum(x['uuid']==u and x['uid']==os.getuid() for x in rows),g,resource))
  except ValueError:pass
 return sorted(choices)[0][1] if choices else None
def complete_metadata(label):
 # Status refresh and consistency checks occur before publishing a snapshot.
 subprocess.run([sys.executable,str(m.C/'tools/batch01_status.py'),'--write'],check=True,stdout=subprocess.DEVNULL)
 subprocess.run([sys.executable,str(m.C/'tools/validate_terl_mappo_batch01.py')],check=True,stdout=subprocess.DEVNULL)
 with (R/'registry.lock').open('w') as fd:
  fcntl.flock(fd,fcntl.LOCK_EX)
  subprocess.run(['git','add','artifacts/2026-10-09_terl_mappo_batch01','artifacts/2026-09-21_ac_master_dag/state.json'],cwd=m.C,check=True)
  if subprocess.run(['git','diff','--cached','--quiet'],cwd=m.C).returncode:
   subprocess.run(['git','commit','-m',f'ops(batch01): record queued {label} bounded pilot after source/resource gates'],cwd=m.C,check=True)
  subprocess.run(['git','push','origin','HEAD'],cwd=m.C,check=True)
  branch=subprocess.check_output(['git','branch','--show-current'],cwd=m.C,text=True).strip();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=m.C,text=True).strip();assert head==subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=m.C,text=True).split()[0]
def main():
 R.mkdir(parents=True,exist_ok=True);lock=(R/'v3r2_secondary_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if QUEUE.exists():raise ValueError('existing queue evidence; inspect before any restart')
 doc={'status':'WAITING_SHARED_GPU_SLOT','candidate_sha':m.CANDIDATE,'canonical_lock_sha256':m.PIN,'max_decisions_per_pilot':25000,'automatic_restart':False,'automatic_extension':False,'formal_training':False,'deadline':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)).isoformat(),'arms':{x:{'status':'CPU_PASS_WAIT_GPU_FOR_CUDA_BENCHMARK_EVALUATOR','run_id':'B01-'+x.upper()+'-PROV-V3R2-20261009','delta_hash':m.fp(m.read(m.delta(x))),'previous_pilot_quarantined':True} for x in ('t1','c0')}};snapshot(doc)
 for label in ('t1','c0'):
  try:
   while True:
    state_gate()
    if datetime.datetime.now(datetime.timezone.utc)>datetime.datetime.fromisoformat(doc['deadline']):raise TimeoutError('queue8hour deadline; no training budget extension')
    gpu=available()
    if gpu is not None:break
    doc['arms'][label]['status']='WAIT_GPU_MAX2_OWN_LINES_PER_SHARED_GPU';snapshot(doc);time.sleep(30)
   cpu=m.read(m.art(label)/'cpu_contract.json');d=m.read(m.delta(label));assert cpu['status']=='ARM_SELFTEST_PASS' and cpu['delta_hash']==m.fp(d)
   resource=m.gate_gpu(gpu);doc['arms'][label].update(status='CUDA_PREFLIGHT',physical_gpu=gpu,resource_gate=resource);snapshot(doc)
   m.run(label,'terl_mappo.batch01.arms.preflight',['--delta',str(m.delta(label)),'--device','cuda:0','--length','256','--output',str(m.art(label)/'cuda_contract.json')],gpu=gpu,log=m.art(label)/'cuda.txt')
   state_gate();doc['arms'][label]['status']='CONCURRENCY_BENCHMARK';snapshot(doc);m.benchmark([label],gpu)
   doc['arms'][label]['status']='EVALUATOR_SMOKE';snapshot(doc);m.evaluator_smoke(label)
   doc['arms'][label]['status']='QUALIFY_COMMIT_PUSH';snapshot(doc);m.qualify(label)
   state_gate();doc['arms'][label]['status']='PILOT_STARTING';snapshot(doc);m.launch(label,gpu)
   doc['arms'][label]['status']='PROVISIONAL_LAUNCHED_VERIFIED_FIRST_UPDATE';snapshot(doc);complete_metadata(label)
  except Exception:
   doc['arms'][label].update(status='QUEUE_BLOCKED_OR_FAILED_NO_AUTOMATIC_RETRY',last_exception=traceback.format_exc());snapshot(doc)
 doc['status']='QUEUE_FINISHED';snapshot(doc)
if __name__=='__main__':main()
