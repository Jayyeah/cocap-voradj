#!/usr/bin/env python3
"""Durable T1/C0 queue under explicit budgets and live shared GPU gates."""
import fcntl,os,subprocess,sys,time,traceback
from pathlib import Path
import batch01_long_commander as m
QUEUE=m.R/'v3r2_long_queue.json'
def state(doc):doc['heartbeat']=m.now();doc['pid']=os.getpid();m.write(QUEUE,doc)
def choose():
 candidates=[]
 for gpu in (0,1):
  try:g=m.gate_gpu(gpu);candidates.append(g)
  except ValueError:pass
 if not candidates:return None
 return max(candidates,key=lambda x:x['free_gpu_MiB'])['gpu']
def main():
 fd=(m.R/'long_queue.lock').open('w');fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 doc=m.read(QUEUE) if QUEUE.exists() else {'status':'QUEUE_MONITORING','automatic_budget_extension':False,'automatic_restart_failed_runs':False,'independent_QA':'QA_PENDING','BASE_FROZEN':False,'formal_evidence':False,'items':{x:{'status':'WAITING_SHARED_GPU_SLOT'} for x in ('t1','c0')}}
 for label,item in doc['items'].items():
  if item['status'] in {'PROVISIONAL_LONG_LAUNCHED','FAILED_NO_AUTOMATIC_RETRY'}:continue
  while True:
   gpu=choose();state(doc)
   if gpu is not None:break
   item.update(status='WAITING_SHARED_GPU_SLOT',reason='both GPUs occupied; shared GPU maximum2 own compute lines');time.sleep(30)
  try:
   item.update(status='CUDA_GATE_ACTIVE',gpu=gpu);state(doc)
   if not (m.art(label)/'cuda/contract.json').exists():m.preflight([label],'cuda:0',gpu)
   if not (m.art(label)/'evaluator_gate.json').exists():m.evaluators(label)
   if not (m.art(label)/'gate_summary.json').exists():m.qualify(label)
   item['status']='GATES_PASS_WAITING_SLOT';state(doc)
   # Capacity is rechecked after preflight and remote delivery.
   while True:
    gpu=choose();state(doc)
    if gpu is not None:break
    time.sleep(30)
   m.launch(label,gpu);item.update(status='PROVISIONAL_LONG_LAUNCHED',gpu=gpu,time=m.now())
  except Exception:
   item.update(status='FAILED_NO_AUTOMATIC_RETRY',last_exception=traceback.format_exc(),attention_required=True)
   doc['attention_required']=True
  state(doc)
 doc['status']='QUEUE_FINISHED';state(doc)
if __name__=='__main__':main()
