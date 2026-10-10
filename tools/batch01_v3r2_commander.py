#!/usr/bin/env python3
"""Single-commander source-first gates and bounded pilot launch; no agents."""
import argparse,datetime,fcntl,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
B=Path('/home/yjq/rl/CoCap1');C=B/'ac-master-dag-20260921';D=C/'artifacts/2026-10-09_terl_mappo_batch01';R=B/'batch01-commander-runtime'
PIN='384b9ba587b905b879a08f01fa1073ca470bdc0a836459067d193d97b03a84ab';CANDIDATE='9cc2d47c532c76239a61a0f6a19c8603358c92bf'
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def fp(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def hf(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def root(label):return B/f'batch01-{label}-v3r2-20261009'
def delta(label):return root(label)/f'configs/experiments/terl_mappo_batch01_20261009/{label}_delta.json'
def art(label):return root(label)/'artifacts/2026-10-09_arm_preflight'
def compute():
 rows=[]
 for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True).splitlines():
  pid,uuid,mem=[s.strip() for s in line.split(',')];p=Path('/proc')/pid
  if p.exists():rows.append({'pid':int(pid),'uuid':uuid,'memory':mem,'uid':p.stat().st_uid,'cmd':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()})
 return rows
def gate_gpu(gpu,adding=1):
 uuids={};free={}
 for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines():
  i,u,m=[s.strip() for s in line.split(',')];uuids[int(i)]=u;free[int(i)]=float(m)
 rows=compute();external={g:[r for r in rows if r['uuid']==u and r['uid']!=os.getuid()] for g,u in uuids.items()};idle=[g for g,u in uuids.items() if not any(r['uuid']==u for r in rows)]
 # Prefer an entirely idle GPU whenever one is available. Both external busy
 # GPUs permit at most two own compute lines each, including bounded tests.
 if idle and gpu not in idle:raise ValueError('idle GPU available; user preference requires it')
 active=[r for r in rows if r['uuid']==uuids[gpu] and r['uid']==os.getuid()]
 maximum=2 if all(external.values()) or external[gpu] else 4
 # A measured lease remains a cap after an external task exits.
 state=read(D/'batch_state.json');registry=read(D/'run_registry.json');active_ids={r['run_id'] for r in registry.get('pilot_runs',[]) if r.get('pid') in {x['pid'] for x in active}}
 for lease in state.get('pilot_gpu_leases',[]):
  if lease.get('gpu_uuid')==uuids[gpu] and active_ids.intersection(lease.get('runs',[])):
   maximum=min(maximum,lease.get('maximum_compute_processes',maximum))
 if len(active)+adding>maximum:raise ValueError('GPU own-line cap exceeded')
 if free[gpu]<4096+2048*adding:raise ValueError('GPU measured headroom insufficient')
 st=os.statvfs('/home/yjq');reserve=st.f_bavail*st.f_frsize
 if reserve<38*1024**3:raise ValueError('disk reserve insufficient')
 ram=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
 if ram<12*1024**3:raise ValueError('RAM reserve insufficient')
 return {'time':now(),'gpu':gpu,'uuid':uuids[gpu],'external_sharing_authorized':bool(external[gpu]),'maximum_own_lines':maximum,'protected_processes':rows,'free_gpu_MiB':free[gpu],'free_disk_bytes':reserve,'available_RAM_bytes':ram,'status':'RESOURCE_PASS'}
def env(label,gpu=None):
 p=root(label);tmp=p/'runs/tmp';tmp.mkdir(parents=True,exist_ok=True)
 return dict(os.environ,TMPDIR=str(tmp),CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(p/'src')+':/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/.runtime-deps')
def command(label,module,args):return [sys.executable,'-S',str(root(label)/'tools/batch01_python.py'),'--lock-sha',PIN,'--delta',str(delta(label)),'--python-args','-m',module,*args]
def run(label,module,args,gpu=None,log=None):
 with Path(log).open('w') as f:subprocess.run(command(label,module,args),cwd=root(label),env=env(label,gpu),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
def benchmark(labels,gpu):
 resource=gate_gpu(gpu,len(labels));jobs=[];started=time.monotonic()
 for label in labels:
  out=root(label)/'runs/preflight/v3r2_shared_benchmark';out.parent.mkdir(parents=True,exist_ok=True)
  if (out/'progress.json').exists():raise ValueError('refuse benchmark overwrite')
  log=(art(label)/'benchmark.txt').open('w');p=subprocess.Popen(command(label,'terl_mappo.batch01.arms.runner',['--delta',str(delta(label)),'--run-id',f'B01-{label}-V3R2-BENCH','--output',str(out),'--device','cuda:0','--threads','1','--mode','BENCHMARK','--decisions','512','--no-evaluation']),cwd=root(label),env=env(label,gpu),stdout=log,stderr=subprocess.STDOUT);log.close();jobs.append((label,p,out))
 errors=[]
 for label,p,out in jobs:
  try:
   rc=p.wait(timeout=300)
   if rc:raise ValueError('benchmark failed '+label)
   progress=read(out/'progress.json');metrics=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
   assert progress['status']=='COMPLETE_BENCHMARK' and progress['step']-progress['start_step']==512 and not progress['last_exception']
   assert progress['torch_peak_allocated_mib']<=2048 and metrics and all(m['metrics']['minibatch_updates']>0 for m in metrics)
   write(art(label)/'gpu_benchmark.json',{'status':'SHARED_GPU_CONCURRENCY_PASS','resource_gate':resource,'progress':progress,'metrics':metrics,'concurrency':len(labels),'common_window_seconds':time.monotonic()-started,'formal_evidence':False})
  except Exception as e:errors.append(str(e))
 if errors:raise ValueError('; '.join(errors))
def evaluator_smoke(label):
 out=root(label)/'runs/preflight/v3r2_shared_benchmark';progress=read(out/'progress.json');cp=progress['checkpoint']['path'];scenes=['stage1','stage2','stage3'] if label=='t1' else ['continuous'] if label=='c0' else ['stage1']
 for scene in scenes:
  target=art(label)/f'evaluator_{scene}_smoke.json'
  args=['--manifest',str(out/'manifest.json'),'--checkpoint',cp,'--output',str(target),'--domain','smoke','--horizon-cap','8']
  if label=='t1':args+=['--scene-stage',str({'stage1':1,'stage2':2,'stage3':3}[scene])]
  run(label,'terl_mappo.batch01.arms.evaluator',args,log=art(label)/f'evaluator_{scene}.txt')
  r=read(target);offset={'stage1':0,'continuous':0,'stage2':1000,'stage3':2000}[scene]
  assert r['seed_domain']=='smoke' and r['seed_base']==2056100900+offset and len(r['episodes'])==2
  assert set(r['modes'])==({'mean','sample'} if label=='c0' else {'argmax','sample'})
  assert r['evaluation_protocol']['final']['seed_base']==2076100900
 write(art(label)/'evaluator_gate.json',{'status':'PASS','scenes':scenes,'seed_contract':'NEW_PAIRED_SCREEN2056100900_SELECTION2066100900_FINAL2076100900_OFFSET100000_SCENES0_1000_2000','final_consumed':False})
def qualify(label):
 d=read(delta(label));pin=fp(d);assert d['base']['candidate_sha']==CANDIDATE and d['base']['lock_sha256']==PIN
 for file in ('cpu_contract.json','cuda_contract.json'):
  q=read(art(label)/file);assert q['status']=='ARM_SELFTEST_PASS' and q['delta_hash']==pin and q['checks']['full_state_resume_exact']
 for change in d['source_changes']:assert hf(root(label)/change['path'])==change['after']
 assert read(art(label)/'gpu_benchmark.json')['status']=='SHARED_GPU_CONCURRENCY_PASS'
 assert read(art(label)/'evaluator_gate.json')['status']=='PASS'
 write(art(label)/'gate_summary.json',{'status':'ARM_PROVISIONAL_GATES_PASS','independent_QA':'QA_PENDING','BASE_FROZEN':False,'candidate_sha':CANDIDATE,'canonical_lock_sha256':PIN,'delta_hash':pin,'science_CPU_CUDA_resume_evaluator_resource':'PASS','maximum_additional_joint_decisions':25000,'previous_pilot':'QUARANTINED_NO_REUSE','formal_evidence':False})
 subprocess.run(['git','add','artifacts/2026-10-09_arm_preflight'],cwd=root(label),check=True)
 subprocess.run(['git','commit','-m',f'test(batch01): qualify {label} source-first shared-GPU bounded pilot'],cwd=root(label),check=True)
 subprocess.run(['git','push','--set-upstream','origin','HEAD'],cwd=root(label),check=True)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root(label),text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=root(label),text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=root(label),text=True).split()[0];assert head==remote
 return head,branch

def launch(label,gpu):
 d=read(delta(label));gates=read(art(label)/'gate_summary.json');assert gates['status']=='ARM_PROVISIONAL_GATES_PASS' and gates['delta_hash']==fp(d)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root(label),text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=root(label),text=True).strip();assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=root(label),text=True).split()[0]==head
 run_id='B01-'+label.upper()+'-PROV-V3R2-20261009';out=root(label)/'runs'/run_id;session='b01_'+label.replace('-','_')+'_prov_v3r2_20261009'
 R.mkdir(parents=True,exist_ok=True)
 with (R/'registry.lock').open('w') as fd:
  fcntl.flock(fd,fcntl.LOCK_EX);state=read(D/'batch_state.json');registry=read(D/'run_registry.json')
  assert state['core']['status']=='CORE_V3_SELFTEST_PASS' and state['core']['candidate_sha']==CANDIDATE and state.get('new_run_hold') is None
  assert state['base_freeze_status']=='BASE_FREEZE_BLOCKED' and state['QA_SAME_SHA']=='PENDING'
  if run_id in {r['run_id'] for r in registry.get('pilot_runs',[])} or (out/'progress.json').exists() or subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('duplicate run')
  resource=gate_gpu(gpu);start=775000 if label.startswith('p1-') else 0;lease_id=f'B01-V3R2-{label.upper()}-GPU{gpu}-SHARED-PROVISIONAL'
  row={'run_id':run_id,'arm':d['line'],'variant':label,'execution_mode':'PROVISIONAL','status':'PROVISIONAL_STARTING','source_candidate_sha':CANDIDATE,'source_lock_sha256':PIN,'branch':branch,'head':head,'worktree':str(root(label)),'delta_path':str(delta(label)),'delta_hash':fp(d),'resolved_config':read(art(label)/'cpu_contract.json')['active_values']['ppo'],'output_root':str(out),'tmux':session,'physical_gpu':str(gpu),'gpu_uuid':resource['uuid'],'pid':None,'lease_id':lease_id,'start_step':start,'start_update':read(root(label)/'runs/preflight/v3r2_shared_benchmark/progress.json')['update']-2,'authorized_end_step':start+25000,'max_additional_joint_decisions':25000,'formal_evidence':False,'retroactive_promotion':False,'checkpoint':None,'evaluation':[],'gates':gates,'registered_at':now(),'replaces_quarantined_run':'B01-'+label.upper()+'-PROV-V3-20261009','previous_weights_reused':False}
  registry.setdefault('pilot_runs',[]).append(row)
  lease=next((l for l in state['pilot_gpu_leases'] if l['id']==lease_id),None)
  if lease is None:
   lease={'id':lease_id,'gpu':gpu,'gpu_uuid':resource['uuid'],'runs':[],'maximum_compute_processes':resource['maximum_own_lines'],'maximum_additional_decisions_per_run':25000,'expires_at':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)).isoformat(),'formal':False,'external_sharing_authorized':resource['external_sharing_authorized'],'max_output_gib_per_run':8,'keep_reserve_gib':30};state['pilot_gpu_leases'].append(lease)
  lease['runs'].append(run_id);write(D/'run_registry.json',registry);write(D/'batch_state.json',state);write(D/'commander_v3r2'/f'{label}_launch_resource_gate.json',resource)
 (out/'logs').mkdir(parents=True);(out/'tmp').mkdir()
 args=command(label,'terl_mappo.batch01.arms.runner',['--delta',str(delta(label)),'--run-id',run_id,'--output',str(out),'--device','cuda:0','--threads','1','--mode','PROVISIONAL','--decisions','25000'])
 launch_script=out/'launch.sh';environment=env(label,gpu);environment['TMPDIR']=str(out/'tmp');selected=('CUDA_VISIBLE_DEVICES','PYTHONPATH','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TMPDIR')
 launch_script.write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(root(label)))+'\n'+'\n'.join('export '+k+'='+shlex.quote(environment[k]) for k in selected)+'\nexec '+shlex.join(args)+' >> '+shlex.quote(str(out/'logs/train.log'))+' 2>&1\n');launch_script.chmod(0o755)
 # Recheck immediately before starting the process; metadata is still STARTING.
 gate_gpu(gpu);subprocess.run(['tmux','new-session','-d','-s',session,'bash '+shlex.quote(str(launch_script))],check=True)
 deadline=time.monotonic()+300
 while time.monotonic()<deadline:
  if (out/'progress.json').exists():
   live=read(out/'progress.json')
   if live['last_exception']:raise ValueError('pilot failed before launch confirmation '+live['last_exception'])
   if live['step']>start and live['update']>row['start_update'] and live.get('checkpoint'):
    write(D/'commander_v3r2'/f'{label}_actual_first_update.json',{'run_id':run_id,'time':now(),'PID':live['pid'],'step':live['step'],'update':live['update'],'checkpoint':live['checkpoint'],'source':d['base'],'formal_evidence':False})
    subprocess.run([sys.executable,str(C/'tools/batch01_status.py'),'--write'],check=True);print(json.dumps({'run_id':run_id,'PID':live['pid'],'GPU':gpu,'step':live['step'],'update':live['update'],'status':'PROVISIONAL_RUNNING'}),flush=True);return
  time.sleep(1)
 raise TimeoutError('no verified first update; registry remains STARTING')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['benchmark','evaluator','qualify','launch'],required=True);ap.add_argument('--labels',nargs='+',required=True);ap.add_argument('--gpu',type=int,choices=[0,1]);a=ap.parse_args()
 if a.phase=='benchmark':benchmark(a.labels,a.gpu)
 else:
  for label in a.labels:
   if a.phase=='evaluator':evaluator_smoke(label)
   elif a.phase=='qualify':qualify(label)
   else:launch(label,a.gpu)
if __name__=='__main__':main()
