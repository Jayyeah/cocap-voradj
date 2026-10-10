#!/usr/bin/env python3
"""User-authorized long provisional continuation, with no formal promotion."""
import argparse,datetime,fcntl,json,os,shlex,subprocess,sys,time
from pathlib import Path
import batch01_v3r2_commander as common
B,C,D,R,PIN,CANDIDATE=common.B,common.C,common.D,common.R,common.PIN,common.CANDIDATE
read,write,fp,hf,now=common.read,common.write,common.fp,common.hf,common.now
AUTH=D/'commander_v3r2/provisional_long_authorization_20261010.json'
def root(label):return B/f'batch01-{label}-long-v3r2-20261010'
def delta(label):return root(label)/f'configs/experiments/terl_mappo_batch01_20261009/{label}_delta.json'
def art(label):return root(label)/'artifacts/2026-10-10_provisional_long'
common.root,common.delta,common.art=root,delta,art
command,env,gate_gpu=common.command,common.env,common.gate_gpu

def preflight(labels,device,gpu=None):
 resource=gate_gpu(gpu,len(labels)) if device.startswith('cuda') else None
 jobs=[]
 for label in labels:
  target=art(label)/('cuda' if device.startswith('cuda') else 'cpu');target.mkdir(parents=True,exist_ok=True)
  if (target/'contract.json').exists():raise ValueError('refuse proof overwrite')
  log=(target/'contract.txt').open('w')
  args=['--delta',str(delta(label)),'--device',device,'--output',str(target/'contract.json'),'--length','256' if device.startswith('cuda') else '8']
  p=subprocess.Popen(command(label,'terl_mappo.batch01.arms.long_preflight',args),cwd=root(label),env=env(label,gpu),stdout=log,stderr=subprocess.STDOUT);log.close();jobs.append((label,p,target))
 errors=[]
 for label,p,target in jobs:
  try:
   rc=p.wait(timeout=300)
   if rc:raise ValueError(f'{label} preflight exit{rc}; see {target}/contract.txt')
   q=read(target/'contract.json');assert q['status']=='LONG_OPERATIONAL_PREFLIGHT_PASS' and q['delta_hash']==fp(read(delta(label))) and q['checks']['new_bound_full_state_resume_exact'] and q['torch_peak_allocated_mib']<=2048
   if resource:write(target/'resource_gate.json',resource)
   print(label,device,'PASS',round(q['duration_seconds'],2),flush=True)
  except Exception as e:errors.append(str(e))
 if errors:raise ValueError('; '.join(errors))

def evaluators(label):
 target=art(label)/'cuda';proof=read(target/'contract.json');cp=proof['checkpoint']['path'];scenes=[1,2,3] if label=='t1' else [1]
 for scene in scenes:
  args=['--manifest',str(target/'manifest.json'),'--checkpoint',cp,'--output',str(art(label)/f'evaluator_stage{scene}.json'),'--domain','smoke','--horizon-cap','8']
  if label=='t1':args+=['--scene-stage',str(scene)]
  if not (art(label)/f'evaluator_stage{scene}.json').exists():common.run(label,'terl_mappo.batch01.arms.evaluator',args,log=art(label)/f'evaluator_stage{scene}.txt')
  result=read(art(label)/f'evaluator_stage{scene}.json');assert len(result['episodes'])==2 and result['seed_base']==2056100900+1000*(scene-1) and result['execution_mode']=='PROVISIONAL_LONG' and result['evaluation_protocol']['final']['seed_base']==2076100900
 # Synthetic metadata tests that even a matching receipt cannot open final.
 write(art(label)/'selected.json',{'checkpoint_sha256':hf(cp),'status':'SYNTHETIC_NEGATIVE_TEST_ONLY_NOT_SELECTION','formal_evidence':False})
 args=['--manifest',str(target/'manifest.json'),'--checkpoint',cp,'--output',str(art(label)/'FORBIDDEN_FINAL.json'),'--domain','final']
 log=art(label)/'final_reject.txt'
 with log.open('w') as f:q=subprocess.run(command(label,'terl_mappo.batch01.arms.evaluator',args),cwd=root(label),env=env(label),stdout=f,stderr=subprocess.STDOUT,timeout=60)
 assert q.returncode!=0 and 'cannot consume final holdout' in log.read_text() and not (art(label)/'FORBIDDEN_FINAL.json').exists()
 write(art(label)/'evaluator_gate.json',{'status':'PASS','scenes':scenes,'final_rejected':True,'final_consumed':False,'execution_mode':'PROVISIONAL_LONG'})

def qualify(label):
 d=read(delta(label));old=root(label)/'artifacts/2026-10-09_arm_preflight';grant=read(AUTH)['runs'][label]
 for device in ('cpu','cuda'):
  proof=read(art(label)/device/'contract.json');assert proof['status']=='LONG_OPERATIONAL_PREFLIGHT_PASS' and proof['delta_hash']==fp(d) and proof['authorization_sha256']==hf(AUTH)
  science=read(old/f'{device}_contract.json');assert science['status']=='ARM_SELFTEST_PASS' and science['checks']['full_state_resume_exact']
 assert read(art(label)/'evaluator_gate.json')['final_rejected']
 for change in d['source_changes']:assert hf(root(label)/change['path'])==change['after']
 gate={'status':'ARM_PROVISIONAL_LONG_GATES_PASS','authorization_sha256':hf(AUTH),'grant':grant,'candidate_sha':CANDIDATE,'lock_sha256':PIN,'delta_hash':fp(d),'science_CPU_CUDA_receipts':{'previous_head':grant['previous_head'],'science_code_unchanged_except_registered_operational_runner_and_evaluator':True},'full_state_migration_resume':'PASS','evaluator_and_final_blindness':'PASS','independent_QA':'QA_PENDING','BASE_FROZEN':False,'formal_evidence':False,'retroactive_promotion':False}
 write(art(label)/'gate_summary.json',gate)
 subprocess.run(['git','add','src/terl_mappo/batch01/arms','configs/experiments/terl_mappo_batch01_20261009',str(art(label).relative_to(root(label))),'artifacts/2026-10-09_arm_preflight'],cwd=root(label),check=True)
 subprocess.run(['git','commit','-m',f'feat(batch01): bind {label} explicit provisional long budget and strict continuation'],cwd=root(label),check=True)
 subprocess.run(['git','push','--set-upstream','origin','HEAD'],cwd=root(label),check=True)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root(label),text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=root(label),text=True).strip()
 assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=root(label),text=True).split()[0]==head
 print(label,'QUALIFIED_REMOTE',head,flush=True)

def launch(label,gpu):
 d=read(delta(label));gate=read(art(label)/'gate_summary.json');grant=read(AUTH)['runs'][label];assert gate['status']=='ARM_PROVISIONAL_LONG_GATES_PASS' and gate['delta_hash']==fp(d) and gate['authorization_sha256']==hf(AUTH)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root(label),text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=root(label),text=True).strip()
 assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=root(label),text=True).split()[0]==head
 run_id=grant['run_id'];out=root(label)/'runs'/run_id;session='b01_'+label.replace('-','_')+'_long_v3r2_20261010';start=grant['start_step'];end=grant['authorized_end_step'];proof=read(art(label)/'cuda/contract.json')
 with (R/'registry.lock').open('w') as fd:
  fcntl.flock(fd,fcntl.LOCK_EX);state=read(D/'batch_state.json');registry=read(D/'run_registry.json')
  assert state['core']['status']=='CORE_V3_SELFTEST_PASS' and state['core']['candidate_sha']==CANDIDATE and state.get('new_run_hold') is None and state['QA_SAME_SHA']=='PENDING' and state['base_freeze_status']=='BASE_FREEZE_BLOCKED'
  if run_id in {r['run_id'] for r in registry.get('pilot_runs',[])} or (out/'progress.json').exists() or subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('duplicate long run')
  resource=gate_gpu(gpu);lease_id=f'B01-LONG-{label.upper()}-20261010-GPU{gpu}'
  start_update=0
  if label!='t1':start_update=next(r['update'] for r in registry['pilot_runs'] if r['run_id']==grant['previous_run_id'])
  row={'run_id':run_id,'arm':d['line'],'variant':label,'execution_mode':'PROVISIONAL_LONG','status':'PROVISIONAL_LONG_STARTING','source_candidate_sha':CANDIDATE,'source_lock_sha256':PIN,'branch':branch,'head':head,'worktree':str(root(label)),'delta_path':str(delta(label)),'delta_hash':fp(d),'resolved_config':read(art(label)/'cuda/manifest.json')['resolved_config'],'output_root':str(out),'tmux':session,'physical_gpu':str(gpu),'gpu_uuid':resource['uuid'],'pid':None,'lease_id':lease_id,'start_step':start,'start_update':start_update,'authorized_end_step':end,'max_additional_joint_decisions':end-start,'authorization_sha256':hf(AUTH),'authorization_id':'USER-PROVISIONAL-LONG-20261010','formal_evidence':False,'retroactive_promotion':False,'checkpoint':None,'evaluation':[],'gates':gate,'registered_at':now(),'previous_run_id':grant.get('previous_run_id'),'parent_checkpoint':grant['parent_checkpoint'],'parent_checkpoint_sha256':grant.get('parent_checkpoint_sha256')}
  registry.setdefault('pilot_runs',[]).append(row)
  lease={'id':lease_id,'gpu':gpu,'gpu_uuid':resource['uuid'],'runs':[run_id],'maximum_compute_processes':resource['maximum_own_lines'],'maximum_additional_decisions_per_run':end-start,'expires_at':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=48)).isoformat(),'formal':False,'external_sharing_authorized':resource['external_sharing_authorized'],'max_output_gib_per_run':8,'keep_reserve_gib':30,'authorization_sha256':hf(AUTH)}
  state['pilot_gpu_leases'].append(lease);state['provisional_long_authorization']={'path':str(AUTH),'sha256':hf(AUTH),'id':'USER-PROVISIONAL-LONG-20261010','automatic_extension':False,'formal_evidence':False}
  write(D/'run_registry.json',registry);write(D/'batch_state.json',state);write(D/'commander_v3r2'/f'{label}_long_launch_resource_gate.json',resource)
 (out/'logs').mkdir(parents=True);(out/'tmp').mkdir()
 args=command(label,'terl_mappo.batch01.arms.runner',['--delta',str(delta(label)),'--run-id',run_id,'--output',str(out),'--device','cuda:0','--threads','1','--mode','PROVISIONAL_LONG','--decisions',str(end-start)])
 environment=env(label,gpu);environment['TMPDIR']=str(out/'tmp');keys=('CUDA_VISIBLE_DEVICES','PYTHONPATH','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TMPDIR')
 script=out/'launch.sh';script.write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(root(label)))+'\n'+'\n'.join('export '+k+'='+shlex.quote(environment[k]) for k in keys)+'\nexec '+shlex.join(args)+' >> '+shlex.quote(str(out/'logs/train.log'))+' 2>&1\n');script.chmod(0o755)
 gate_gpu(gpu);subprocess.run(['tmux','new-session','-d','-s',session,'bash '+shlex.quote(str(script))],check=True)
 deadline=time.monotonic()+300
 while time.monotonic()<deadline:
  if (out/'progress.json').exists():
   live=read(out/'progress.json')
   if live['last_exception']:raise ValueError('long run failed before first update '+live['last_exception'])
   if live['step']>start and live['update']>start_update and live.get('checkpoint') and live['checkpoint']['step']>start:
    write(D/'commander_v3r2'/f'{label}_long_actual_first_update.json',live)
    subprocess.run([sys.executable,str(C/'tools/batch01_status.py'),'--write'],check=True,stdout=subprocess.DEVNULL)
    print(json.dumps({'run_id':run_id,'PID':live['pid'],'GPU':gpu,'step':live['step'],'update':live['update'],'status':'PROVISIONAL_LONG_RUNNING'}),flush=True);return
  if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode:raise ValueError('training process exited before confirmation; inspect train.log')
  time.sleep(1)
 raise TimeoutError('no verified first update')

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preflight','evaluator','qualify','launch'],required=True);p.add_argument('--labels',nargs='+',required=True);p.add_argument('--device',default='cpu');p.add_argument('--gpu',type=int,choices=[0,1]);a=p.parse_args()
 if a.phase=='preflight':preflight(a.labels,a.device,a.gpu)
 else:
  for label in a.labels:
   if a.phase=='evaluator':evaluators(label)
   elif a.phase=='qualify':qualify(label)
   else:launch(label,a.gpu)
if __name__=='__main__':main()
