import argparse,hashlib,json,os,shlex,subprocess,time,fcntl
from pathlib import Path
from datetime import datetime,timezone,timedelta
B=Path('/home/yjq/rl/CoCap1');C=B/'ac-master-dag-20260921';D=C/'artifacts/2026-10-09_terl_mappo_batch01';R=B/'batch01-commander-runtime'
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def fp(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def hashfile(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def owned(row):
 try:
  p=Path(f"/proc/{row.get('pid')}");cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  return p.stat().st_uid==os.getuid() and 'terl_mappo.batch01.arms.runner' in cmd and row['run_id'] in cmd
 except FileNotFoundError:return False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--label',choices=['t1','c0'],required=True);a=ap.parse_args();label=a.label;line=label.upper()
 root=B/f'batch01-{label}-v3-20261009';art=root/'artifacts/2026-10-09_arm_preflight';bench=root/'runs/preflight/gpu1_benchmark_final'
 delta_path=root/f'configs/experiments/terl_mappo_batch01_20261009/{label}_delta.json';delta=read(delta_path);pin=fp(delta)
 cpu=read(art/'final_cpu_contract.json');cuda=read(art/'cuda_contract.json');progress=read(bench/'progress.json')
 assert cpu['status']==cuda['status']=='ARM_SELFTEST_PASS' and cpu['delta_hash']==cuda['delta_hash']==progress['delta_hash']==pin
 assert progress['status']=='COMPLETE_BENCHMARK' and progress['physical_gpu']=='1' and not progress['last_exception'] and progress['torch_peak_allocated_mib']<=2048
 metrics=[json.loads(r) for r in (bench/'metrics.jsonl').read_text().splitlines()];assert metrics and all(x['metrics']['minibatch_updates']>0 for x in metrics)
 scenes=[('stage1','3P1E0O4C',0),('stage2','4P1E1O6C',1000),('stage3','7P2E2O8C',2000)] if line=='T1' else [('continuous','3P1E0O4C',0)]
 evaluator={}
 for name,scene,offset in scenes:
  result=read(art/f'evaluator_{name}_smoke.json')
  assert result['seed_domain']=='smoke' and result['seed_base']==2056100900+offset and result['scene']==scene and len(result['episodes'])==2
  assert set(result['modes'])==({'mean','sample'} if line=='C0' else {'argmax','sample'})
  evaluator[name]={'scene':scene,'seed_base':result['seed_base'],'status':'PASS'}
 for source in delta['source_changes']:assert source['after']==hashfile(root/source['path'])
 write(art/'gpu1_benchmark.json',{'status':'PROVISIONAL_CONCURRENCY_PASS','progress':progress,'metrics':metrics,'formal_evidence':False,'evaluator':evaluator})
 write(art/'gate_summary.json',{'status':'ARM_PROVISIONAL_GATES_PASS','formal_gate':'BASE_INDEPENDENT_QA_PENDING','delta_hash':pin,'CPU':'PASS','CUDA':'PASS','exact_resume':'PASS','evaluator':evaluator,'candidate':delta['base'],'gpu1_concurrency_benchmark':'PASS','mode':'PROVISIONAL','max_additional_decisions':25000})
 subprocess.run(['git','add','src/terl_mappo/batch01/arms',str(delta_path.relative_to(root)),'artifacts/2026-10-09_arm_preflight',f'docs/BATCH01_{line}_V3_PILOT_20261009_ZH.md'],cwd=root,check=True)
 subprocess.run(['git','commit','-m',f'feat(batch01): implement {line} physical curriculum contracts and bounded pilot gates'],cwd=root,check=True)
 subprocess.run(['git','-c','http.proxy=http://127.0.0.1:17892','push','--set-upstream','origin','HEAD'],cwd=root,check=True)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()
 remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17892','ls-remote','origin','refs/heads/'+branch],cwd=root,text=True).split()[0];assert head==remote
 run_id=f'B01-{line}-PROV-V3-20261009';out=root/'runs'/run_id;session=f'b01_{label}_prov_v3_20261009'
 if (out/'progress.json').exists() or subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('duplicate run/session')
 R.mkdir(parents=True,exist_ok=True);fd=(R/'registry.lock').open('w');fcntl.flock(fd,fcntl.LOCK_EX)
 registry=read(D/'run_registry.json');state=read(D/'batch_state.json');assert state['pilot_authorization']['maximum_additional_joint_decisions_per_run']==25000 and state['base_freeze_status']=='BASE_FREEZE_BLOCKED'
 assert run_id not in {r['run_id'] for r in registry.get('pilot_runs',[])}
 active=[r for r in registry.get('pilot_runs',[]) if owned(r)];assert len(active)<2
 allowed={r['pid'] for r in active};uuid=progress['gpu_uuid'];compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
 assert not any(uuid in row and int(row.split(',')[0]) not in allowed for row in compute.splitlines()),'GPU1 external occupied'
 stats=os.statvfs('/home/yjq');assert stats.f_bavail*stats.f_frsize>62*1024**3
 row={'run_id':run_id,'arm':line,'variant':label,'status':'PROVISIONAL_STARTING','execution_mode':'PROVISIONAL','max_additional_joint_decisions':25000,
  'start_step':0,'authorized_end_step':25000,'start_update':0,'formal_evidence':False,'retroactive_promotion':False,
  'source_candidate_sha':delta['base']['candidate_sha'],'source_lock_sha256':delta['base']['lock_sha256'],'branch':branch,'head':head,
  'delta_hash':pin,'delta_path':str(delta_path),'resolved_config':cpu['active_values']['ppo'],'worktree':str(root),'output_root':str(out),
  'tmux':session,'physical_gpu':'1','gpu_uuid':uuid,'pid':None,'lease_id':'B01-COMMANDER-V3-GPU1-SECONDARY2-PROVISIONAL',
  'gates':{'source':'PASS','arm_science_CPU_CUDA':'PASS','runtime_consumer':'PASS','full_state_resume':'PASS','evaluation_seed_protocol':'PASS_SELFTEST_QA_PENDING','resource_gpu1_two_way':'PASS','independent_BASE_QA':'QA_PENDING','BASE_FREEZE':'BLOCKED'},
  'checkpoint':None,'evaluation':[],'last_exception':None,'registered_at':datetime.now(timezone.utc).isoformat()}
 registry.setdefault('pilot_runs',[]).append(row);write(D/'run_registry.json',registry)
 for lease in state['pilot_gpu_leases']:
  if lease['id']=='B01-COMMANDER-V3-GPU1-4WAY-PROVISIONAL':lease.update(training_released=True,training_release_reason='all four pilots reached hardcap; CPU screens remain separate')
 lease=next((x for x in state['pilot_gpu_leases'] if x['id']==row['lease_id']),None)
 if lease is None:
  lease={'id':row['lease_id'],'gpu':1,'gpu_uuid':uuid,'runs':[],'maximum_compute_processes':2,'maximum_additional_decisions_per_run':25000,'expires_at':(datetime.now(timezone.utc)+timedelta(hours=8)).isoformat(),'formal':False,'keep_reserve_gib':30,'max_output_gib_per_run':8};state['pilot_gpu_leases'].append(lease)
 lease['runs'].append(run_id);write(D/'batch_state.json',state)
 write(D/'commander_v3'/f'{label}_pilot_launch_gate.json',{'status':'PASS','head':head,'remote_head':remote,'delta_hash':pin,'protected_compute':compute,'free_bytes':stats.f_bavail*stats.f_frsize,'gates':row['gates'],'formal':False,'maximum_additional_decisions':25000})
 fcntl.flock(fd,fcntl.LOCK_UN)
 (out/'tmp').mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True);q=shlex.quote
 script='#!/bin/bash\nset -euo pipefail\ncd '+q(str(root))+'\nexport CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\nexport PYTHONPATH='+q(str(root/'src')+':'+str(root)+':/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/.runtime-deps')+'\nexport TMPDIR='+q(str(out/'tmp'))+'\nexec python -m terl_mappo.batch01.arms.runner --delta '+q(str(delta_path))+' --run-id '+q(run_id)+' --output '+q(str(out))+' --mode PROVISIONAL --device cuda:0 --threads 1 --decisions 25000 >> '+q(str(out/'logs/train.log'))+' 2>&1\n'
 (out/'launch.sh').write_text(script);os.chmod(out/'launch.sh',0o755)
 subprocess.run(['tmux','new-session','-d','-s',session,'bash '+q(str(out/'launch.sh'))],check=True);print('LAUNCHED',run_id,flush=True)
 end=time.monotonic()+120
 while time.monotonic()<end:
  if (out/'progress.json').exists():
   live=read(out/'progress.json')
   if live['status'].startswith('FAILED'):raise ValueError(live['last_exception'])
   if live['step']>0 and live['update']>0:
    assert live['checkpoint'] and hashfile(live['checkpoint']['path'])==live['checkpoint']['sha256']
    fcntl.flock(fd,fcntl.LOCK_EX);registry=read(D/'run_registry.json')
    r=next(r for r in registry['pilot_runs'] if r['run_id']==run_id);r.update({k:v for k,v in live.items() if k not in ('base','head','branch','delta_hash')});r['actual_process_head']=live['head'];write(D/'run_registry.json',registry);fcntl.flock(fd,fcntl.LOCK_UN)
    subprocess.run(['python',str(C/'tools/batch01_status.py'),'--write'],check=True)
    print('VERIFIED_REAL_PPO_UPDATE',run_id,'PID',live['pid'],'GPU',live['gpu_uuid'],'step',live['step'],'checkpoint',live['checkpoint']['path'],flush=True);return
  time.sleep(1)
 raise ValueError('first-update proof timed out; leave status STARTING, never claim RUNNING')
if __name__=='__main__':main()
