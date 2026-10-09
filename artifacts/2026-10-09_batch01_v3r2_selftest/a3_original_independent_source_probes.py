"""Independent V2 source-contract probes in disposable exact-candidate archives.

Source code edits and generated caches affect only ignored QA snapshots. No
Core/T0 files are edited. CPU-only, bounded diagnostics; each child is fresh.
"""
import io,json,os,subprocess,sys,tarfile,time
from pathlib import Path
CANDIDATE='863a0cf55aca0ace8a0aaab36d9166aa4c97268f'
PIN='885ec7b8617cf88e2537b08ebf041e1d8bfc41f120b7ec4aef1708c4a513a8a4'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CHILD=r'''
import copy,importlib,json,os,sys,time,py_compile,statistics
from pathlib import Path
import numpy as np
import torch
from terl_mappo.batch01.contracts import ROOT,LOCK_PATH,fingerprint,verify_base,declared_sources
from terl_mappo.batch01.source_guard import SourceGuard,SourceViolation
from terl_mappo.batch01.interfaces import Hooks,assemble
from terl_mappo.batch01.provenance import runtime_manifest
from terl_mappo.batch01.checkpoints import save_bound,load_bound
from terl_mappo.run import file_hash
case=sys.argv[1]; torch.set_num_threads(1)
lock=json.loads((ROOT/LOCK_PATH).read_text())
name='cocap_voradj.models.continuous.qa_dynamic_dep'
p=ROOT/('src/'+name.replace('.','/')+'.py')
marker=ROOT/'runs/dependency_executed.txt'
source='from pathlib import Path\nPath('+repr(str(marker))+').write_text("yes")\nVALUE = 0.4\n'
p.write_text(source)
ext=ROOT/'src/terl_mappo/batch01/extensions/qa_reward_probe.py';ext.parent.mkdir(parents=True,exist_ok=True)
expression=repr(name) if case=='literal_undeclared' else "parameters['module']"
ext.write_text('import importlib\nfrom terl_mappo.batch01.interfaces import Hooks\n'
 'def factory(parameters):\n    def reward(raw,info):\n        m=importlib.import_module('+expression+')\n        return raw+m.VALUE\n    return Hooks(reward_transform=reward)\n')
changes=[{'path':str(ext.relative_to(ROOT)),'before':None,'after':file_hash(ext),'reason':'independent QA reward hook','science_impact':'diagnostic only'}]
legal=case not in {'computed_undeclared','literal_undeclared','late_undeclared','optimizer_import'}
if legal:changes.append({'path':str(p.relative_to(ROOT)),'before':None,'after':file_hash(p),'reason':'explicit dynamic dependency','science_impact':'diagnostic only'})
delta={'schema':'terl.batch01.delta.v1','line':'R1','base':{'parent_sha':lock['parent_sha'],'candidate_sha':'863a0cf55aca0ace8a0aaab36d9166aa4c97268f','lock_sha256':fingerprint(lock)},'config_changes':[],'source_changes':changes,'extensions':{'entrypoint':'terl_mappo.batch01.extensions.qa_reward_probe:factory','parameters':{'module':name}}}
resolved=verify_base(lock,delta,fingerprint(lock))
guard=SourceGuard.from_base(lock,delta,fingerprint(lock),diagnostics=ROOT/'runs/source_failures')
with guard.scope():hooks=importlib.import_module('terl_mappo.batch01.extensions.qa_reward_probe').factory({'module':name})
record={'case':case,'candidate_sha':delta['base']['candidate_sha'],'canonical_lock_sha256':fingerprint(lock)}
def rejected(fn):
 try:fn()
 except SourceViolation as e:return str(e)
 raise AssertionError('unexpected source-contract acceptance: '+case)
if case=='literal_undeclared':
 record['rejection']=rejected(lambda:assemble(resolved,hooks=hooks,line='R1',source_guard=guard))
 assert not marker.exists()
 record['module_body_executed']=False
else:
 runtime=assemble(resolved,hooks=hooks,line='R1',source_guard=guard)
 manifest=runtime_manifest(lock,delta,resolved,runtime,{'device':'cpu','output':str(ROOT/'runs/manifest.json')})
 path=ROOT/'runs/full_bound.pt'
 if case=='computed_undeclared':
  record['rejection']=rejected(lambda:runtime.collect(2));assert not marker.exists() and name not in sys.modules
  record['module_body_executed']=False
 elif case=='late_undeclared':
  importlib.import_module(name)
  record['rejection']=rejected(lambda:runtime.update({}));assert marker.exists()
 elif case in {'legal','disk_drift','pyc_stale','symlink','spec_swap','location_swap','generated_drift','generated_extra','timing'}:
  if case=='pyc_stale':
   # Normal timestamp-based Python bytecode: compile an older same-size source,
   # then restore the declared source. The loader accepts same-second/size pyc.
   stamp=int(p.stat().st_mtime)
   p.write_text(source.replace('VALUE = 0.4','VALUE = 0.9'));os.utime(p,(stamp,stamp))
   py_compile.compile(str(p),doraise=True)
   p.write_text(source);os.utime(p,(stamp,stamp));assert file_hash(p)==guard.expected[str(p.relative_to(ROOT))]
  if case=='symlink':
   target=ROOT/'runs/alias_dep.py';target.write_text(source);p.unlink();p.symlink_to(target)
   record['rejection']=rejected(lambda:runtime.collect(1))
  elif case in {'generated_drift','generated_extra'}:
   from torch.distributed.nn.jit import instantiator
   generated=Path(instantiator.INSTANTIATED_TEMPLATE_DIR_PATH)/'_remote_module_non_scriptable.py'
   if case=='generated_drift':
    generated.write_text(generated.read_text()+'\nQA_UNPINNED=True\n')
    record['rejection']=rejected(lambda:runtime.collect(1))
   else:
    extra=generated.parent/'qa_extra.py';extra.write_text('VALUE=9\n')
    spec=importlib.util.spec_from_file_location('qa_generated_extra',extra)
    def execute():
     with guard.scope():spec.loader.exec_module(importlib.util.module_from_spec(spec))
    record['rejection']=rejected(execute)
  else:
   batch,_=runtime.collect(2)
   assert marker.exists()
   if case=='disk_drift':
    p.write_text(source.replace('VALUE = 0.4','VALUE = 0.9'))
    record['rejection']=rejected(lambda:runtime.update(batch))
   elif case in {'spec_swap','location_swap'}:
    module=sys.modules[name]
    if case=='spec_swap':module.__spec__.origin=str(ROOT/'runs/another.py')
    else:module.__file__=str(ROOT/'runs/another.py')
    record['rejection']=rejected(lambda:runtime.update(batch))
   elif case=='timing':
    samples=[]
    for i in range(7):
     start=time.perf_counter();guard.check();samples.append(time.perf_counter()-start)
    record['force_check_seconds']={'median':statistics.median(samples),'min':min(samples),'max':max(samples),'samples':samples}
    samples=[]
    for i in range(20):
     start=time.perf_counter();guard.call(hooks.reward_transform,np.zeros(3),{});samples.append(time.perf_counter()-start)
    record['cached_hook_seconds']={'median':statistics.median(samples),'max':max(samples)}
   else:
    metrics=runtime.update(batch)
    save_bound(path,runtime,manifest,2,6,int(metrics['minibatch_updates']));loaded=load_bound(path,runtime,manifest)
    record.update(rollout_decisions=2,paired_minibatches=metrics['minibatch_updates'],finite=all(np.isfinite(v) for v in metrics.values()),save_load_passed=True,loaded_steps=loaded['steps'],source_value=float(importlib.import_module(name).VALUE))
    if case=='pyc_stale':
     record['declared_source_value']=0.4
     record['actual_executed_value']=sys.modules[name].VALUE
     record['source_hash_still_matches_declared']=file_hash(p)==guard.expected[str(p.relative_to(ROOT))]
     record['stale_bytecode_accepted']=sys.modules[name].VALUE==0.9
  path.unlink(missing_ok=True);path.with_suffix('.batch01.json').unlink(missing_ok=True)
 else:raise AssertionError(case)
 if guard.failure:
  record['subsequent_collect_rejected']=rejected(lambda:runtime.collect(1))
  record['subsequent_update_rejected']=rejected(lambda:runtime.update({}))
  record['save_rejected']=rejected(lambda:save_bound(path,runtime,manifest,0,0,0))
  record['load_rejected']=rejected(lambda:load_bound(path,runtime,manifest))
  assert runtime.trainer.update_count==0 and not path.exists()
  record['optimizer_updates']=0
record['failure_latched']=guard.failure is not None
print(json.dumps(record),flush=True)
'''
if __name__=='__main__':
 cases=sys.argv[1:] or ['computed_undeclared','literal_undeclared','legal','disk_drift','late_undeclared','symlink','spec_swap','location_swap','generated_drift','generated_extra','timing','pyc_stale']
 archive=subprocess.check_output(['git','archive',CANDIDATE,'src','vendor','configs/experiments/terl_mappo_20261008','configs/experiments/terl_mappo_batch01_20261009','test','tools'])
 records=[]
 for case in cases:
  snapshot=ROOT/'runs/a3_v2_cpu'/('independent_'+case)
  if snapshot.exists():raise SystemExit('Refuse existing snapshot: '+str(snapshot))
  snapshot.mkdir(parents=True)
  with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(snapshot,filter='data')
  (snapshot/'runs/tmp').mkdir(parents=True)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(snapshot/'src')+':'+str(snapshot)+':/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/.runtime-deps',TMPDIR=str(snapshot/'runs/tmp'))
  start=time.monotonic();r=subprocess.run([sys.executable,'-c',CHILD,case],cwd=snapshot,env=env,capture_output=True,text=True,timeout=90)
  (HERE/(case+'.txt')).write_text(r.stdout+r.stderr)
  if r.returncode: print(r.stdout+r.stderr);raise SystemExit(r.returncode)
  record=json.loads(r.stdout.strip().splitlines()[-1]);record['seconds']=time.monotonic()-start
  records.append(record);print(json.dumps(record),flush=True)
 (HERE/'source_probes.json').write_text(json.dumps(records,indent=2)+'\n')
