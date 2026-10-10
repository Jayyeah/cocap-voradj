"""Explicit user budget binding and strict full-state operational continuation.
Scientific sources and resolved config are immutable across this continuation.
Old manifests, sidecars, source commits and quarantines are checked before load.
"""
from pathlib import Path
import hashlib,json,subprocess
from terl_mappo.batch01.contracts import ROOT,fingerprint
from terl_mappo.run import file_hash,load_checkpoint
AUTH_PATH=Path('/home/yjq/rl/CoCap1/ac-master-dag-20260921/artifacts/2026-10-09_terl_mappo_batch01/commander_v3r2/provisional_long_authorization_20261010.json')
AUTH_SHA='d7f85e98a20adab536fbac5ee66251af07f945686ad9712118e1f3d3a4d36e66'
LABEL='c0'
OPS={'src/terl_mappo/batch01/arms/runner.py','src/terl_mappo/batch01/arms/evaluator.py','src/terl_mappo/batch01/arms/long_budget.py','src/terl_mappo/batch01/arms/long_preflight.py'}
def read(p):return json.loads(Path(p).read_text())
def authorization(delta):
    if file_hash(AUTH_PATH)!=AUTH_SHA:raise ValueError('explicit long authorization hash changed')
    a=read(AUTH_PATH);r=a['runs'][LABEL]
    if a['candidate_sha']!=delta['base']['candidate_sha'] or a['canonical_lock_sha256']!=delta['base']['lock_sha256']:raise ValueError('authorization BASE pin mismatch')
    if a['execution_mode']!='PROVISIONAL_LONG' or a['formal_evidence'] or a['retroactive_promotion'] or a['automatic_budget_extension'] or a['final_domain_allowed'] or a['stage3_training_authorized']:raise ValueError('authorization scientific gate drift')
    if r['authorized_end_step']!=(100000 if LABEL=='t1' else 1000000):raise ValueError('unregistered long endpoint')
    if LABEL=='t1' and delta['extensions']['parameters']['stage']!=2:raise ValueError('Stage3 is not authorized')
    return r

def validate_launch(delta,run_id,decisions,stored=None):
    r=authorization(delta)
    if run_id!=r['run_id'] or decisions!=r['authorized_end_step']-r['start_step']:raise ValueError('long run identity / budget differs from explicit user grant')
    reg=read(AUTH_PATH.parents[1]/'run_registry.json')
    rows=[x for x in reg['pilot_runs'] if x['run_id']==run_id]
    if len(rows)!=1:raise ValueError('long run must be uniquely registered before execution')
    row=rows[0]
    if row['execution_mode']!='PROVISIONAL_LONG' or row['delta_hash']!=fingerprint(delta) or row['authorization_sha256']!=AUTH_SHA or row['authorized_end_step']!=r['authorized_end_step']:raise ValueError('registered long source/budget pin mismatch')
    if row.get('scientific_quarantine') or row.get('formal_evidence'):raise ValueError('quarantined/formal long run prohibited')
    if stored is not None and (stored['budget_authorization_sha256']!=AUTH_SHA or stored['authorized_end_step']!=r['authorized_end_step']):raise ValueError('resume cannot change long grant')
    return r

def migrate(runtime,delta):
    r=authorization(delta)
    if r['parent_checkpoint'] is None:return None,None
    manifest_path=Path(r['parent_manifest']);path=Path(r['parent_checkpoint'])
    if file_hash(manifest_path)!=r['parent_manifest_sha256']:raise ValueError('parent manifest hash changed')
    old=read(manifest_path);side=read(path.with_suffix('.batch01.json'))
    if old['run_id']!=r['previous_run_id'] or old['execution_mode']!='PROVISIONAL' or old['base']!=delta['base'] or old['source_git_head']!=r['previous_head']:raise ValueError('wrong parent identity/source BASE')
    if old['resolved_config']!=runtime.config:raise ValueError('operational continuation may not change resolved scientific config')
    if old['authorized_end_step']!=r['start_step'] or side['steps']!=r['start_step']:raise ValueError('parent is not the registered completed pilot endpoint')
    if side['manifest_fingerprint']!=fingerprint(old) or side['base']!=old['base'] or side['checkpoint_sha256']!=r['parent_checkpoint_sha256'] or file_hash(path)!=r['parent_checkpoint_sha256']:raise ValueError('parent checkpoint/sidecar/manifest binding mismatch')
    previous_delta=read(manifest_path.parent/'delta.json')
    if previous_delta!=old['delta_manifest']:raise ValueError('parent delta file differs from manifest')
    reg=read(AUTH_PATH.parents[1]/'run_registry.json')
    prior=[x for x in reg['pilot_runs'] if x['run_id']==r['previous_run_id']]
    if len(prior)!=1 or prior[0].get('scientific_quarantine') or prior[0]['status']!='PROVISIONAL_COMPLETE' or prior[0]['delta_hash']!=fingerprint(previous_delta):raise ValueError('parent is quarantined, incomplete or unregistered')
    expected=runtime.source_guard.expected;previous=old['common_source_hashes']
    differences={p for p in set(expected)|set(previous) if expected.get(p)!=previous.get(p)}
    if differences-OPS:raise ValueError('scientific source change in continuation: '+str(sorted(differences-OPS)))
    if old['generated_runtime_source_hashes']!=runtime.source_guard.generated_sources:raise ValueError('generated source mismatch')
    # Every old source is verified against its original committed object, rather
    # than adopting the current disk or trusting a filename/mtime.
    for p,digest in previous.items():
        source=subprocess.check_output(['git','-C',str(ROOT),'show',r['previous_head']+':'+p])
        if hashlib.sha256(source).hexdigest()!=digest:raise ValueError('parent source receipt mismatch: '+p)
    with runtime.source_guard.scope():
        runtime.source_guard.check()
        loaded=load_checkpoint(path,runtime.trainer,runtime.adapter,runtime.config,runtime.trainer.device,strict=True)
        runtime.source_guard.check()
    if loaded['steps']!=r['start_step']:raise ValueError('checkpoint counter mismatch')
    receipt={'type':'STRICT_FULL_STATE_OPERATIONAL_CONTINUATION','parent_run_id':r['previous_run_id'],'parent_head':r['previous_head'],'checkpoint':str(path),'checkpoint_sha256':r['parent_checkpoint_sha256'],'parent_manifest_sha256':r['parent_manifest_sha256'],'steps':loaded['steps'],'strict_load':True,'config_unchanged':True,'science_sources_unchanged':True,'declared_operational_source_changes':sorted(differences),'authorization_sha256':AUTH_SHA,'formal_evidence':False,'retroactive_promotion':False}
    return loaded,receipt
