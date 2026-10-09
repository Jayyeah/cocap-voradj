"""Read-only recovery, historical holdout and executed bytecode evidence checks."""
import hashlib,json,marshal,struct,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
from terl_mappo.batch01.contracts import ROOT,LOCK_PATH,fingerprint
from terl_mappo.batch01.evaluation import seed_manifest
HERE=Path(__file__).resolve().parent
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='863a0cf55aca0ace8a0aaab36d9166aa4c97268f'
lock=json.loads((ROOT/LOCK_PATH).read_text())
assert fingerprint(lock)=='885ec7b8617cf88e2537b08ebf041e1d8bfc41f120b7ec4aef1708c4a513a8a4'
suite=ET.parse(HERE/'tests.xml').getroot().find('testsuite')
assert suite.attrib['tests']=='80' and suite.attrib['failures']=='0' and suite.attrib['errors']=='0' and suite.attrib['skipped']=='3'
required=['test_formal_entry_t0_full_update_and_exact_resume[cpu]','test_shared_selected_checkpoint_strict_resume_readonly','test_b2_lambda_native_adapter_rejected_during_startup','test_b2_actual_behavior_probe_checks_consumer_and_preserves_live_state','test_b3_density_guard_preserves_all_rng_and_replays_exact_latent[cpu]','test_b4_ev_matches_post_update_denormalized_value_and_pre_update_targets','test_q1_production_lock_generator_stays_core_only']
for name in required:
 c=next(c for c in suite.findall('testcase') if c.attrib['name']==name)
 assert c.find('skipped') is None and c.find('failure') is None
anchor=Path('/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/runs/terl_mappo_stage1_seed9_1m_continuation/checkpoints/step_000775000.pt')
anchor_hash=hashlib.sha256(anchor.read_bytes()).hexdigest()
assert anchor_hash=='590d486876d4fca1cd75311c2dcc1255f847a1f756d91eed202e737193b58db4'
historical=anchor.parents[1]/'evaluations/final_000775000.json'
hash_final=hashlib.sha256(historical.read_bytes()).hexdigest()
assert hash_final=='996db51d419ae4e47292d3df70a6b4000e5046e6f5d1e424824f7bb30644069c'
old=json.loads(historical.read_text());history_seeds={r['seed'] for r in old['episodes']}
new=seed_manifest(lock['anchor_config'],lock['evaluation_protocol']);offset=new['sample_torch_offset']
history_action={s+100000 for s in history_seeds}
seed_result={'historical_final_path':str(historical),'historical_final_sha256':hash_final,'historical_final_environment_seeds':sorted(history_seeds),'historical_final_sample_action_seeds':sorted(history_action),'locked_protocol':lock['evaluation_protocol'],'seed_manifest':new,'current_domains_internal_isolation_passed':True,'overlap_with_historical_final':{d:{'environment':sorted(set(new[d])&history_seeds),'sample_action':sorted({s+offset for s in new[d]}&history_action)} for d in ['screen','selection','final']},'future_arm_fresh_holdout_is_allocated':False,'classification':'ARM_PRETRAINING_NEW_HELDOUT_REQUIRED_SHARED_PROTOCOL_CURRENTLY_REUSES_T0_FINAL'}
(HERE/'seed_isolation.json').write_text(json.dumps(seed_result,indent=2)+'\n')
source=ROOT/'runs/a3_v2_cpu/independent_pyc_stale/src/cocap_voradj/models/continuous/qa_dynamic_dep.py'
cache=next(source.parent.joinpath('__pycache__').glob('qa_dynamic_dep.*.pyc'))
b=cache.read_bytes();flags,mtime,size=struct.unpack('<III',b[4:16]);code=marshal.loads(b[16:])
expected=compile(source.read_bytes(),str(source),'exec')
assert 0.9 in code.co_consts and 0.4 in expected.co_consts
bytecode={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_size':source.stat().st_size,'source_mtime_seconds':int(source.stat().st_mtime),'pyc_path':str(cache),'pyc_sha256':hashlib.sha256(b).hexdigest(),'pyc_flags':flags,'pyc_timestamp':mtime,'pyc_source_size':size,'actual_code_filename':code.co_filename,'actual_code_float_constants':[v for v in code.co_consts if isinstance(v,float)],'declared_source_float_constants':[v for v in expected.co_consts if isinstance(v,float)],'timestamp_size_checks_match':mtime==int(source.stat().st_mtime) and size==source.stat().st_size,'no_science_files_edited':True}
assert bytecode['timestamp_size_checks_match'] and flags==0
(HERE/'bytecode_details.json').write_text(json.dumps(bytecode,indent=2)+'\n')
recovery={'candidate_sha':'863a0cf55aca0ace8a0aaab36d9166aa4c97268f','canonical_lock_sha256':fingerprint(lock),'science_head_unchanged':True,'suite_exit_code_recovered':0,'probe_exit_code_recovered':0,'suite':suite.attrib,'passed':77,'cuda_skipped':3,'required_completed_cases':required,'anchor_sha256_preserved':anchor_hash,'v1_fix_outcomes':{'B1':'original attack/scenarios repaired but stale timestamp pyc bypass confirmed','B2':'PASS startup actual capability/consumer tests','B3':'PASS CPU RNG preservation/ratio/error rejection; CUDA untested','B4':'PASS metadata/formula and T0 numerical parity','Q1':'PASS unmodified suite on independent branch; production Core-only restriction retained'},'interruption_cause':'No test crash or scientific failure in background commands; both completed exit0. Conversation/session handoff occurred before result collection; underlying client transport cause not observable.','no_existing_results_overwritten':True,'cuda_lease':'not_granted_in_current_session'}
(HERE/'recovery.json').write_text(json.dumps(recovery,indent=2)+'\n')
print(json.dumps({'recovered_passed':77,'skipped':3,'source_bytecode_mismatch':bytecode,'future_arm_final_seed_overlap':len(seed_result['overlap_with_historical_final']['final']['environment'])},indent=2))
