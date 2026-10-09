#!/usr/bin/env python3
"""Validate central Batch01 metadata without importing or launching training."""

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'artifacts/2026-10-09_terl_mappo_batch01'
AXES = {'Environment', 'Network', 'Reward', 'Information', 'Task', 'Action', 'Algorithm', 'Curriculum', 'Initialization'}
ARMS = {'T1', 'N1', 'R1', 'P1', 'C0'}
SHA = re.compile(r'[0-9a-f]{40}')
HASH = re.compile(r'[0-9a-f]{64}')


def load(path):
    return json.loads(path.read_text())


def validate_v3(require_ready=False):
    state=load(DIRECTORY/'batch_state.json');lock=load(DIRECTORY/'batch_contract.lock.json');registry=load(DIRECTORY/'run_registry.json')
    central=load(ROOT/'artifacts/2026-09-21_ac_master_dag/state.json');errors=[]
    def check(condition,message):
        if not condition:errors.append(message)
    check(state['BATCH01_BASE_SHA'] is None and lock['BATCH01_BASE_SHA'] is None and registry['BATCH01_BASE_SHA'] is None and central['batch01']['BATCH01_BASE_SHA'] is None,'unreviewed V3 cannot be frozen')
    check(state['status']==lock['status']==central['batch01']['status']=='WAITING_QA_V3','V3 central status inconsistent')
    check(state['QA_SAME_SHA']==lock['QA_SAME_SHA']==registry['QA_SAME_SHA']==central['batch01']['QA_SAME_SHA']=='PENDING','independent V3 QA cannot be self-signed')
    check(state['core']['status']=='CORE_V3_SELFTEST_PASS' and state['qa']['status']=='QA_PENDING','selftest and independent QA statuses conflated')
    check(not state['qa'].get('a3_independent_signature') and state['qa']['report_head'] is None and state['qa']['tested_candidate_sha'] is None,'fabricated V3 independent signature')
    check(state['base_freeze_status']==lock['base_freeze_status']==registry['base_freeze_status']==central['batch01']['base_freeze_status']=='BASE_FREEZE_BLOCKED','freeze gate silently removed')
    check(state['historical_v2_snapshot']['received_independent_qa']['verdict']=='BASE_QA_BLOCK','V2 QA evidence lost')
    check(state['gates']==lock['gates']==central['batch01']['gates'],'launch gates inconsistent')
    receipt=load(ROOT/lock['core_verification_ref']);candidate_lock=load(ROOT/lock['candidate_lock_ref'])
    digest=hashlib.sha256(json.dumps(candidate_lock,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    check(receipt['candidate_sha']==lock['core_candidate_sha']==state['core']['candidate_sha']==registry['core_candidate_sha']==central['batch01']['core_candidate_sha'],'V3 candidate pins inconsistent')
    check(receipt['delivery_head']==lock['core_delivery_head']==state['core']['delivery_head']==registry['core_delivery_head']==central['batch01']['core_delivery_head'],'V3 delivery pins inconsistent')
    check(digest==receipt['canonical_lock_sha256']==lock['core_candidate_canonical_lock_sha256']==state['core']['canonical_lock_sha256']==registry['core_candidate_canonical_lock_sha256'],'V3 lock pin inconsistent')
    check(receipt['core_selftests']=='CORE_V3_SELFTEST_PASS' and not receipt['independent_qa'] and not receipt['base_frozen'],'selftest receipt misclassified')
    check(lock['candidate_common_source_hashes']==candidate_lock['common_sources'],'candidate source hashes inconsistent')
    for path,d in receipt['source_hash_checks'].items():check(set(d.values())=={candidate_lock['common_sources'][path]},'candidate/delivery/worktree drift: '+path)
    check(set(receipt['source_hash_checks'])==set(candidate_lock['common_sources']),'source verification incomplete')
    check(not state['training_launched'] and not state['leases'],'formal training or leases present before freeze')
    check(state['storage']['personal_quota']=='UNKNOWN' and not state['storage']['shared_space_is_exclusive_quota'] and not state['storage']['cleanup_authorized'],'storage scope drift')
    check(state['pilot_authorization']['maximum_additional_joint_decisions_per_run']==25000 and not state['pilot_authorization']['automatic_extension'] and not state['pilot_authorization']['retroactive_promotion'],'pilot authorization broadened')
    ids=[r['run_id'] for r in registry['runs']+registry.get('pilot_runs',[])]
    check(len(ids)==len(set(ids)),'duplicate run id')
    for run in registry['runs']:
        if run.get('historical_evidence_only'):check(run['status']=='COMPLETE','T0 historical status changed');continue
        check(run['status']=='WAITING_BASE' and run['base_sha'] is None and run['formal_results'] is None,'formal ARM unlocked before independent QA')
    for run in registry.get('pilot_runs',[]):
        check(run['execution_mode']=='PROVISIONAL' and run['max_additional_joint_decisions']<=25000,'unbounded or mislabeled pilot')
        check(not run.get('formal_evidence') and not run.get('retroactive_promotion'),'pilot treated as formal evidence')
        if run.get('scientific_quarantine'):
            old=state['historical_v3r1_snapshot']
            check(run['status']=='QUARANTINED_PROVISIONAL' and run['source_candidate_sha']==old['candidate_sha'] and run['source_lock_sha256']==old['canonical_lock_sha256'],'historical quarantined pilot pin/status mismatch')
        else:
            check(run['source_candidate_sha']==receipt['candidate_sha'] and run['source_lock_sha256']==digest,'pilot source pin drift')
        check(run['authorized_end_step']-run['start_step']<=25000,'pilot endpoint exceeds cap')
        if run.get('status')=='PROVISIONAL_RUNNING':
            check(run.get('pid') and run.get('gpu_uuid') and run.get('step',run['start_step'])>run['start_step'] and run.get('update',0)>run.get('start_update',0),'RUNNING without actual PPO update evidence')
    if require_ready:errors.append('QA_PENDING: independent V3 signature and final BASE freeze are required for formal readiness')
    return {'schema':'terl-mappo-batch-metadata-validation-v3','passed':not errors,'errors':errors,'batch_status':state['status'],'BATCH01_BASE_SHA':None,'core_candidate_sha':receipt['candidate_sha'],'QA_SAME_SHA':'PENDING','independent_qa':False,'execution_performed':bool(state.get('provisional_training_launched') or state.get('engineering_benchmark_executed')),'formal_execution_performed':False,'readiness_requested':require_ready}


def validate(require_ready=False):
    if load(DIRECTORY / "batch_state.json").get("status") == "WAITING_QA_V3":
        return validate_v3(require_ready)
    lock = load(DIRECTORY / 'batch_contract.lock.json')
    state = load(DIRECTORY / 'batch_state.json')
    registry = load(DIRECTORY / 'run_registry.json')
    matrix = load(DIRECTORY / 'cross_arm_delta_matrix.json')
    central = load(ROOT / 'artifacts/2026-09-21_ac_master_dag/state.json')
    snapshot = load(DIRECTORY / 'recovery_snapshot.json')
    migrations = load(DIRECTORY / 'version_migration_ledger.json')
    template = load(DIRECTORY / 'handoff_template.json')
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    base = lock['BATCH01_BASE_SHA']
    check(lock.get('QA_SAME_SHA') != 'BASE_QA_BLOCK' or base is None, 'QA_BLOCK candidate cannot be frozen')
    check(base == state['BATCH01_BASE_SHA'] == registry['BATCH01_BASE_SHA'] == central['batch01']['BATCH01_BASE_SHA'], 'BASE mismatch between lock/state/registry/central')
    check(state['status'] == central['batch01']['status'] == lock['status'], 'Batch status mismatch')
    check(lock['gates'] == state['gates'] == central['batch01']['gates'], 'Central launch gate mismatch')
    check(set(state['arm_states']) == ARMS, 'Five arm states must be present')
    check(lock['owner'] == registry['owner'] == state['owner'] == migrations['owner'] == 'MASTER', 'Central ownership changed')
    check(set(matrix['axes']) == AXES, 'Nine scientific axes incomplete')
    check(set(matrix['reference_profiles']) == {'TERL_IQN_ORIGINAL', 'T0_TERL_MAPPO', 'HISTORICAL_COCAP_MAPPO', 'NEXT_COCAP_MAPPO_TARGET'}, 'Four reference profiles incomplete')
    for arm in ARMS | {'T0'}:
        check(set(matrix['effective_profiles'][arm]) == AXES, f'{arm}: effective profile missing axis')
        check(set(matrix['comparisons'][arm]) == set(matrix['reference_profiles']), f'{arm}: comparison missing reference')
        for reference, fields in matrix['comparisons'][arm].items():
            check(set(fields) == AXES, f'{arm}/{reference}: comparison missing axis')
            for axis, values in fields.items():
                check(values['reference_contract'] == matrix['reference_profiles'][reference][axis], f'{arm}/{reference}/{axis}: reference drift')
                check(values['effective_contract'] == matrix['effective_profiles'][arm][axis], f'{arm}/{reference}/{axis}: arm drift')
    runs = registry['runs']
    ids = [r['run_id'] for r in runs]
    check(len(ids) == len(set(ids)), 'Duplicate run ID')
    check(set(ids) == {'BATCH01-T0-HISTORICAL', 'BATCH01-P1-control'} | {'BATCH01-' + a for a in ARMS}, 'Run list incomplete')
    tasks = {t['task_id']: t for t in central['tasks']}
    check(len(tasks) == len(central['tasks']), 'Duplicate central task ID')
    for r in runs:
        task_id = 'BATCH01-T0' if r.get('historical_evidence_only') else r['run_id']
        check(task_id in tasks and tasks[task_id]['status'] == r['status'], f'{r["run_id"]}: central task status mismatch')
        if r.get('historical_evidence_only'):
            check(r['status'] == 'COMPLETE', 'T0 historical anchor must remain COMPLETE')
            check(r['selected_step'] == 775000, 'Wrong historical selected step')
            check(r['selected_checkpoint_sha256'] == snapshot['t0']['checkpoint_sha256'], 'T0 anchor hash mismatch')
            continue
        check(r['base_sha'] == base, f'{r["run_id"]}: different common BASE')
        check(r['gates'] == state['gates'], f'{r["run_id"]}: launch gate drift')
        if base is None:
            check(r['status'] == 'WAITING_BASE', f'{r["run_id"]}: unfrozen BASE must block experiment')
            check(r['pid'] is None and r['lease_id'] is None and r['launch_authorization_id'] is None, f'{r["run_id"]}: launch state present before BASE freeze')
            check(r['formal_results'] is None, f'{r["run_id"]}: fabricated formal result')
        else:
            check(r['common_source_hashes'] == lock['common_source_hashes'], f'{r["run_id"]}: common code hash drift')
    if base is None:
        check(lock['status'] == state['status'] and state['status'] in {'WAITING_BASE', 'WAITING_CORE_V2', 'WAITING_QA_V2'}, 'Unfrozen batch must wait for BASE, revised Core or independent QA')
        check(all(v == 'WAITING_BASE' for v in state['arm_states'].values()), 'Arm unlocked before BASE freeze')
        check(not state['training_launched'] and not state['benchmark_executed'] and not state['leases'], 'Formal execution/lease present before BASE freeze')
    else:
        check(bool(SHA.fullmatch(base)), 'BASE must be a real full SHA')
        check(base == lock['core_candidate_sha'] == lock['qa_tested_candidate_sha'], 'QA did not test frozen Core candidate')
        check(lock.get('QA_SAME_SHA') in {'PASS', 'BASE_QA_PASS', 'BASE_QA_PASS_WITH_ARM_GATES'}, 'Frozen BASE lacks explicit independent same-SHA acceptance')
        check(bool(lock['qa_report_head'] and SHA.fullmatch(lock['qa_report_head'])), 'QA report HEAD missing')
        check(bool(lock['common_source_hashes']), 'Common source manifest missing')
        check(all(HASH.fullmatch(v) for v in lock['common_source_hashes'].values()), 'Invalid common source hash')
        canonical = {k: v for k, v in lock.items() if k != 'base_lock_sha256'}
        actual_lock_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        check(lock['base_lock_sha256'] == actual_lock_hash, 'Frozen lock digest mismatch')
    candidate = lock.get('core_candidate_sha')
    if candidate:
        receipt = load(ROOT / lock.get('core_verification_ref', 'artifacts/2026-10-09_terl_mappo_batch01/master02/core_candidate_verification.json'))
        candidate_lock = load(ROOT / lock.get('candidate_lock_ref', 'artifacts/2026-10-09_terl_mappo_batch01/master02/received_core_candidate_lock.json'))
        check(bool(SHA.fullmatch(candidate)), 'Core candidate must be a full scientific SHA')
        check(candidate != lock.get('core_delivery_head'), 'Document HEAD incorrectly used as scientific candidate')
        check(candidate == state['core']['candidate_sha'] == registry['core_candidate_sha'] == central['batch01']['core_candidate_sha'] == receipt['candidate_sha'], 'Candidate receipt mismatch')
        check(lock['core_delivery_head'] == state['core']['delivery_head'] == registry['core_delivery_head'] == central['batch01']['core_delivery_head'] == receipt['delivery_head'], 'Delivery HEAD mismatch')
        digest = hashlib.sha256(json.dumps(candidate_lock, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        check(digest == lock['core_candidate_canonical_lock_sha256'] == state['core']['canonical_lock_sha256'] == registry['core_candidate_canonical_lock_sha256'] == central['batch01']['core_candidate_canonical_lock_sha256'] == receipt['canonical_lock_sha256'], 'Candidate canonical lock mismatch')
        check(lock['candidate_common_source_hashes'] == candidate_lock['common_sources'], 'Candidate manifest drift')
        check(set(receipt['source_hash_checks']) == set(candidate_lock['common_sources']), 'Candidate hash verification incomplete')
        for path, values in receipt['source_hash_checks'].items():
            check(set(values.values()) == {candidate_lock['common_sources'][path]}, f'{path}: candidate/delivery/worktree hash mismatch')
        if base is None:
            qa_status = lock.get('QA_SAME_SHA')
            check(qa_status == state.get('QA_SAME_SHA') == registry.get('QA_SAME_SHA') == central['batch01'].get('QA_SAME_SHA') and qa_status in {'PENDING', 'BASE_QA_BLOCK'}, 'Unfrozen candidate QA state inconsistent')
            if qa_status == 'PENDING':
                check(lock['qa_report_head'] is None and lock['qa_tested_candidate_sha'] is None and state['qa']['report_head'] is None and state['qa']['tested_candidate_sha'] is None, 'Core self-tests or interim activity mistaken for independent acceptance')
            elif qa_status == 'BASE_QA_BLOCK':
                qa_receipt = load(DIRECTORY / 'master03/qa_block_receipt.json')
                check(qa_receipt['verdict'] == state['qa']['verdict'] == 'BASE_QA_BLOCK', 'Blocking QA verdict missing')
                check(candidate == lock['qa_tested_candidate_sha'] == state['qa']['tested_candidate_sha'] == qa_receipt['tested_candidate_sha'], 'Blocking QA tested another candidate')
                check(lock['qa_report_head'] == state['qa']['report_head'] == registry['qa_report_head'] == central['batch01']['qa_report_head'] == qa_receipt['qa_report_head'], 'Blocking QA report HEAD mismatch')
                check(bool(SHA.fullmatch(qa_receipt['qa_report_head'])) and qa_receipt['qa_same_sha_identity_verified'], 'Independent blocking report pin missing')
                check(lock['qa_report_sha256'] == state['qa']['report_sha256'] == qa_receipt['report_sha256'], 'Blocking report hash mismatch')
                check(digest == qa_receipt['canonical_lock_sha256'], 'Blocking QA lock mismatch')
                check(state['status'] == state['core']['status'] == 'WAITING_CORE_V2', 'QA block must wait for revised Core candidate')
                check(lock['core_v1_status'] == state['core_v1_status'] == state['core']['v1_status'] == registry['core_v1_status'] == central['batch01']['core_v1_status'] == 'CORE_V1_QA_BLOCKED', 'Rejected V1 candidate status mismatch')
                check(state['gates']['BASE_FREEZE'] == lock['base_freeze_status'] == state['base_freeze_status'] == registry['base_freeze_status'] == central['batch01']['base_freeze_status'] == 'BASE_FREEZE_BLOCKED', 'Rejected candidate freeze gate open')
                check(state['gates']['CORE_CANDIDATE'] == 'WAITING_CORE_V2' and state['gates']['INDEPENDENT_QA_SAME_SHA'] == 'BASE_QA_BLOCK', 'QA block launch gates incorrect')
                check(tasks['BATCH01-CORE-A2']['status'] == 'CORE_V1_QA_BLOCKED' and tasks['BATCH01-CORE-V2']['status'] == 'WAITING_CORE_V2' and tasks['BATCH01-QA-BASE']['status'] == 'BASE_QA_BLOCK' and tasks['BATCH01-BASE-FREEZE']['status'] == 'BASE_FREEZE_BLOCKED', 'Central QA/Core/freeze tasks mismatch')
                check(state['core']['v2_candidate_sha'] is None and state['core']['v2_lock_sha256'] is None and lock['next_core_candidate_sha'] is None, 'V2 candidate fabricated before delivery')
                check({f['id'] for f in state['qa_fix_requirements']} == {'B1', 'B2', 'B3', 'B4', 'Q1'}, 'Core V2 fix requirements incomplete')
                check(not state.get('qa_smoke_leases') and not central['batch01'].get('qa_smoke_leases'), 'Rejected candidate retains active QA lease')
                closeout = load(DIRECTORY / 'master03/qa_lease_closeout.json')
                check(closeout['status'] == 'CLOSED_UNUSED_CORE_V1_QA_BLOCKED' and not closeout['cuda_executed_by_qa'] and not closeout['new_lease_issued'], 'Rejected candidate lease not closed unused')
                check(closeout['candidate_sha'] == candidate and closeout['qa_report_head'] == qa_receipt['qa_report_head'], 'Lease closeout pin mismatch')
                check(not migrations['versions'] and not migrations['migrations'], 'Rejected unfrozen candidate fabricated a BASE release or migration')
            check(lock['base_lock_sha256'] is None and lock['base_version'] is None and not lock['common_source_hashes'], 'Candidate snapshot mistaken for frozen BASE')
        storage = load(DIRECTORY / 'master02/storage_a0_receipt.json')
        check(state['storage']['audit_id'] == storage['audit_id'], 'Storage receipt audit ID mismatch')
        check(state['storage']['personal_quota'] == 'UNKNOWN' and not state['storage']['shared_space_is_exclusive_quota'], 'Shared available disk incorrectly treated as private quota')
        check(not storage['deletion_authorized'] and not storage['archive_authorized'] and storage['actual_reclaimed_bytes_this_task'] == 0, 'Unauthorized Storage cleanup registered')
        check(not state['storage']['cleanup_authorized'] and not state['storage']['archive_authorized'], 'Storage cleanup scope changed')
        if state['status'] == 'WAITING_QA_V2':
            old = load(DIRECTORY / 'master03/qa_block_receipt.json')
            history = state['qa']['historical_v1_review']
            check(history['verdict'] == 'BASE_QA_BLOCK' and history['tested_candidate_sha'] == old['tested_candidate_sha'] and history['report_head'] == old['qa_report_head'], 'V1 independent QA block history lost')
            check(candidate != old['tested_candidate_sha'] and digest != old['canonical_lock_sha256'], 'V2 reuses rejected V1 pins')
            check(lock['core_v1_status'] == state['core_v1_status'] == registry['core_v1_status'] == central['batch01']['core_v1_status'] == 'CORE_V1_QA_BLOCKED', 'V1 status changed after V2 delivery')
            check(state['core']['status'] == lock['core_candidate_status'] == tasks['BATCH01-CORE-V2']['status'] == 'CORE_V2_DELIVERED', 'V2 delivery status mismatch')
            check(state['qa']['status'] == tasks['BATCH01-QA-V2']['status'] == state['gates']['INDEPENDENT_QA_SAME_SHA'] == 'WAITING_QA_V2', 'V2 QA wait status mismatch')
            check(state['qa']['requested_candidate_sha'] == tasks['BATCH01-QA-V2']['candidate_sha'] == candidate and state['qa']['requested_canonical_lock_sha256'] == tasks['BATCH01-QA-V2']['canonical_lock_sha256'] == digest, 'V2 QA request not pinned to current candidate')
            check(state['gates']['BASE_FREEZE'] == tasks['BATCH01-BASE-FREEZE']['status'] == 'BASE_FREEZE_BLOCKED', 'V2 delivery automatically unlocked freeze')
            check(tasks['BATCH01-CORE-A2']['status'] == 'CORE_V1_QA_BLOCKED' and tasks['BATCH01-QA-BASE']['status'] == 'BASE_QA_BLOCK', 'Historical V1 tasks changed')
            check(not migrations['versions'] and not migrations['migrations'], 'Unfrozen V2 fabricated a BASE release/migration')
            check({f['id'] for f in state['qa_fix_requirements']} == {'B1', 'B2', 'B3', 'B4', 'Q1'} and all(f['status'] == 'CORE_V2_REPORTED_FIXED_PENDING_A3' for f in state['qa_fix_requirements']), 'Core repair claims mistaken for independent closure')
            recheck = load(ROOT / state['storage']['latest_recheck_ref'])
            check(recheck['A0_audit_still_valid'] and all(x['matches_received_a0'] for x in recheck['original_file_hash_checks'].values()) and not recheck['cleanup_authorized'], 'A0 integrity or cleanup scope changed')
            proposal = lock['evaluation_proposal']
            domains = state['evaluation_protocol_open_issue']
            check(proposal['status'] == 'PROPOSED_NOT_FROZEN', 'Future ARM seed proposal silently frozen')
            check(domains['core_candidate_domains'] != domains['central_batch01_proposed_domains'] and domains['no_observed_t0_final_reuse'], 'T0 historical and future ARM seed domains conflated')
    qa_lease_ids = state.get('qa_smoke_leases', [])
    if qa_lease_ids:
        lease = load(ROOT / lock['qa_smoke_lease_ref'])
        check(qa_lease_ids == central['batch01']['qa_smoke_leases'] == registry['active_qa_smoke_leases'] == lock['active_qa_smoke_leases'] == [lease['lease_id']], 'QA lease registry mismatch')
        check(lease['purpose'] == 'INDEPENDENT_QA_CUDA_CORRECTNESS_ONLY' and not lease['formal_training_authorized'] and not lease['benchmark_authorized'] and not lease['arms_authorized'], 'QA lease authorizes formal work')
        check(lease['candidate_sha'] == candidate and lease['canonical_lock_sha256'] == lock['core_candidate_canonical_lock_sha256'], 'QA lease tests a different candidate/lock')
        check(lease['physical_gpu'] == 0 and lease['CUDA_VISIBLE_DEVICES'] == '0' and lease['process_local_device'] == 'cuda:0', 'QA GPU mapping mismatch')
        check(lease['max_compute_processes'] == lease['max_attempts'] == 1 and lease['parallel_gpu_jobs'] == 0, 'QA lease concurrency/retry broadened')
        caps = lease['execution_caps']
        v2 = state['status'] == 'WAITING_QA_V2'
        for key, maximum in {'cuda_test_total_wall_seconds': 300, 'lease_active_wall_seconds': 600, 'total_joint_environment_decisions': 1024 if v2 else 128, 'total_rollout_ppo_calls': 8, 'torch_peak_allocated_mib': 2048, 'own_driver_vram_mib': 4096, 'cpu_threads': 1, 'own_ram_rss_gib': 8, 'new_output_growth_gib': 2 if v2 else 1}.items():
            check(0 < caps[key] <= maximum, f'QA lease exceeds {key}')
        if v2:
            closeout = load(DIRECTORY / 'master03/qa_lease_closeout.json')
            check(lease['lease_id'] != closeout['lease_id'] and closeout['status'] == 'CLOSED_UNUSED_CORE_V1_QA_BLOCKED', 'V2 reused V1 lease')
            check(lease['expected_test_work']['full_width_network'] == [256, 8, 4] and lease.get('real_candidate_checkpoint_guard_supplement', {}).get('required'), 'V2 smoke lacks formal network or real-pin resume/source guard')
            check(set(lease['allowed_pytest_nodes']) == {
                'test/test_terl_mappo_batch01_base_20261009.py::test_formal_entry_t0_full_update_and_exact_resume[cuda:0]',
                'test/test_terl_mappo_batch01_qa_repairs_20261009.py::test_b3_density_guard_preserves_all_rng_and_replays_exact_latent[cuda:0]',
            }, 'V2 CUDA lease includes a non-formal network or unreviewed node')
        check(caps['evaluator_workers'] == 0, 'QA lease creates evaluator backlog')
        window = (dt.datetime.fromisoformat(lease['expires_at']) - dt.datetime.fromisoformat(lease['issued_at'])).total_seconds()
        check(0 < window <= 7200, 'QA smoke grant window exceeds two hours')
        if base is None:
            check(not lock['gpu_leases_granted'] and not central['batch01']['new_gpu_leases'], 'QA lease leaked into formal lease namespace')
            check(state['gates']['GPU_LEASE'] == 'NOT_REQUESTED' and state['gates']['RUN_AUTHORIZATION'] == 'NOT_AUTHORIZED_THIS_TURN', 'QA lease unlocked formal gates')
    p1 = next(r for r in runs if r['run_id'] == 'BATCH01-P1')
    control = next(r for r in runs if r['run_id'] == 'BATCH01-P1-control')
    check(p1['parent_checkpoint_sha256'] == control['parent_checkpoint_sha256'] == snapshot['t0']['checkpoint_sha256'], 'P1 paired parent mismatch')
    check(p1['proposed_ppo_target_kl'] == .01 and control['proposed_ppo_target_kl'] == .02, 'P1 single-variable proposal drift')
    check(control['primary_delta'] == {}, 'P1 control has an experimental delta')
    check(set(template['exact_changes']) == set(template['inherited_contracts']) == AXES, 'Handoff lacks nine axes')
    check({x['task_id'] for x in state['batch02_preregistration']} == {'BATCH02-' + a for a in ['E1', 'R2', 'N2', 'V1', 'M1', 'M2']}, 'Batch02 preregistration incomplete')
    for x in state['batch02_preregistration']:
        check(x['status'] == 'PREREGISTERED_LOCKED' and not x['training_authorized'] and not x['implementation_authorized'], f'{x["task_id"]}: second batch unlocked')
        check(tasks[x['task_id']]['status'] == x['status'], f'{x["task_id"]}: central mismatch')
    for mode in ['argmax', 'sample']:
        counts = snapshot['t0']['final_raw_recount'][mode]
        check(counts['episodes'] == 50 and counts['normal_capture'] == 45 and counts['collision'] == 5, 'Historical final evidence mismatch')
    if require_ready:
        check(base is not None, 'WAITING_BASE: no Core + independent QA frozen SHA')
        for name, status in state['gates'].items():
            check(status == 'PASS', f'{name}: {status}')
    return {'schema': 'terl-mappo-batch-metadata-validation-v1', 'passed': not errors, 'readiness_requested': require_ready, 'batch_status': state['status'], 'BATCH01_BASE_SHA': base, 'core_candidate_sha': candidate, 'QA_SAME_SHA': state.get('QA_SAME_SHA'), 'qa_only_lease_ids': qa_lease_ids, 'storage_status': state['storage']['status'], 'errors': errors, 'execution_performed': False, 'scope': 'Metadata consistency only; not independent scientific QA or launch authorization'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-ready', action='store_true', help='Fail closed until BASE and all execution gates are PASS')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate(args.require_ready)
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
