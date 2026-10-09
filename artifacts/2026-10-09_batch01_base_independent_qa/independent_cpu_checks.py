"""Independent full-rollout T0 regression and committed provenance checks."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from terl_mappo.native import ROOT, NativeStage1
from terl_mappo.run import build, collect, rng_state, file_hash, source_hash
from terl_mappo.batch01.contracts import (source_inventory, fingerprint, verify_base,
    verify_committed_pin, validate_resources, LOCK_PATH)
from terl_mappo.batch01.interfaces import assemble
from terl_mappo.batch01.provenance import runtime_manifest, assert_loaded_sources_covered, active_values
from terl_mappo.batch01.checkpoints import save_bound, load_bound, load_anchor

CANDIDATE = '40bd91b56cd199e527f7a2bb2317b2c1c5b61e4a'
DELIVERY = 'bdbd502ff2f071b0cac07a0d0d956a2f35015ea1'
PIN = '91dc6ea76c0f7b76ffa43441286db40b9e4ea1e9e19885a328df16ab6d024bf3'
HERE = Path(__file__).resolve().parent

def same(a, b, path='root', seen=None):
    seen = set() if seen is None else seen
    pair = (id(a), id(b))
    if pair in seen: return
    seen.add(pair)
    if isinstance(a, torch.Tensor): assert torch.equal(a.cpu(), b.cpu()), path
    elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b, err_msg=path)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a: same(a[k], b[k], path + '.' + str(k), seen)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): same(x, y, path + '.' + str(i), seen)
    elif isinstance(a, np.random.RandomState): same(a.get_state(), b.get_state(), path, seen)
    elif hasattr(a, 'query') and hasattr(a, 'data'):
        same(a.data, b.data, path, seen); same(a.leafsize, b.leafsize, path, seen)
        same(a.boxsize, b.boxsize, path, seen)
        same(a.query(a.data, k=len(a.data)), b.query(b.data, k=len(b.data)), path, seen)
    elif hasattr(a, '__dict__'):
        assert type(a) is type(b), path
        same(a.__dict__, b.__dict__, path, seen)
    else: assert a == b, path

torch.set_num_threads(1)
started = time.monotonic()
lock = json.loads((ROOT / LOCK_PATH).read_text())
assert fingerprint(lock) == PIN and len(lock['common_sources']) == 78
assert source_inventory() == lock['common_sources']
for p, digest in lock['common_sources'].items():
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'show', CANDIDATE + ':' + p])
    assert hashlib.sha256(raw).hexdigest() == digest, p
for p, digest in source_hash().items():
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'show', lock['parent_sha'] + ':' + p])
    assert hashlib.sha256(raw).hexdigest() == digest, p
for p, digest in lock['common_sources'].items():
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'show', DELIVERY + ':' + p])
    assert hashlib.sha256(raw).hexdigest() == digest, p
old = json.loads(subprocess.check_output(['git', '-C', str(ROOT), 'show', '9a97d392:' + LOCK_PATH]))

delta = {'schema': 'terl.batch01.delta.v1', 'line': 'T0',
    'base': {'candidate_sha': CANDIDATE, 'parent_sha': lock['parent_sha'], 'lock_sha256': PIN},
    'config_changes': [], 'source_changes': [], 'extensions': {}}
config = verify_base(lock, delta, PIN); verify_committed_pin(lock, delta)
runtime = assemble(config, 'cpu')
manifest = runtime_manifest(lock, delta, config, runtime, validate_resources(HERE / 't0_runtime_manifest.json'))
(HERE / 't0_delta.json').write_text(json.dumps(delta, indent=2) + '\n')
(HERE / 't0_runtime_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
loaded_start = sorted({Path(m.__file__).resolve().relative_to(ROOT).as_posix()
    for m in list(sys.modules.values()) if getattr(m, '__file__', None)
    and any(Path(m.__file__).resolve().is_relative_to(ROOT / d) for d in ('src', 'vendor/terl'))})
assert not set(loaded_start) - set(lock['common_sources'])

initial = copy.deepcopy(runtime.trainer.state_dict())
batch, episodes = runtime.collect(config['rollout_length'])
after_collection_rng = rng_state()
metrics = runtime.update(batch)
updated = copy.deepcopy(runtime.trainer.state_dict()); updated_rng = rng_state()
reference = build(config, 'cpu'); adapter = NativeStage1(config['seed'])
same(initial, reference.state_dict())
expected, expected_episodes = collect(reference, adapter, config['rollout_length'])
same(batch, expected); same(episodes, expected_episodes); same(after_collection_rng, rng_state())
same(metrics, reference.update(expected, True)); same(updated, reference.state_dict()); same(updated_rng, rng_state())
assert metrics['minibatch_updates'] > 0 and all(np.isfinite(v) for v in metrics.values())
assert_loaded_sources_covered(source_inventory())

checkpoint = ROOT / 'runs/a3_qa02/full_rollout_cpu.pt'
try:
    save_bound(checkpoint, runtime, manifest, 256, int(batch['active_mask'].sum()), int(metrics['minibatch_updates']))
    following, following_episodes = runtime.collect(16); following_metrics = runtime.update(following)
    following_state = copy.deepcopy(runtime.trainer.state_dict())
    following_env = runtime.adapter.state_dict(); following_rng = rng_state()
    restored = assemble(config, 'cpu'); loaded = load_bound(checkpoint, restored, manifest)
    actual, actual_episodes = restored.collect(16); actual_metrics = restored.update(actual)
    same(following, actual); same(following_episodes, actual_episodes); same(following_metrics, actual_metrics)
    same(following_state, restored.trainer.state_dict()); same(following_env, restored.adapter.state_dict())
    same(following_rng, rng_state())
    assert loaded['steps'] == 256
finally:
    checkpoint.unlink(missing_ok=True); checkpoint.with_suffix('.batch01.json').unlink(missing_ok=True)

anchor_path = Path('/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/runs/terl_mappo_stage1_seed9_1m_continuation/checkpoints/step_000775000.pt')
anchor_hash = '590d486876d4fca1cd75311c2dcc1255f847a1f756d91eed202e737193b58db4'
anchor_runtime = assemble(config); anchor = load_anchor(anchor_path, anchor_hash, anchor_runtime)
anchor_active = active_values(anchor_runtime)
anchor_batch, _ = anchor_runtime.collect(8)
assert anchor_runtime.trainer.assert_behavior_log_probs(anchor_batch) < 1e-5
assert file_hash(anchor_path) == anchor_hash

ppo_delta = copy.deepcopy(delta); ppo_delta['line'] = 'P1'
ppo_delta['config_changes'] = [{'path': 'ppo.actor_lr', 'before': 3e-5, 'after': 1e-5, 'reason': 'QA controlled fork'}]
ppo_config = verify_base(lock, ppo_delta, PIN); ppo_runtime = assemble(ppo_config, line='P1')
p1_error = None
try: load_anchor(anchor_path, anchor_hash, ppo_runtime)
except ValueError as exc: p1_error = str(exc)
assert p1_error is not None

n1_delta = copy.deepcopy(delta); n1_delta['line'] = 'N1'
n1_delta['config_changes'] = [{'path': 'hidden_dim', 'before': 256, 'after': 128, 'reason': 'QA capacity scope check'}]
n1_config = verify_base(lock, n1_delta, PIN); n1_runtime = assemble(n1_config, line='N1')
n1_arch = active_values(n1_runtime)['architecture']

result = {'candidate_sha': CANDIDATE, 'canonical_lock_sha256': PIN,
    'actual_git_head_during_checks': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
    'common_files': 78, 'all_78_hashes_match_candidate_and_delivery': True,
    'historical_t0_scientific_hashes_match_parent': True,
    'first_lock_files': len(old['common_sources']),
    'first_lock_missing_current_loaded_dependencies': sorted(set(loaded_start) - set(old['common_sources'])),
    'current_loaded_sources': loaded_start, 'baseline_loaded_source_coverage': True,
    'full_size_cpu': {'hidden_dim': 256, 'heads': 8, 'layers': 4, 'decisions': 256,
        'active_transitions': int(batch['active_mask'].sum()), 'paired_minibatches': metrics['minibatch_updates'],
        'exact_t0_initializer_rollout_update_adam_valuenorm_rng': True,
        'checkpoint_resume_decisions': 16, 'exact_resume_all_fields': True},
    'anchor': {'sha256': anchor_hash, 'steps': anchor['steps'], 'episode_steps': anchor['runtime']['env'].episode_time_steps,
        'trainer_keys': list(anchor['trainer']), 'runtime_keys': list(anchor['runtime']), 'rng_keys': list(anchor['rng']),
        'strict_load_passed': True, 'readonly_hash_preserved': True, 'active_optimizer': anchor_active['actor_optimizer']},
    'p1': {'fresh_live_lr': ppo_runtime.trainer.actor_optimizer.param_groups[0]['lr'],
        'changed_config_full_anchor_load_rejected': p1_error, 'controlled_fork_remains_arm_gate': True},
    'n1_capacity_scope': {'declared_hidden_dim': 128, 'actual_architecture': n1_arch,
        'shared_critic_also_changed': n1_arch['critic']['hidden_dim'] == 128},
    'metric_definition_issue': {'manifest_critic_ev': manifest['metric_definitions']['critic_ev'],
        'actual_learner': 'post-update denormalized V versus pre-update GAE return targets'},
    'cuda_executed': False, 'seconds': time.monotonic() - started, 'formal_training': False}
(HERE / 'independent_cpu_checks.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
