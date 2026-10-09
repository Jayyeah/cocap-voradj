"""Minimal source-lock escape in an allowed R1 callback; isolated Git snapshot.

No candidate/Core/T0 files are edited. The disposable snapshot and extension
are under the QA worktree's ignored runs directory. All execution is CPU.
"""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

CANDIDATE = '40bd91b56cd199e527f7a2bb2317b2c1c5b61e4a'
PIN = '91dc6ea76c0f7b76ffa43441286db40b9e4ea1e9e19885a328df16ab6d024bf3'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

if '--child' not in sys.argv:
    snapshot = ROOT / 'runs/a3_qa02/lazy_dependency_full_checkpoint_snapshot'
    if snapshot.exists():
        raise SystemExit('Refuse to reuse an existing reproduction snapshot')
    snapshot.mkdir(parents=True)
    archive = subprocess.check_output(['git', 'archive', CANDIDATE, 'src', 'vendor',
        'configs/experiments/terl_mappo_20261008',
        'configs/experiments/terl_mappo_batch01_20261009', 'tools/batch01_base_20261009.py'])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(snapshot, filter='data')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(snapshot / 'src') + ':/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/.runtime-deps'
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', str(snapshot)],
        cwd=snapshot, env=env, capture_output=True, text=True)
    (HERE / 'lazy_dependency_repro.txt').write_text(result.stdout + result.stderr)
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    raise SystemExit(result.returncode)

snapshot = Path(sys.argv[-1])
sys.path.insert(0, str(snapshot / 'src'))
import numpy as np
import torch
from terl_mappo.batch01.contracts import source_inventory, verify_base, verify_committed_pin, validate_resources
from terl_mappo.batch01.interfaces import Hooks, assemble
from terl_mappo.batch01.provenance import runtime_manifest, assert_loaded_sources_covered
from terl_mappo.batch01.checkpoints import assert_manifest_sources, save_bound, load_bound
from terl_mappo.run import file_hash
import importlib

torch.set_num_threads(1)
lock = json.loads((snapshot / 'configs/experiments/terl_mappo_batch01_20261009/base_lock.json').read_text())
assert source_inventory() == lock['common_sources']
module_name = 'cocap_voradj.models.continuous.box_actor'
source_path = 'src/cocap_voradj/models/continuous/box_actor.py'
assert module_name not in sys.modules and source_path not in lock['common_sources']

extension_path = 'src/terl_mappo/batch01/extensions/qa_lazy_reward.py'
extension = snapshot / extension_path
extension.parent.mkdir()
extension.write_text('''import importlib
from terl_mappo.batch01.interfaces import Hooks
def factory(parameters):
    def transform(reward, info):
        module = importlib.import_module("cocap_voradj.models.continuous.box_actor")
        return reward + module.BoxActorConfig().a_max
    return Hooks(reward_transform=transform)
''')
delta = {'schema': 'terl.batch01.delta.v1', 'line': 'R1',
    'base': {'candidate_sha': CANDIDATE, 'parent_sha': lock['parent_sha'], 'lock_sha256': PIN},
    'config_changes': [], 'source_changes': [
        {'path': extension_path, 'before': None, 'after': file_hash(extension),
         'reason': 'QA lazy reward callback', 'science_impact': 'R1 reward only'}],
    'extensions': {'entrypoint': 'terl_mappo.batch01.extensions.qa_lazy_reward:factory'}}
resolved = verify_base(lock, delta, PIN)
verify_committed_pin(lock, delta)
hooks = importlib.import_module('terl_mappo.batch01.extensions.qa_lazy_reward').factory({})
runtime = assemble(resolved, 'cpu', hooks, 'R1')
manifest = runtime_manifest(lock, delta, resolved, runtime, validate_resources(snapshot / 'runtime_manifest.json'))
assert_manifest_sources(manifest)
before = file_hash(snapshot / source_path)
target = snapshot / source_path
text = target.read_text()
assert 'a_max: float = 0.4' in text
target.write_text(text.replace('a_max: float = 0.4', 'a_max: float = 0.35', 1))
after = file_hash(target)
batch, _ = runtime.collect(2)
metrics = runtime.update(batch)
assert_manifest_sources(manifest)
checkpoint = snapshot / 'qa_bound.pt'
try:
    save_bound(checkpoint, runtime, manifest, 2, 6, int(metrics['minibatch_updates']))
    loaded = load_bound(checkpoint, runtime, manifest)
    assert loaded['steps'] == 2
finally:
    checkpoint.unlink(missing_ok=True)
    checkpoint.with_suffix('.batch01.json').unlink(missing_ok=True)
uncovered = None
try:
    assert_loaded_sources_covered(source_inventory())
except ValueError as exc:
    uncovered = str(exc)
assert before != after and uncovered is not None
result = {'candidate_sha': CANDIDATE, 'canonical_lock_sha256': PIN,
    'declared_extension': extension_path, 'unlocked_dependency': source_path,
    'dependency_before_sha256': before, 'dependency_after_sha256': after,
    'prepare_manifest_passed': True, 'rollout_decisions': 2,
    'ppo_minibatches': metrics['minibatch_updates'], 'ppo_metrics_finite': all(np.isfinite(v) for v in metrics.values()),
    'checkpoint_manifest_source_check_passed_after_dependency_edit': True,
    'full_save_bound_and_load_bound_passed_after_dependency_edit': True,
    'explicit_post_rollout_coverage_check': uncovered,
    'raw_rewards_preserved_in_batch': 'raw_rewards' in batch,
    'classification': 'BASE_SOURCE_LOCK_LAZY_RUNTIME_ESCAPE',
    'fixture_only_changes': True, 'formal_training': False}
(HERE / 'lazy_dependency_repro.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
