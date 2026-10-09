"""CPU-only C0 startup routing and RNG-guard probes in a disposable snapshot."""
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
    snapshot = ROOT / 'runs/a3_qa02/continuous_guards_snapshot'
    if snapshot.exists(): raise SystemExit('Refuse to reuse existing snapshot')
    snapshot.mkdir(parents=True)
    archive = subprocess.check_output(['git', 'archive', CANDIDATE, 'src', 'vendor',
        'configs/experiments/terl_mappo_20261008',
        'configs/experiments/terl_mappo_batch01_20261009', 'tools/batch01_base_20261009.py'])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar: tar.extractall(snapshot, filter='data')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(snapshot / 'src') + ':/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/.runtime-deps')
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', str(snapshot)],
        cwd=snapshot, env=env, capture_output=True, text=True)
    (HERE / 'continuous_guards_repro.txt').write_text(result.stdout + result.stderr)
    print(result.stdout, end=''); print(result.stderr, end='', file=sys.stderr)
    raise SystemExit(result.returncode)

snapshot = Path(sys.argv[-1]); sys.path.insert(0, str(snapshot / 'src'))
import importlib
import numpy as np
import torch
from terl_mappo.batch01.contracts import source_inventory, verify_base, verify_committed_pin, validate_resources
from terl_mappo.batch01.interfaces import assemble
from terl_mappo.batch01.provenance import runtime_manifest
from terl_mappo.native import pack_local
from terl_mappo.run import file_hash

torch.set_num_threads(1)
lock = json.loads((snapshot / 'configs/experiments/terl_mappo_batch01_20261009/base_lock.json').read_text())
path = snapshot / 'src/terl_mappo/batch01/extensions/qa_continuous.py'
path.parent.mkdir()
path.write_text('''import torch
from terl_mappo.native import NativeStage1
from terl_mappo.model import TERLActor
from terl_mappo.batch01.interfaces import Hooks, ActionBackend
from cocap_voradj.models.continuous.box_actor import BoxActorConfig, SquashedGaussianAccelerationAngularVelocityActor
class Features(TERLActor):
    def forward(self, obs):
        return self.layer_norm(torch.relu(self.hidden_layer(self.encode_entities(obs))))
def actor_factory(h, heads, layers, seed, masked_pool):
    encoder = Features(h, heads, layers, seed, masked_pool)
    encoder.decision_feature_dim = h
    return SquashedGaussianAccelerationAngularVelocityActor(encoder, BoxActorConfig(hidden_dim=h))
def factory(parameters):
    # This erroneous wrapper passes the factory-object identity check.
    return Hooks(actor_factory=actor_factory, env_factory=lambda seed: NativeStage1(seed),
        backend=ActionBackend("tanh_gaussian_aw", "physical_differential_mc_nats",
            "tanh_and_scale_corrected_density", False))
''')
sources = source_inventory()
delta = {'schema': 'terl.batch01.delta.v1', 'line': 'C0',
    'base': {'candidate_sha': CANDIDATE, 'parent_sha': lock['parent_sha'], 'lock_sha256': PIN},
    'config_changes': [], 'source_changes': [
        {'path': p, 'before': None, 'after': h, 'reason': 'QA continuous probe dependency',
         'science_impact': 'C0 declared actor/action interface'}
        for p, h in sources.items() if p not in lock['common_sources']],
    'extensions': {'entrypoint': 'terl_mappo.batch01.extensions.qa_continuous:factory'}}
resolved = verify_base(lock, delta, PIN); verify_committed_pin(lock, delta)
hooks = importlib.import_module('terl_mappo.batch01.extensions.qa_continuous').factory({})
runtime = assemble(resolved, 'cpu', hooks, 'C0')
manifest = runtime_manifest(lock, delta, resolved, runtime, validate_resources(snapshot / 'manifest.json'))
error = None
try: runtime.collect(1)
except (TypeError, ValueError) as exc: error = repr(exc)
assert error is not None

local = pack_local(runtime.adapter.observations)
state = runtime.hooks.state_encoder(runtime.adapter.env)
actions, logp, latent, _ = runtime.trainer.act(local, {k: v[None] for k, v in state.items()})
batch = {'local_obs': {k: v[None] for k, v in local.items()},
    'active_mask': np.ones((1, len(actions)), bool), 'latent': latent[None], 'log_prob': logp[None]}
before = torch.get_rng_state().clone()
entered = []
runtime.trainer.update = lambda batch, categorical: entered.append(torch.get_rng_state().clone()) or {}
runtime.update(batch)
changed = not torch.equal(before, entered[0])
assert changed
result = {'candidate_sha': CANDIDATE, 'canonical_lock_sha256': PIN,
    'declared_extra_sources': delta['source_changes'], 'startup_manifest_accepted': True,
    'actual_adapter_class': type(runtime.adapter).__qualname__,
    'declared_backend': manifest['active_runtime_values']['action_backend'],
    'first_continuous_rollout_error': error, 'zero_update_likelihood_check_passed': True,
    'continuous_density_guard_changed_torch_rng_before_learner': changed,
    'optimizer_updates_executed': 0, 'fixture_only_changes': True, 'formal_training': False}
(HERE / 'continuous_guards_repro.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
