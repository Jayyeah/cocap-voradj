"""QA B1/B2/B3/B4/Q1 regressions; all disposable data stays under /home/yjq."""
import copy
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from terl_mappo.batch01 import contracts
from terl_mappo.batch01.source_guard import SourceGuard, SourceViolation
from terl_mappo.batch01.interfaces import (Hooks, ActionBackend, ActionCapabilities,
                                         Runtime, assemble, validate_adapter)
from terl_mappo.batch01.evaluation import METRIC_DEFINITIONS, isolated_rng
from terl_mappo.model import TERLActor
from terl_mappo.native import NativeStage1, pack_local
from terl_mappo.run import rng_state, file_hash
from cocap_voradj.training.small_step_ac import compute_gae, explained_variance, tensor_tree
from test_terl_mappo_batch01_base_20261009 import same, config, committed_lock

BOX = 'cocap_voradj.models.continuous.' + 'box_actor'
BOX_PATH = 'src/' + BOX.replace('.', '/') + '.py'


def test_q1_production_lock_generator_stays_core_only(monkeypatch):
    monkeypatch.setattr(contracts, 'git', lambda *args: 'audit/terl-mappo-batch01-base-qa-20261009')
    with pytest.raises(ValueError, match='authorized Core branch'): contracts.create_lock()
    lock = committed_lock()
    assert lock['status'] == 'BATCH01_BASE_CANDIDATE_V2'


@pytest.fixture
def source_snapshot(tmp_path):
    root = tmp_path / 'snapshot'
    for relative in set(committed_lock()['common_sources']) | {contracts.LOCK_PATH, BOX_PATH}:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(contracts.ROOT / relative, target)
    yield root
    shutil.rmtree(root)


_SOURCE_REPRO = r'''
import copy, importlib, json, sys
from pathlib import Path
import numpy as np
import torch
from terl_mappo.batch01.contracts import ROOT, LOCK_PATH, ANCHOR_CONFIG, fingerprint, verify_base, declared_sources
from terl_mappo.batch01.source_guard import SourceGuard, SourceViolation
from terl_mappo.batch01.interfaces import Hooks, assemble
from terl_mappo.batch01.checkpoints import save_bound, load_bound, assert_manifest_sources
from terl_mappo.run import file_hash
torch.set_num_threads(1)
case = sys.argv[1]
lock=json.loads((ROOT/LOCK_PATH).read_text())
config=json.loads((ROOT/ANCHOR_CONFIG).read_text())
box='cocap_voradj.models.continuous.'+'box_actor'
box_path='src/'+box.replace('.','/')+'.py'
original_box_path=box_path
if case != 'original_qa_box_drift':
    box=box.replace('box_actor','qa_dynamic_box')
    box_path='src/'+box.replace('.','/')+'.py'
    (ROOT/box_path).write_bytes((ROOT/original_box_path).read_bytes())
extension=ROOT/'src/terl_mappo/batch01/extensions/qa_lazy_reward.py'
extension.parent.mkdir(parents=True,exist_ok=True)
literal = case == 'preflight'
expression = repr(box) if literal else "parameters['module']"
extension.write_text('import importlib\nfrom terl_mappo.batch01.interfaces import Hooks\n'
    'def factory(parameters):\n    def transform(reward, info):\n'
    '        module=importlib.import_module('+expression+')\n'
    '        return reward+module.BoxActorConfig().a_max\n'
    '    return Hooks(reward_transform=transform)\n')
changes=[{'path':str(extension.relative_to(ROOT)), 'before':None,'after':file_hash(extension),
          'reason':'QA R1 reproduction','science_impact':'diagnostic only'}]
declared=case in {'legal','disk_drift','checkpoint_save','checkpoint_load','executed_drift'}
if declared:
    changes.append({'path':box_path,'before':lock['common_sources'].get(box_path),'after':file_hash(ROOT/box_path),
                    'reason':'explicit dynamic dependency','science_impact':'diagnostic only'})
delta={'schema':'terl.batch01.delta.v1','line':'R1',
       'base':{'parent_sha':lock['parent_sha'],'candidate_sha':'a'*40,'lock_sha256':fingerprint(lock)},
       'config_changes':[],'source_changes':changes,'extensions':{}}
verify_base(lock,delta,fingerprint(lock))
guard=SourceGuard(declared_sources(lock,delta))
with guard.scope():
    hooks=importlib.import_module('terl_mappo.batch01.extensions.qa_lazy_reward').factory({'module':box})
def rejected(call):
    try: call()
    except SourceViolation as error: return str(error)
    raise AssertionError('source escape accepted: '+case)
if case == 'preflight':
    result=rejected(lambda: assemble(config,hooks=hooks,line='R1',source_guard=guard))
else:
    runtime=assemble(config,hooks=hooks,line='R1',source_guard=guard)
    manifest={'base':delta['base'],'common_source_hashes':dict(guard.expected)}
    path=ROOT/'runs/probe.pt'
    if case == 'original_qa_box_drift':
        p=ROOT/box_path
        p.write_text(p.read_text().replace('a_max: float = 0.4','a_max: float = 0.35'))
        result=rejected(lambda:runtime.collect(2))
        assert runtime.trainer.update_count == 0
        rejected(lambda:save_bound(path,runtime,manifest,2,6,0))
        rejected(lambda:load_bound(path,runtime,manifest))
    elif case == 'optimizer_import':
        from terl_mappo.run import collect
        batch,_=collect(runtime.trainer,runtime.adapter,2)
        original=runtime.trainer.actor.forward
        def forward(obs):
            importlib.import_module(box)
            return original(obs)
        runtime.trainer.actor.forward=forward
        steps=[]
        runtime.trainer.actor_optimizer.register_step_pre_hook(lambda *args:steps.append(True))
        result=rejected(lambda:runtime.update(batch))
        assert steps == [] and runtime.trainer.update_count == 0 and box not in sys.modules
    elif case == 'foreign_origin':
        foreign=ROOT.parent/'foreign.py'
        foreign.write_text('EXECUTED=True\n')
        spec=importlib.util.spec_from_file_location('qa_foreign',foreign)
        def execute():
            with guard.scope(): spec.loader.exec_module(importlib.util.module_from_spec(spec))
        result=rejected(execute)
    elif declared:
        batch,_=runtime.collect(2)
        if case == 'legal':
            metrics=runtime.update(batch)
            assert np.isfinite(list(metrics.values())).all()
            save_bound(path,runtime,manifest,2,6,int(metrics['minibatch_updates']))
            load_bound(path,runtime,manifest)
            result='LEGAL_DECLARED_IMPORT_UPDATE_SAVE_LOAD_PASS'
        else:
            save_bound(path,runtime,manifest,2,6,0)
            p=ROOT/box_path
            before=p.read_text()
            p.write_text(before.replace('a_max: float = 0.4','a_max: float = 0.35'))
            if case == 'executed_drift':
                expected=dict(guard.expected); expected[box_path]=file_hash(p)
                guard=SourceGuard(expected)
                runtime.source_guard=guard
            operation={'disk_drift':lambda:runtime.update(batch),
                       'executed_drift':lambda:runtime.update(batch),
                       'checkpoint_save':lambda:save_bound(path,runtime,manifest,2,6,0),
                       'checkpoint_load':lambda:load_bound(path,runtime,manifest)}[case]
            result=rejected(operation)
            assert runtime.trainer.update_count == 0
    elif case == 'update_boundary':
        importlib.import_module(box)
        result=rejected(lambda:runtime.update({}))
        assert runtime.trainer.update_count == 0
    else:
        if case == 'origin':
            module=sys.modules['terl_mappo.model']
            module.__file__=str(ROOT/'runs/fake_model.py')
            result=rejected(lambda:runtime.collect(1))
        elif case == 'symlink':
            (ROOT/'src/escape.py').symlink_to('/usr/local/lib/escape.py')
            result=rejected(lambda:guard.check_path(ROOT/'src/escape.py'))
        else:
            result=rejected(lambda:runtime.collect(2))
            assert box not in sys.modules
            assert runtime.trainer.update_count == 0
            rejected(lambda:runtime.update({}))
            rejected(lambda:save_bound(path,runtime,manifest,2,6,0))
            rejected(lambda:load_bound(path,runtime,manifest))
            assert not path.exists()
if guard.failure is not None:
    assert guard.failure_path.is_file()
    assert json.loads(guard.failure_path.read_text())['status']=='RUN_FAILED_SOURCE_CONTRACT'
print(json.dumps({'case':case,'result':result,'failure_latched':guard.failure is not None}))
'''


@pytest.mark.parametrize('case', ['preflight', 'lazy_import', 'legal', 'disk_drift',
                                 'update_boundary', 'checkpoint_save', 'checkpoint_load',
                                 'origin', 'symlink', 'executed_drift', 'optimizer_import',
                                 'foreign_origin', 'original_qa_box_drift'])
def test_b1_dynamic_source_guards_fail_closed_and_allow_explicit_delta(source_snapshot, case):
    env = dict(os.environ, PYTHONPATH=str(source_snapshot / 'src') + ':' + str(source_snapshot) + ':' + os.environ.get('PYTHONPATH', ''),
               CUDA_VISIBLE_DEVICES='', TMPDIR=str(source_snapshot / 'runs'))
    (source_snapshot / 'runs').mkdir()
    result = subprocess.run([sys.executable, '-c', _SOURCE_REPRO, case], cwd=source_snapshot,
                            env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(result.stdout.strip().splitlines()[-1])
    assert record['failure_latched'] == (case != 'legal')
    (source_snapshot.parent / 'result.json').write_text(json.dumps(record, indent=2) + '\n')


def continuous_backend():
    return ActionBackend('tanh_gaussian_aw', 'physical_differential_mc_nats',
                         'tanh_and_scale_corrected_density', False)


def test_b2_lambda_native_adapter_rejected_during_startup():
    with pytest.raises(ValueError, match='actual adapter capabilities'):
        assemble(config(), hooks=Hooks(env_factory=lambda seed: NativeStage1(seed),
                                        backend=continuous_backend()), line='C0')


class AdapterProbe:
    """Test double for the capability seam; not a C0 environment implementation."""
    def __init__(self):
        self.observations = NativeStage1(9).observations
        self.calls = 0
    def action_capabilities(self):
        return ActionCapabilities('tanh_gaussian_aw', 'physical_aw', ((-.4, .4), (-np.pi/6, np.pi/6)))
    def validate_actions(self, actions):
        return continuous_backend().validate(actions, 3)
    def step(self, actions):
        self.validate_actions(actions)
        self.calls += 1
        return None, np.zeros(3), np.zeros(3, bool), np.zeros(3, bool), False, {}


class BadConsumerProbe(AdapterProbe):
    def step(self, actions):
        int(actions[0])


def test_b2_actual_behavior_probe_checks_consumer_and_preserves_live_state():
    guard = SourceGuard.committed()
    adapter = AdapterProbe()
    before = rng_state()
    validate_adapter(adapter, continuous_backend(), guard)
    same(before, rng_state())
    assert adapter.calls == 0
    with pytest.raises(ValueError, match='behavior probe'):
        validate_adapter(BadConsumerProbe(), continuous_backend(), guard)
    with pytest.raises(ValueError, match='actual adapter capabilities'):
        validate_adapter(adapter, ActionBackend(), guard)


class Features(TERLActor):
    decision_feature_dim = 256
    def forward(self, obs):
        return self.layer_norm(torch.relu(self.hidden_layer(self.encode_entities(obs))))


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_b3_density_guard_preserves_all_rng_and_replays_exact_latent(device):
    if device.startswith('cuda') and not torch.cuda.is_available(): pytest.skip('CUDA requires Master gate/available device')
    expected = dict(committed_lock()['common_sources'])
    expected[BOX_PATH] = file_hash(contracts.ROOT / BOX_PATH)
    guard = SourceGuard(expected)
    try:
        with guard.scope(): box = importlib.import_module(BOX)
        actor = box.SquashedGaussianAccelerationAngularVelocityActor(
            Features(256, 8, 4), box.BoxActorConfig(hidden_dim=256)).to(device).eval()
        local = tensor_tree(pack_local(NativeStage1(9).observations), device)
        with torch.no_grad(): actions, logp, latent = actor.sample(local)
        with isolated_rng(), torch.no_grad():
            replay = actor.evaluate_latent(local, latent)[0]
            density = actor.log_prob_from_latent(local, latent)
        torch.testing.assert_close(replay, density, atol=0, rtol=0)
        torch.testing.assert_close(torch.exp(replay-logp), torch.ones_like(logp), atol=1e-6, rtol=0)
        batch = {'local_obs': {k: v.cpu().numpy()[None] for k, v in local.items()},
                 'active_mask': np.ones((1, 3), bool), 'latent': latent.cpu().numpy()[None],
                 'log_prob': logp.cpu().numpy()[None]}
        before = rng_state()
        def update(batch, categorical):
            same(before, rng_state())
            assert categorical is False
            return {'entered_learner': 1.}
        trainer = SimpleNamespace(actor=actor, device=torch.device(device), update=update)
        runtime = Runtime(trainer, None, Hooks(backend=continuous_backend()), config(), guard)
        assert runtime.update(batch)['entered_learner'] == 1.
        same(before, rng_state())
        bad = copy.deepcopy(batch); bad['log_prob'] += 1.
        with pytest.raises(ValueError, match='behavior log-prob mismatch'): runtime.update(bad)
        same(before, rng_state())
    finally:
        sys.modules.pop(BOX, None)


def test_b4_ev_matches_post_update_denormalized_value_and_pre_update_targets():
    assert 'post-update denormalized V' in METRIC_DEFINITIONS['critic_ev']
    assert 'pre-update GAE raw return targets' in METRIC_DEFINITIONS['critic_ev']
    runtime = assemble(config())
    batch, _ = runtime.collect(16)
    t = runtime.trainer
    active = torch.as_tensor(batch['active_mask'], dtype=torch.bool)
    _, returns = compute_gae(torch.as_tensor(batch['rewards'], dtype=torch.float32),
        t._denormalize_values(torch.as_tensor(batch['values'])),
        t._denormalize_values(torch.as_tensor(batch['next_values'])),
        torch.as_tensor(batch['terminated']), active, gamma=t.config.gamma, gae_lambda=t.config.gae_lambda,
        truncated=torch.as_tensor(batch['truncated']), episode_end=torch.as_tensor(batch['episode_end']))
    metrics = runtime.update(batch)
    with torch.no_grad(): prediction = t._denormalize_values(t.value(tensor_tree(batch['global_obs'], 'cpu')))
    assert metrics['explained_variance'] == explained_variance(prediction[active], returns[active])
