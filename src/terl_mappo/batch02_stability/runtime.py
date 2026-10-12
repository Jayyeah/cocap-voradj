"""Reuse each baseline runtime, checkpoint schema, evaluator, and PPO equations."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch
from terl_mappo.batch02.cocap_env import ROOT
from terl_mappo.batch02_a_long import runtime as original
from terl_mappo.batch02_a_long import evaluation as original_evaluation
from terl_mappo.batch02_native import runtime as native
from terl_mappo.batch02_native import evaluation as native_evaluation
from terl_mappo.run import file_hash, rng_state

sys.path.insert(0, str(ROOT / 'tools'))
from verify_batch02_e1_20261011 import digest_graph

ARTIFACTS = ROOT / 'artifacts/2026-10-12_batch02_stability'
CONFIGS = ROOT / 'configs/experiments/batch02_stability_20261012'


def profile(arm):
    if arm not in ('R2', 'R3', 'V3'):
        raise ValueError('only registered R2/R3/V3 stability continuations')
    runtime = original if arm == 'R2' else native
    evaluator = original_evaluation if arm == 'R2' else native_evaluation
    return runtime, evaluator


def config_and_plan(arm):
    config = json.loads((CONFIGS / f'{arm.lower()}.json').read_text())
    plan = json.loads((ARTIFACTS / 'run_plan.json').read_text())['runs'][arm]
    baseline = json.loads((ARTIFACTS / f'{arm}_baseline_config.json').read_text())
    allowed = copy.deepcopy(baseline)
    allowed['learner']['ppo']['target_kl'] = .01
    if config != allowed or baseline['learner']['ppo']['target_kl'] != .02:
        raise ValueError('only recovered P1 target_kl .02 -> .01 is authorized')
    return config, baseline, plan


def wiring_sources():
    paths = list((ROOT / 'src/terl_mappo/batch02_stability').glob('*.py'))
    paths += list((ROOT / 'tools').glob('*batch02_stability*20261012.py'))
    paths += [ROOT / 'tools/verify_batch02_e1_20261011.py']
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)}


def state_hashes(trainer, adapter):
    return {**{key:digest_graph(value) for key,value in trainer.state_dict().items()},
            'runtime':digest_graph(adapter.state_dict()), 'rng':digest_graph(rng_state())}


def fork_parent(arm, device):
    config, baseline, plan = config_and_plan(arm)
    runtime, evaluator = profile(arm)
    path = Path(plan['parent_checkpoint']['path'])
    if file_hash(path) != plan['parent_checkpoint']['sha256']:
        raise ValueError('fixed500k parent hash changed')
    adapter_type = runtime.CapabilityAdapter if arm == 'R2' else runtime.NativeAdapter
    adapter = adapter_type(baseline['learner']['seed'], arm)
    trainer = runtime.build(baseline, adapter, device)
    # Strict baseline loader validates all original sources and the complete config.
    payload = runtime.load_checkpoint(path, trainer, adapter, baseline)
    if payload['decisions'] != 500000:
        raise ValueError('fixed500k required; never select best')
    expected = {**{key:digest_graph(value) for key,value in payload['trainer'].items()},
                'runtime':digest_graph(payload['runtime']), 'rng':digest_graph(payload['rng'])}
    before = state_hashes(trainer, adapter)
    if before != expected:
        raise ValueError('strict parent actor/V/Adam/ValueNorm/env/RNG restore mismatch')
    if adapter.contract() != plan['contract']:
        raise ValueError('fixed parent task contract changed')
    trainer.config = replace(trainer.config, target_kl=config['learner']['ppo']['target_kl'])
    after = state_hashes(trainer, adapter)
    if before != after:
        raise ValueError('P1 fork changed full state')
    if file_hash(Path(evaluator.__file__)) != plan['evaluator_sha256']:
        raise ValueError('original evaluator changed')
    receipt = dict(strict_resume=True, parent_checkpoint=str(path),
                   parent_sha256=plan['parent_checkpoint']['sha256'], decisions=500000,
                   before=before, after=after, complete_state_unchanged=True,
                   baseline_target_kl=.02, live_target_kl=trainer.config.target_kl,
                   only_config_delta=plan['scientific_delta'],
                   evaluator_version=evaluator.EVALUATOR_VERSION,
                   evaluator_sha256=plan['evaluator_sha256'],
                   task_contract=adapter.contract(), classification='PROVISIONAL')
    return trainer, adapter, config, payload, receipt


def restore(arm, checkpoint, device):
    config, _, _ = config_and_plan(arm)
    runtime, _ = profile(arm)
    adapter_type = runtime.CapabilityAdapter if arm == 'R2' else runtime.NativeAdapter
    adapter = adapter_type(config['learner']['seed'], arm)
    trainer = runtime.build(config, adapter, device)
    payload = runtime.load_checkpoint(checkpoint, trainer, adapter, config)
    return trainer, adapter, config, payload


def save(arm, path, trainer, adapter, config, step, agent_steps, optimizer_steps):
    # Preserve the original schema/sources so the original evaluator is reused verbatim.
    return profile(arm)[0].save_checkpoint(path, trainer, adapter, config, step, agent_steps, optimizer_steps)
