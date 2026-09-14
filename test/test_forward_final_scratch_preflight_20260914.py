"""Scratch initialization and fail-closed preflight contract regressions."""
import pytest
import torch
import yaml
from tools.preflight_forward_final_scratch_20260914 import (
    load_contract, make_actor, make_trainer, tensor_hash, observe_episode,
    make_env, CoCapTrainer, set_global_config, seed_all, summarize,
)
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.models.continuous.local_entity_token_encoder import LegacyVorAdjFeatureBackbone
from cocap_voradj.control.apf import ApfAgent


def test_random_constructor_has_no_checkpoint_teacher_and_fresh_optimizer(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Teacher/weight loading is forbidden')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(CoCapIQN, '__init__', forbidden)
    monkeypatch.setattr(LegacyVorAdjFeatureBackbone, 'load_legacy_iqn_state_dict', forbidden)
    c = load_contract()
    trainer = make_trainer(c, 81)
    original = tensor_hash(trainer.actor.state_dict())
    assert original == tensor_hash(make_actor(c, 81).state_dict())
    assert original != tensor_hash(make_actor(c, 82).state_dict())
    assert all(p.requires_grad for p in trainer.actor.parameters())
    assert not trainer.actor_optimizer.state and not trainer.value_optimizer.state
    assert trainer.update_count == 0
    assert float(trainer.value_norm.debiasing_term) == 0
    assert float(trainer.value_norm.std) == 1
    torch.manual_seed(81)
    from cocap_voradj.models.small_step_ac import CentralValueNetwork
    assert tensor_hash(trainer.value.state_dict()) == tensor_hash(CentralValueNetwork(**c['critic']).state_dict())


@pytest.mark.parametrize('key,value', [('formal_training','PASS'), ('bc_dataset','data.npz'),
                                      ('teacher_q',True), ('imitation_loss',True),
                                      ('initialization','bc'), ('ce_speed_weight',0.)])
def test_contract_rejects_teacher_and_semantic_drift(tmp_path, key, value):
    c = load_contract()
    c[key] = value
    p = tmp_path / 'bad.yaml'
    p.write_text(yaml.safe_dump(c))
    with pytest.raises((AssertionError, ValueError)):
        load_contract(p)


@pytest.mark.parametrize('scene', ['mixed', 'coverage'])
@pytest.mark.parametrize('mode', ['sample', 'argmax'])
def test_eval_no_teacher_and_reproducible(monkeypatch, scene, mode):
    def forbidden(*args, **kwargs):
        raise AssertionError('Teacher forward/load forbidden')
    monkeypatch.setattr(CoCapIQN, 'forward', forbidden)
    monkeypatch.setattr(CoCapIQN, 'load', forbidden)
    actor = make_actor(load_contract(), 83)
    def run():
        seed_all(83)
        env, obs = make_env(scene, 83)
        class Host:
            pass
        host = Host()
        host.envs = {'eval': env}
        host.apf_agents = {'eval': [ApfAgent(e.a, e.w) for e in env.evaders]}
        def step_fn(actions):
            set_global_config(env.config)
            return env.step(actions, CoCapTrainer._evader_actions(host, 'eval'))
        return observe_episode(actor, env, obs, step_fn, scene, mode, max_steps=2)
    first, second = run(), run()
    assert first == second
    assert first['active_rows'] == 8
    assert first['phase_steps']['post_capture'] == 0
    assert first['phase_steps']['pure_coverage'] == (2 if scene == 'coverage' else 0)
    assert sum(first['action_histogram']) == 8
    assert first['entropy_sum']/8 > 2.19
    summary = summarize([first])[scene]
    assert summary['phase_rewards']['post_capture']['nonzero_density'] is None


@pytest.mark.parametrize('extra', [['--mode', 'formal'], ['--mode', 'smoke', '--steps', '25000']])
def test_cli_cannot_launch_formal_training(tmp_path, extra):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / 'must_not_exist'
    result = subprocess.run([sys.executable, str(root / 'tools/preflight_forward_final_scratch_20260914.py'),
                             '--output', str(out), *extra], capture_output=True, text=True)
    assert result.returncode == 2
    assert not out.exists()
