#!/usr/bin/env python3
"""Contract/PPO/checkpoint engineering evidence, not capability validation."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'tools')]
import numpy as np
import torch
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.small_step_ac import tensor_tree
from terl_mappo.batch02.runtime import sources as frozen_sources
from terl_mappo.batch02_a_long.runtime import CapabilityAdapter, build as baseline_build
from terl_mappo.batch02_native.runtime import NativeAdapter, build, collect, sources, save_checkpoint, load_checkpoint
from terl_mappo.run import atomic_json, file_hash, rng_state
from verify_batch02_e1_20261011 import same, digest_graph, robot_state, inject

CONFIGS = ROOT / 'configs/experiments/terl_mappo_batch02_native_20261011'
FREEZE = Path('/home/yjq/rl/CoCap1/ac-master-dag-20260921/artifacts/2026-10-11_batch02_e1_independent/freeze.json')


def continuation(trainer, adapter):
    batch, episodes = collect(trainer, adapter, 16)
    metrics = trainer.update(batch, categorical=True)
    return dict(batch=digest_graph(batch), episodes=digest_graph(episodes), metrics=metrics,
                trainer=digest_graph(trainer.state_dict()), runtime=digest_graph(adapter.state_dict()),
                rng=digest_graph(rng_state()))


def main(args):
    torch.set_num_threads(1)
    config = json.loads((CONFIGS / f'{args.arm.lower()}.json').read_text())
    adapter = NativeAdapter(config['learner']['seed'], args.arm)
    trainer = build(config, adapter, 'cpu')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.restore:
        load_checkpoint(args.output / 'resume.pt', trainer, adapter, config)
        actual = continuation(trainer, adapter)
        same(actual, json.loads((args.output / 'expected.json').read_text()))
        atomic_json(args.output / 'restored.json', dict(exact=True, continuation=actual))
        return
    same(frozen_sources(), json.loads(FREEZE.read_text())['source_manifest'])
    reference = Path('/home/yjq/rl/CoCap1/iqn-z05-independent-20260923/src/cocap_voradj/models/iqn.py')
    same(file_hash(reference), file_hash(ROOT / 'src/cocap_voradj/models/iqn.py'))
    baseline = CapabilityAdapter(config['learner']['seed'], config['baseline_arm'])
    original = baseline_build(config, baseline, 'cpu')
    same(digest_graph(original.value.state_dict()), digest_graph(trainer.value.state_dict()))
    shapes = {key:list(value.shape) for key,value in adapter.packed_observations().items()}
    actor_checks = dict(decentralized_interface=True, central_critic_initial_weights_identical=True)
    if args.arm == 'R3':
        same(digest_graph(original.actor.state_dict()), digest_graph(trainer.actor.state_dict()))
        actor_checks['R2_actor_initial_weights_identical'] = True
    if args.arm in ('N3','V3'):
        actor = trainer.actor
        iqn = CoCapIQN(actor.config).eval()
        iqn.load_state_dict(actor.backbone.state_dict(), strict=False)
        obs = tensor_tree(adapter.packed_observations(), 'cpu')
        with torch.no_grad():
            actual = actor.backbone(obs)
            expected = iqn.voradj_single_feature(obs, iqn.features(obs))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            logits = actor(obs)
            changed = {k:v.clone() for k,v in obs.items()}
            offset = 1
            for key in ('pursuers','evaders','obstacles'):
                width=changed[key].shape[1]
                invalid=~changed['masks'][:,offset:offset+width].bool()
                changed[key][invalid]=12345.
                offset += width
            torch.testing.assert_close(logits,actor(changed),rtol=1e-5,atol=1e-6)
            # A different batch row is not an information channel to a focal policy.
            other={k:v.clone() for k,v in obs.items()}
            other['self'][1:] += 5.
            torch.testing.assert_close(logits[0],actor(other)[0],rtol=0,atol=0)
            dead={k:v.clone() for k,v in obs.items()}
            dead['masks'][0]=False
            assert torch.isfinite(actor(dead)).all()
        for key in ('hidden_layer','layer_norm','output_layer'):
            same(digest_graph(getattr(actor,key).state_dict()),digest_graph(getattr(original.actor,key).state_dict()))
        actor_checks.update(successful_IQN_feature_exact=True, masked_padding_invariant=True,
                            separate_agent_row_invariant=True, inactive_row_finite=True,
                            TERL_AW9_head_initial_weights_identical=True)
    # Same scripted state/actions: check obs, reward, terminal, physical next state.
    decision_rng=np.random.RandomState(2116111900)
    reward_residual=0.
    role_counts={}
    for index in range(64):
        same(adapter.packed_observations(),baseline.packed_observations())
        before=digest_graph(adapter.state_dict()); rng=digest_graph(rng_state())
        adapter.packed_observations()
        same(before,digest_graph(adapter.state_dict())); same(rng,digest_graph(rng_state()))
        if index==32:
            adapter.env.total_steps=baseline.env.total_steps=200000
        actions=decision_rng.randint(0,9,len(adapter.env.pursuers))
        a,b=adapter.step(actions),baseline.step(actions)
        for field in range(5): same(a[field],b[field])
        same(a[5]['task_infos'],b[5]['task_infos']); same(a[5]['evader_actions'],b[5]['evader_actions'])
        same(robot_state(adapter.env),robot_state(baseline.env)); same(adapter.env.z_state,baseline.env.z_state)
        for row in a[5].get('reward_decomposition',[]):
            reward_residual=max(reward_residual,abs(row['residual']))
            role_counts[row['reward_role']]=role_counts.get(row['reward_role'],0)+1
        if a[4]: adapter.reset(); baseline.reset()
    fixtures=[]
    if args.arm in ('R3','V3'):
        reference=json.loads((FREEZE.parents[1]/'2026-10-11_cocap_mappo_batch02/reference_transitions.json').read_text())
        for case in reference['fixtures']:
            candidate=NativeAdapter(config['learner']['seed'],args.arm)
            control=CapabilityAdapter(config['learner']['seed'],config['baseline_arm'])
            inject(candidate.env,case['input']); inject(control.env,case['input'])
            candidate.observations=candidate.env.get_observations(); control.observations=control.env.get_observations()
            a,b=candidate.step(case['input']['pursuer_actions']),control.step(case['input']['pursuer_actions'])
            for field in range(5): same(a[field],b[field])
            same(a[5]['task_infos'],b[5]['task_infos']);same(robot_state(candidate.env),robot_state(control.env))
            residual=max(abs(row['residual']) for row in a[5]['reward_decomposition'])
            reward_residual=max(reward_residual,residual)
            fixtures.append(dict(name=case['name'],input=case['input'],reward_unchanged=True,
                                 reconstruction_residual_max=residual,ledger=a[5]['reward_decomposition']))
        atomic_json(args.output/'reward_transition_fixtures.json',fixtures)
    adapter=NativeAdapter(config['learner']['seed'],args.arm); trainer=build(config,adapter,'cpu')
    batch,episodes=collect(trainer,adapter,256)
    metrics=trainer.update(batch,categorical=True)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics['actor_update_l2']>0 and metrics['minibatch_updates']>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainer.actor.parameters())
    digest=save_checkpoint(args.output/'resume.pt',trainer,adapter,config,256,int(batch['active_mask'].sum()),int(metrics['minibatch_updates']))
    atomic_json(args.output/'expected.json',continuation(trainer,adapter))
    subprocess.run([sys.executable,str(Path(__file__)), '--arm',args.arm,'--output',str(args.output),'--restore'],check=True,cwd=ROOT)
    report=dict(status='ENGINEERING_PASS',classification='PROVISIONAL',arm=args.arm, config_sha256=file_hash(CONFIGS/f'{args.arm.lower()}.json'),
        sources=sources(),input_shapes=shapes,actor_checks=actor_checks,
        actor_parameters=sum(p.numel() for p in trainer.actor.parameters()),
        critic_parameters=sum(p.numel() for p in trainer.value.parameters()),baseline_actor_parameters=sum(p.numel() for p in original.actor.parameters()),
        scripted_transition_parity=64,original_schedule_pre_and_post200k=True, reward_reconstruction_residual_max=reward_residual,
        reward_role_counts=role_counts, fixed_reward_fixtures=len(fixtures), PPO_metrics=metrics,checkpoint_sha256=digest, fresh_process_resume_exact=True,
        evaluator_compatibility='to be checked by pilot with unchanged native episode protocol',
        learning_capability_claim=False,independent_validation=False)
    atomic_json(args.output/'engineering_verification.json',report)
    print(json.dumps({key:value for key,value in report.items() if key not in ('sources','PPO_metrics')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['N3','R3','V3'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--restore',action='store_true')
    main(parser.parse_args())
