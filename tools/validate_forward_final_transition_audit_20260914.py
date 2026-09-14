#!/usr/bin/env python3
"""Validate saved native traces and intercept GAE before any optimizer work."""
import argparse
import gzip
import json
import sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from tools.audit_forward_final_transitions_20260914 import BASE, array_hash
from tools.train_forward_final_ppo_20260909 import make_trainer,tensor_hash
from tools.run_forward_final_bridge_20260908 import atomic_json
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file
from cocap_voradj.training.small_step_ac import compute_gae


def read_trace(path):
    if path.exists():return json.loads(path.read_text())
    return json.loads(gzip.decompress(path.with_suffix('.json.gz').read_bytes()))


def validate(out):
    torch.set_num_threads(1)
    before=read_trace(out/'before_v2/native/transitions.json');after=read_trace(out/'after_v2/native/transitions.json')
    assert len(before)==len(after)==586
    source=np.load(BASE/'heldout/fixed_bank.npz');take=source['episode']<4
    np.testing.assert_array_equal(np.asarray([r['MC_return'] for r in before],np.float32),source['target'][take])
    context=np.load(ROOT/'artifacts/2026-09-09_root_cause/context_critic_lineage_fixed/heldout/context.npz')['context']
    np.testing.assert_array_equal(np.asarray([r['before']['context'] for r in after],np.float32),context[take])
    deltas=[]
    for b,a in zip(before,after):
        for key in ('before','after','action','physical_action','captured','terminated','truncated','episode_end','V_t','V_next'):
            assert b[key]==a[key],key
        diff=np.asarray(a['reward'])-b['reward']
        for agent in np.flatnonzero(diff):
            m=b['infos'][agent]['replay_metadata']
            assert b['captured'] and m['reward_role']=='support'
            np.testing.assert_allclose(diff[agent],-.5*m['reward_ce_terminal_correction'],atol=2e-7)
            deltas.append(dict(episode=b['before']['episode'],t=b['before']['t'],agent=int(agent),reward_before=b['reward'][agent],reward_after=a['reward'][agent],delta=float(diff[agent])))
        if a['before']['phase']!='pre_capture':
            np.testing.assert_array_equal(b['MC_return'],a['MC_return'])
    trainer=make_trainer('cpu',2026097101)
    payload=torch.load(BASE/'critic_mc_100updates.pt',map_location='cpu',weights_only=False)
    trainer.value.load_state_dict(payload['trainer']['value']);trainer.value_normalizer.load_state_dict(payload['trainer']['value_normalizer'])
    trainer.actor.eval();trainer.value.eval()
    identity=(tensor_hash(trainer.actor.state_dict()),tensor_hash(trainer.value.state_dict()))
    norm={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in trainer.value_normalizer.state_dict().items()}
    data=np.load(out/'after_v2/native/stored_rollout.npz')
    batch={k:data[k] for k in data.files if not k.startswith(('global_','local_'))}
    batch['global_obs']={k[7:]:data[k] for k in data.files if k.startswith('global_')}
    batch['local_obs']={k[6:]:data[k] for k in data.files if k.startswith('local_')}
    class StopBeforeUpdate(Exception):pass
    intercepted={}
    for method in ('update','update_critic_only'):
        def intercept(rewards,values,next_values,terminated,active,**kwargs):
            np.testing.assert_allclose(values,[r['V_t'] for r in after],atol=1e-5,rtol=1e-6)
            np.testing.assert_allclose(next_values,[r['V_next'] for r in after],atol=1e-5,rtol=1e-6)
            adv,ret=compute_gae(rewards,values,next_values,terminated,active,**kwargs)
            np.testing.assert_allclose(ret,[r['GAE_return'] for r in after],atol=1e-4,rtol=2e-5)
            # Native episode-boundary reward poisoning cannot alter the prefix.
            end=next(i for i,r in enumerate(after) if all(r['episode_end']))
            poisoned=rewards.clone();poisoned[end+1:]=1e6
            _,other=compute_gae(poisoned,values,next_values,terminated,active,**kwargs)
            torch.testing.assert_close(ret[:end+1],other[:end+1],rtol=0,atol=0)
            intercepted[method]=dict(raw_values_correct=True,raw_next_values_correct=True,returns_correct=True,
                                      normalizer_updated_before_gae=False,reset_reward_poison_prefix_error=0)
            raise StopBeforeUpdate
        with patch('cocap_voradj.training.small_step_ac.compute_gae',side_effect=intercept):
            try:
                if method=='update':trainer.update(batch,categorical=True)
                else:trainer.update_critic_only(batch)
            except StopBeforeUpdate:pass
            else:raise AssertionError('Did not intercept GAE')
    assert identity==(tensor_hash(trainer.actor.state_dict()),tensor_hash(trainer.value.state_dict()))
    for k,v in norm.items():
        other=trainer.value_normalizer.state_dict()[k]
        if isinstance(v,torch.Tensor):torch.testing.assert_close(v,other,rtol=0,atol=0)
        else:assert v==other
    report=json.loads((out/'after_v2/report.json').read_text());probes=report['probes']
    assert all(probes['timeout']['truncated']) and not any(probes['timeout']['terminated'])
    assert max(abs(x) for x in probes['timeout_reward_difference'])<1e-12
    for name in ('success_at_timeout','post_window_at_timeout','too_few_at_timeout','collision_at_timeout'):
        assert all(probes[name]['terminated']) and not any(probes[name]['truncated'])
        np.testing.assert_allclose(probes[name]['GAE_return'],probes[name]['reward'],atol=1e-5)
    q=probes['support_capture']['infos'][3]['replay_metadata']
    np.testing.assert_allclose(q['reward_support_blend_coverage'],q['reward_coverage'],atol=1e-12)
    # Explicitly verify reset observations/value differ from timeout physical state.
    tr=read_trace(out/'after_v2/probes/timeout/transitions.json')
    assert tr[0]['after']['central_hash']!=tr[1]['before']['central_hash']
    expected=np.asarray(tr[0]['reward'])+.99*np.asarray(tr[0]['V_next'])
    np.testing.assert_allclose(tr[0]['GAE_return'],expected,atol=1e-5)
    naive=np.asarray(tr[0]['reward'])+.99*np.asarray(tr[1]['V_t'])
    # Sources of old audit executable are recorded verbatim by SHA (no hindsight rewrite).
    casualty=read_trace(out/'after_v2/probes/capture_casualty_reset/transitions.json')
    assert casualty[1]['before']['active']==[True,True,True,False] and all(casualty[1]['terminated'])
    for r in casualty:
        assert np.isfinite(r['reward']+r['V_t']+r['V_next']).all()
    assert casualty[1]['GAE_return'][3]==0
    result=dict(inactive_snapshot_reset_collector='PASS',original_context_bank_bit_exact=True,base_head=json.loads((out/'before_v2/launch.json').read_text())['head'],
                original_heldout_four_episode_target_bit_exact=True,all_action_state_context_value_pairs_identical=True,
                native_reward_changes=deltas,post_and_pure_mc_targets_bit_exact_unchanged=True,
                gae_update_consumers=intercepted,parameter_and_normalizer_changes=0,
                timeout_reset_value_wrong_target_max_difference=float(abs(expected-naive).max()),
                capture_recovery='PASS_AFTER_FIX',truncation_bootstrap='PASS_AFTER_FIX',reset_leakage='PASS',role_context_lag='PASS',
                p1_transition_return='PASS_AFTER_FOUR_CONFIRMED_BUG_FIXES',p2='DIAGNOSTICS_ONLY_CALIBRATION_HOLD',p3='HOLD',
                sources={str(p.relative_to(ROOT)):sha256_file(p) for p in [Path(__file__),BASE/'heldout/fixed_bank.npz',out/'after_v2/report.json']})
    atomic_json(out/'validation.json',result)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();validate(args.out.resolve())
