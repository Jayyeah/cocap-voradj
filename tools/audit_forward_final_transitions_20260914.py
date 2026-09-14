#!/usr/bin/env python3
"""Frozen-policy native-collector boundary audit; never calls an optimizer."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
import torch
from cocap_voradj.training.forward_final import check_env
from cocap_voradj.training.forward_final_value_context import value_context, NAMES
from cocap_voradj.training.small_step_ac import compute_gae, tensor_tree, flatten_local
from tools.train_forward_final_ppo_20260909 import (
    FinalMissionStream, make_trainer, collect_transition, local_and_global,
    tensor_hash, empty_rollout, stack_rollout, BC,
)
from tools.evaluate_forward_final_critic_20260909 import full_returns
from tools.run_forward_final_bridge_20260908 import atomic_json
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file
from tools.fit_forward_final_context_critic_20260909 import RecordedSourceStream

BASE = ROOT / 'artifacts/2026-09-09_forward_final_d/fixed_mc_critic'


def array_hash(tree):
    h = hashlib.sha256()
    for k, v in sorted(tree.items()):
        a = np.ascontiguousarray(v)
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def snapshot(stream):
    env = stream.env
    local, central = local_and_global(stream)
    labels = list(env.last_task_labels)
    data = env._capture_voronoi_map()
    roles = [env._task_reward_role(i, labels, data) for i in range(4)]
    return dict(episode=stream.episode, t=env.episode_step,
                phase='pure_coverage' if not env.evaders else 'pre_capture' if any(not e.deactivated for e in env.evaders) else 'post_capture',
                roles=roles, labels=labels, context=value_context(env).tolist(),
                obs_pursuing=local['self'][:, -1].tolist(),
                local_hash=array_hash(local), central_hash=array_hash(central),
                active=central['active_mask'].tolist(), reset_source=stream.last_reset_source[stream.task],
                selected_capture_source=getattr(stream, 'selected_capture_source_hash', None))


class AuditStream(RecordedSourceStream):
    def step(self, indices):
        outcome = super().step(indices)
        self.audit_after = snapshot(self)
        self.audit_outcome = outcome
        self.audit_next_central = local_and_global(self)[1]
        self.audit_events = copy.deepcopy(self.env.last_capture_events)
        return outcome


def capture_fixture(stream):
    env = stream.env
    for p, angle in zip(env.pursuers[:3], [0, 2*np.pi/3, 4*np.pi/3]):
        p.x, p.y = 50+7*np.cos(angle), 50+7*np.sin(angle)
        p.speed = 0.; p.velocity = np.zeros(2)
    env.pursuers[3].x, env.pursuers[3].y = 10., 10.
    env.evaders[0].x, env.evaders[0].y = 50., 50.
    env.evaders[0].speed = 0.; env.evaders[0].velocity = np.zeros(2)
    env.obstacles[0].x, env.obstacles[0].y = 90., 90.
    env._invalidate_voronoi_cache()
    env.last_task_labels = env._task_labels_from_map(env._capture_voronoi_map(), update_effective=True)
    stream.observations = env.get_observations()


def converge_fixture(stream):
    env = stream.env
    for _ in range(60):
        data = env._coverage_voronoi_map()
        for i, p in enumerate(env.pursuers):
            p.x, p.y = map(float, data['centroids'][('pursuer', i)])
            p.speed = 0.; p.velocity = np.zeros(2)
        env._invalidate_voronoi_cache()
    assert env._voradj_coverage_geometry(env._coverage_voronoi_map(), strict=True)['converged_now']
    env.distribution_hold_steps = 29
    stream.observations = env.get_observations()


def traced(trainer, stream):
    before = snapshot(stream)
    row, done = collect_transition(trainer, stream)
    after, outcome = stream.audit_after, stream.audit_outcome
    meta = [i['replay_metadata'] for i in outcome.infos]
    assert before['central_hash'] == array_hash(row['global_obs'])
    assert before['local_hash'] == array_hash(row['local_obs'])
    assert before['roles'] == [m['reward_role'] for m in meta]
    assert before['labels'] == [m['task_label'] for m in meta]
    assert all(before['phase'] == m['phase'] for m in meta)
    assert after['labels'] == [m['next_task_label'] for m in meta]
    np.testing.assert_array_equal(row['local_obs']['self'][:, -1], [x == 'capture' for x in before['labels']])
    np.testing.assert_array_equal(row['active_mask'], before['active'])
    np.testing.assert_array_equal(row['episode_end'], outcome.dones)
    for i, m in enumerate(meta):
        np.testing.assert_allclose(row['rewards'][i], m['reward_total'], rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(m['reward_total'], sum(m[k] for k in ('reward_capture', 'reward_coverage', 'reward_safety', 'reward_terminal')), atol=1e-9)
    with torch.no_grad():
        expected = trainer.value(tensor_tree({k:v[None] for k,v in stream.audit_next_central.items()}, trainer.device))[0].cpu().numpy()
    np.testing.assert_array_equal(row['next_values'], expected)
    def raw(a):
        return trainer._denormalize_values(torch.as_tensor(a, device=trainer.device)).cpu().numpy().tolist()
    record = dict(before=before, after=after, action=row['latent'].tolist(), physical_action=row['actions'].tolist(),
                  reward=row['rewards'].tolist(), captured=bool(stream.audit_events), capture_events=stream.audit_events,
                  terminated=row['terminated'].tolist(), truncated=row['truncated'].tolist(), episode_end=row['episode_end'].tolist(),
                  V_t=raw(row['values']), V_next=raw(row['next_values']),
                  V_t_normalized=row['values'].tolist(), V_next_normalized=row['next_values'].tolist(),
                  infos=outcome.infos, completed=done)
    return row, record


def annotate(rows, records, trainer, out):
    rollout = empty_rollout()
    for row in rows:
        for k,v in row.items(): rollout[k].append(v)
    batch = stack_rollout(rollout)
    tensors = {k:torch.as_tensor(batch[k]) for k in ('rewards','terminated','truncated','episode_end','active_mask')}
    v = torch.tensor([r['V_t'] for r in records]); nv = torch.tensor([r['V_next'] for r in records])
    adv, ret = compute_gae(tensors['rewards'], v, nv, tensors['terminated'], tensors['active_mask'],
                           gamma=.99, gae_lambda=.95, truncated=tensors['truncated'], episode_end=tensors['episode_end'])
    # Independent forward sum of TD residuals stops at the first end/inactive row.
    ref = np.zeros_like(ret.numpy(), dtype=np.float64)
    for t in range(len(rows)):
        for a in range(4):
            if not batch['active_mask'][t,a]: continue
            weight = 1.; total = 0.
            for j in range(t, len(rows)):
                if not batch['active_mask'][j,a]: break
                total += weight*(float(batch['rewards'][j,a])+.99*(not batch['terminated'][j,a])*float(nv[j,a])-float(v[j,a]))
                if batch['episode_end'][j,a]: break
                weight *= .99*.95
            ref[t,a] = total+float(v[t,a])
    np.testing.assert_allclose(ret, ref, rtol=2e-5, atol=8e-5)
    # Flattening preserves the same joint-time/focal-agent row for actor and V.
    flat, shape = flatten_local(tensor_tree(batch['local_obs'], 'cpu'))
    np.testing.assert_array_equal(flat['self'], batch['local_obs']['self'].reshape(-1,9))
    assert shape == (len(rows),4)
    for start in range(0,len(rows),256):
        stop=min(start+256,len(rows));sub={k:x[start:stop] for k,x in tensors.items()}
        _, rr=compute_gae(sub['rewards'],v[start:stop],nv[start:stop],sub['terminated'],sub['active_mask'],
                         gamma=.99,gae_lambda=.95,truncated=sub['truncated'],episode_end=sub['episode_end'])
        for i in range(start,stop): records[i]['GAE_256_return']=rr[i-start].tolist()
    start=0
    for end in range(len(rows)):
        if all(batch['episode_end'][end]) or end == len(rows)-1:
            mc=full_returns(batch['rewards'][start:end+1],batch['terminated'][start:end+1])
            is_mc=bool(batch['terminated'][end].all())
            for i in range(start,end+1):
                records[i]['MC_return']=mc[i-start].tolist() if is_mc else None
                records[i]['observed_discounted_return']=mc[i-start].tolist()
            start=end+1
    for i,r in enumerate(records):
        r['GAE_return']=ret[i].tolist(); r['advantage']=adv[i].tolist()
        if i+1<len(records) and not any(r['episode_end']):
            assert r['after']['central_hash']==records[i+1]['before']['central_hash']
            assert r['after']['local_hash']==records[i+1]['before']['local_hash']
            assert r['after']['context']==records[i+1]['before']['context']
            np.testing.assert_array_equal(r['V_next'],records[i+1]['V_t'])
    atomic_json(out/'transitions.json', records)
    columns=['episode','t','phase_before','role_before','obs-context','action','reward','captured','terminated','truncated',
             'phase_after','next-context','V_t','V_next','MC return','GAE return','GAE 256 return']
    def compact(x): return json.dumps(x,separators=(',',':'))
    with (out/'transitions.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(columns)
        for r in records:
            b,n=r['before'],r['after']
            writer.writerow([b['episode'],b['t'],b['phase'],compact(b['roles']),compact(b['context']),compact(r['action']),compact(r['reward']),r['captured'],compact(r['terminated']),compact(r['truncated']),n['phase'],compact(n['context']),compact(r['V_t']),compact(r['V_next']),compact(r['MC_return']),compact(r['GAE_return']),compact(r['GAE_256_return'])])
    np.savez_compressed(out/'stored_rollout.npz', **{k:v for k,v in batch.items() if not isinstance(v,dict)},
                        **{'global_'+k:v for k,v in batch['global_obs'].items()}, **{'local_'+k:v for k,v in batch['local_obs'].items()})
    return dict(steps=len(rows), agent_rows=int(batch['active_mask'].sum()), independent_gae_max_error=float(np.max(abs(ret.numpy()-ref))),
                transition_sha256=sha256_file(out/'transitions.json'), rollout_sha256=sha256_file(out/'stored_rollout.npz'))


def native_replay(trainer, out):
    seed=2027099201
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    stream=AuditStream(seed,out)
    rows=[];records=[]
    while stream.episode < 4:
        row,r=traced(trainer,stream);rows.append(row);records.append(r)
        if r['completed']: print('completed',r['completed'],flush=True)
        assert len(rows)<2000
    result=annotate(rows,records,trainer,out)
    prior=dict(np.load(BASE/'heldout/fixed_bank.npz')); take=prior['episode']<4
    assert len(rows)==int(take.sum())
    for key in rows[0]['global_obs']:
        np.testing.assert_array_equal(np.stack([r['global_obs'][key] for r in rows]),prior['global_'+key][take])
    np.testing.assert_array_equal([('pre_capture','post_capture','pure_coverage').index(r['before']['phase']) for r in records],prior['phase'][take])
    expected=prior['target'][take];actual=np.asarray([r['MC_return'] for r in records],np.float32)
    result['original_bank_geometry_phase_bit_exact']=True
    result['original_bank_target_bit_exact']=bool(np.array_equal(expected,actual))
    result['original_bank_target_max_change']=float(abs(expected-actual).max())
    result['captures']=[dict(episode=r['before']['episode'],t=r['before']['t'],roles=r['before']['roles']) for r in records if r['captured']]
    result['episodes']=[r['completed'] for r in records if r['completed']]
    atomic_json(out/'summary.json',result)
    result['source_manifest_sha256']=sha256_file(BASE/'heldout/bank_manifest.json')
    result['source_bank_sha256']=sha256_file(BASE/'heldout/fixed_bank.npz')
    # Poison future rewards only: no propagation across either true terminal or truncation.
    ends=[i for i,r in enumerate(records) if all(r['episode_end'])]
    result['episode_end_indices']=ends
    return result


def controlled_probes(trainer,out):
    results={}; original_act=trainer.act
    def fixed_act(local,central,deterministic=False):
        with torch.no_grad():
            ids=np.full(len(local['self']),4);physical=trainer.actor.action_grid.detach().cpu().numpy()[ids]
            logp=trainer.actor.distribution(tensor_tree(local,trainer.device)).log_prob(torch.tensor(ids,device=trainer.device)).cpu().numpy()
            v=trainer.value(tensor_tree(central,trainer.device)).cpu().numpy()
        return physical,logp,ids,v
    trainer.act=fixed_act
    try:
        for name in ('timeout','continuing','success_at_timeout','post_window_at_timeout','support_capture','too_few_at_timeout','collision_at_timeout','capture_casualty_reset'):
            path=out/name;random.seed(19);np.random.seed(19);torch.manual_seed(19)
            s=AuditStream(19,path)
            if name not in ('support_capture','post_window_at_timeout','capture_casualty_reset'):s.reset()
            if name=='capture_casualty_reset':
                capture_fixture(s)
                p=s.env.pursuers[3];p.x=119.;p.y=20.;p.theta=0.;p.speed=3.;p.velocity=np.array([3.,0.])
                s.env._invalidate_voronoi_cache();s.recovery_from_capture_ratio=1.
            elif name=='support_capture':capture_fixture(s)
            elif name=='post_window_at_timeout':
                capture_fixture(s);s.env.step([4]*4,[4]);s.env.post_capture_started=True;s.env.post_capture_step=499;s.env.episode_step=2999
                s.observations=s.env.get_observations()
            else:
                s.env.episode_step=2998 if name=='continuing' else 2999
                if name=='success_at_timeout':converge_fixture(s)
                if name=='too_few_at_timeout':
                    s.env.pursuers[0].deactivated=True;s.env._invalidate_voronoi_cache();s.observations=s.env.get_observations()
                if name=='collision_at_timeout':
                    s.env.pursuers[0].x=s.env.pursuers[1].x;s.env.pursuers[0].y=s.env.pursuers[1].y
                    s.env._invalidate_voronoi_cache();s.observations=s.env.get_observations()
            # Repack after constructed state edits without advancing K10 memory.
            s.env.last_task_labels=s.env._task_labels_from_map(s.env._capture_voronoi_map(),update_effective=False)
            s.observations=s.env.get_observations()
            row,r=traced(trainer,s);rows=[row];records=[r]
            if name=='capture_casualty_reset':
                assert r['captured'] and all(r['terminated']) and s.task=='voradj_coverage'
                trainer.act=original_act
                for _ in range(21):
                    row2,r2=traced(trainer,s);rows.append(row2);records.append(r2)
                trainer.act=fixed_act
                assert records[1]['before']['active']==[True,True,True,False]
                assert all(records[1]['terminated'])
            if name=='timeout':
                # Trace the next episode too, exposing a poisoned/reset next-state bootstrap.
                for _ in range(20):
                    row2,r2=traced(trainer,s);rows.append(row2);records.append(r2)
            annotate(rows,records,trainer,path)
            results[name]=r
    finally:trainer.act=original_act
    a,b=results['timeout'],results['continuing']
    assert a['after']['central_hash']==b['after']['central_hash']
    results['timeout_reward_difference']=(np.array(a['reward'])-b['reward']).tolist()
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    trainer=make_trainer(a.device,2026097101)
    saved=torch.load(BASE/'critic_mc_100updates.pt',map_location=a.device,weights_only=False)
    trainer.value.load_state_dict(saved['trainer']['value'])
    trainer.value_normalizer.load_state_dict(saved['trainer']['value_normalizer'])
    trainer.actor.eval();trainer.value.eval()
    initial=(tensor_hash(trainer.actor.state_dict()),tensor_hash(trainer.value.state_dict()))
    norm=(float(trainer.value_normalizer.mean),float(trainer.value_normalizer.std))
    sources=[Path(__file__),ROOT/'src/cocap_voradj/envs/voronoi_adjacency.py',ROOT/'tools/train_forward_final_ppo_20260909.py',ROOT/'tools/run_continuous_ctde_training.py',ROOT/'src/cocap_voradj/training/small_step_ac.py',BC,BASE/'critic_mc_100updates.pt']
    launch=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                sources={str(x.relative_to(ROOT)):sha256_file(x) for x in sources},context_names=NAMES,device=a.device,
                actor_updates=0,critic_updates=0,normalization=norm,policy='frozen BC, seeded sample replay; controlled probes use AW9 index4',
                value='existing fitted geometry critic, frozen train-only ValueNorm',seed=2027099201)
    atomic_json(out/'launch.json',launch)
    native=native_replay(trainer,out/'native');probes=controlled_probes(trainer,out/'probes')
    assert initial==(tensor_hash(trainer.actor.state_dict()),tensor_hash(trainer.value.state_dict()))
    assert norm==(float(trainer.value_normalizer.mean),float(trainer.value_normalizer.std))
    atomic_json(out/'report.json',dict(native=native,probes=probes,parameters_and_normalizer_unchanged=True,
                                      mc_limit='MC_return=null at external truncation/open tail; observed_discounted_return is not a full MC target',
                                      gate='audit only; P3/PPO remain HOLD'))
    print(json.dumps(native,indent=2))


if __name__=='__main__':main()
