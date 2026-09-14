"""Boundary regressions use the native Final step, not synthetic done labels."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from cocap_voradj.training.forward_final import make_env
from cocap_voradj.training.small_step_ac import compute_gae
from tools.run_continuous_ctde_training import _split_termination_flags
from tools.audit_forward_final_transitions_20260914 import capture_fixture, converge_fixture


def stream(env):
    return SimpleNamespace(env=env, observations=env.get_observations())


def test_external_timeout_preserves_continuing_pbrs_reward():
    env,_=make_env('coverage',19)
    env.episode_step=2999
    continuing=copy.deepcopy(env);continuing.episode_step=2998
    a=env.step([4]*4,[]);b=continuing.step([4]*4,[])
    term,trunc=_split_termination_flags(a.dones,a.infos)
    assert trunc.all() and not term.any() and not any(b.dones)
    np.testing.assert_allclose(a.rewards,b.rewards,atol=1e-12)
    assert all(i['replay_metadata']['reward_ce_terminal_correction']==0 for i in a.infos)


@pytest.mark.parametrize('cause',['success','post_window','too_few','collision'])
def test_true_terminal_wins_when_time_limit_coincides(cause):
    env,_=make_env('mixed' if cause=='post_window' else 'coverage',19)
    if cause=='success':converge_fixture(stream(env))
    elif cause=='post_window':
        capture_fixture(stream(env));env.step([4]*4,[4])
        env.post_capture_started=True;env.post_capture_step=499
    elif cause=='too_few':env.pursuers[0].deactivated=True
    else:
        env.pursuers[0].x=env.pursuers[1].x;env.pursuers[0].y=env.pursuers[1].y
    env.episode_step=2999;env._invalidate_voronoi_cache()
    outcome=env.step([4]*4,[None]*len(env.evaders))
    term,trunc=_split_termination_flags(outcome.dones,outcome.infos)
    assert term.all() and not trunc.any()
    rewards=torch.tensor(np.asarray(outcome.rewards)[None],dtype=torch.float32)
    _,returns=compute_gae(rewards,torch.ones(1,4)*7,torch.ones(1,4)*99,
                          torch.tensor(term[None]),torch.ones(1,4,dtype=torch.bool),gamma=.99,gae_lambda=.95,
                          truncated=torch.tensor(trunc[None]))
    torch.testing.assert_close(returns,rewards)


def test_support_capture_correction_uses_same_half_weight_as_ce():
    env,_=make_env('mixed',19);capture_fixture(stream(env))
    cost=env._voradj_coverage_potentials(
        np.asarray([[p.x,p.y] for p in env.pursuers]),np.asarray([[e.x,e.y] for e in env.evaders]),data=env._coverage_voronoi_map())
    result=env.step([4]*4,[4]);meta=result.infos[3]['replay_metadata']
    assert env.last_capture_events and not any(result.dones)
    assert meta['reward_role']=='support' and meta['coverage_ce_pbrs_reset_reason']=='capture_phase_boundary'
    expected=.5*env._ce_terminal_pbrs_correction(float(cost[3]))
    np.testing.assert_allclose(meta['reward_ce_terminal_correction'],expected,atol=1e-12)
    np.testing.assert_allclose(meta['reward_coverage'],0.,atol=1e-12)
    np.testing.assert_allclose(meta['reward_support_blend_coverage'],meta['reward_coverage'],atol=1e-12)
    assert all(o['self'][-1]==0 for o in result.observations)
    after=env.step([4]*4,[None])
    assert all(i['replay_metadata']['reward_terminal']==0 for i in after.infos)
    assert all(i['replay_metadata']['coverage_ce_pbrs_reset_reason']=='none' for i in after.infos)


def test_gae_does_not_cross_reset_or_agent_axis_with_nonunit_value_scale():
    # Actual storage convention: normalized values -> raw before GAE.
    v_norm=torch.tensor([[1.,3.],[2.,4.],[8.,6.]])
    nv_norm=torch.tensor([[2.,4.],[7.,5.],[9.,7.]])
    v=v_norm*13+17;nv=nv_norm*13+17
    r=torch.tensor([[1.,-10.],[2.,-20.],[3.,-30.]])
    active=torch.tensor([[True,True],[True,True],[True,False]])
    term=torch.tensor([[False,False],[False,True],[True,True]])
    trunc=torch.tensor([[False,False],[True,False],[False,False]])
    def run(reward):return compute_gae(reward,v,nv,term,active,gamma=.99,gae_lambda=.95,truncated=trunc)[1]
    a=run(r);poison=r.clone();poison[2]=1e6;b=run(poison)
    torch.testing.assert_close(a[:2],b[:2])
    torch.testing.assert_close(a[1,0],r[1,0]+.99*nv[1,0])
    torch.testing.assert_close(a[1,1],r[1,1])
    assert a[2,1]==0
    # Poison only agent1, agent0 must remain identical.
    poison=r.clone();poison[:,1]=1e6
    torch.testing.assert_close(a[:,0],run(poison)[:,0])


def test_reset_clears_context_when_coverage_environment_is_reused(tmp_path):
    from tools.train_forward_final_ppo_20260909 import FinalMissionStream,local_and_global
    s=FinalMissionStream(19,tmp_path);s.reset();env=s.env
    env.distribution_hold_steps=29;env.post_capture_started=True;env.post_capture_step=400
    env._stationary_capture_counters={'evader_0':9};env._pursuing_release_counters=[9]*4
    for p in env.pursuers:p.is_pursuing=True
    s.reset();s.reset()
    assert s.env is env and env.episode_step==0 and env.distribution_hold_steps==0
    assert env.post_capture_step==0 and not env.post_capture_started and not env.post_capture_coverage_success
    assert env._stationary_capture_counters=={} and env._pursuing_release_counters==[0]*4
    local,central=local_and_global(s)
    np.testing.assert_array_equal(local['self'],central['self'])
    assert not local['self'][:,-1].any()


def test_native_capture_casualty_snapshot_reset_samples_only_active_agents(tmp_path):
    import random
    from tools.train_forward_final_ppo_20260909 import FinalMissionStream,make_trainer,collect_transition,empty_rollout,stack_rollout
    random.seed(19);np.random.seed(19)
    s=FinalMissionStream(19,tmp_path);capture_fixture(s)
    p=s.env.pursuers[3];p.x=119.;p.y=20.;p.theta=0.;p.speed=3.;p.velocity=np.array([3.,0.])
    s.env._invalidate_voronoi_cache();s.observations=s.env.get_observations()
    s.recovery_from_capture_ratio=1.
    result=s.step([4]*4)
    assert s.env.last_capture_events and all(result.dones)
    assert s.env.capture_snapshot['active_mask']==[True,True,True,False]
    s.finish()
    assert s.last_reset_source['voradj_coverage']=='capture_snapshot'
    assert [o is not None for o in s.observations]==[True,True,True,False]
    trainer=make_trainer('cpu',19)
    row,done=collect_transition(trainer,s)
    assert done is not None  # Native min_active=4 still terminates this scene.
    np.testing.assert_array_equal(row['active_mask'],[True,True,True,False])
    for key in ('rewards','values','next_values','log_prob','actions'):assert np.isfinite(row[key]).all()
    assert row['latent'][3]==4 and row['log_prob'][3]==0
    rollout=empty_rollout()
    for k,v in row.items():rollout[k].append(v)
    assert trainer.assert_behavior_log_probs(stack_rollout(rollout))<1e-4
    _,ret=compute_gae(torch.tensor(row['rewards'][None]),torch.tensor(row['values'][None]),
                      torch.tensor(row['next_values'][None]),torch.tensor(row['terminated'][None]),
                      torch.tensor(row['active_mask'][None]),gamma=.99,gae_lambda=.95,truncated=torch.tensor(row['truncated'][None]))
    assert ret[0,3]==0
