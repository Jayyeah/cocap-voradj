import copy
from types import SimpleNamespace
import numpy as np
import torch
from tools import forward_final_single_task_20260915 as s
from tools.audit_forward_final_transitions_20260914 import capture_fixture, converge_fixture
from cocap_voradj.training.forward_final import scene_config
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from tools.run_continuous_ctde_training import _split_termination_flags


def test_parity_and_exact_common_random_init():
    report=s.parity();assert report['UNEXPLAINED']==0
    c=s.contract()['scratch']
    a=s.p.make_trainer(c,19,'cpu');b=s.p.make_trainer(c,19,'cpu')
    s.p.assert_tree_equal(a.state_dict(),b.state_dict())
    assert a.actor_optimizer is not b.actor_optimizer
    assert a.value_norm is not b.value_norm


def test_current_crms_support_renormalization_and_capture_terminal():
    env,_=s.make_env('capture',19)
    capture_fixture(SimpleNamespace(env=env,observations=env.get_observations()))
    # Moving actions make support's nonzero approach term observable.
    full=copy.deepcopy(env)
    full.config['voradj'].pop('single_task_objective')
    full.config['voradj']['capture_episode_ends_on_capture']=False
    full.config['voradj']['support_reward_capture_weight']=.5
    full.config['voradj']['support_reward_coverage_weight']=.5
    a=env.step([7]*4,[4]);b=full.step([7]*4,[4])
    assert env.last_capture_events and all(a.dones) and not any(b.dones)
    term,trunc=_split_termination_flags(a.dones,a.infos)
    assert term.all() and not trunc.any()
    for x,y in zip(a.infos,b.infos):
        x=x['replay_metadata'];y=y['replay_metadata']
        assert x['reward_coverage']==x['reward_ce_center']==x['reward_ce_pbrs']==0
        if x['support_candidate']:
            assert abs(y['reward_capture'])>1e-8
            np.testing.assert_allclose(x['reward_capture'],2*y['reward_capture'])
        elif x['reward_role']=='capture':
            np.testing.assert_allclose(x['reward_capture'],y['reward_capture'])
        np.testing.assert_allclose(x['reward_safety'],y['reward_safety'])
        np.testing.assert_allclose(x['reward_terminal'],y['reward_terminal'])


def test_coverage_exact_final_reward_on_identical_state_actions():
    env,_=s.make_env('coverage',19)
    full=copy.deepcopy(env)
    full.config['voradj'].pop('single_task_objective')
    full.config['voradj']['support_reward_blend_enabled']=True
    full.config['voradj']['support_reward_capture_weight']=.5
    full.config['voradj']['support_reward_coverage_weight']=.5
    for actions in ([0,4,8,7],[7,8,4,0],[4]*4):
        a=env.step(actions,[]);b=full.step(actions,[])
        np.testing.assert_array_equal(a.rewards,b.rewards)
        for x,y in zip(a.infos,b.infos):
            for k in s.REWARD_KEYS:assert x['replay_metadata'][k]==y['replay_metadata'][k]
        assert a.dones==b.dones


def test_single_task_timeout_bootstrap_and_success_priority():
    env,_=s.make_env('coverage',19);env.episode_step=2999
    continuing=copy.deepcopy(env);continuing.episode_step=2998
    a=env.step([4]*4,[]);b=continuing.step([4]*4,[])
    np.testing.assert_array_equal(a.rewards,b.rewards)
    term,trunc=_split_termination_flags(a.dones,a.infos)
    assert trunc.all() and not term.any()
    env,_=s.make_env('coverage',19)
    converge_fixture(SimpleNamespace(env=env,observations=env.get_observations()))
    env.episode_step=2999;a=env.step([4]*4,[])
    term,trunc=_split_termination_flags(a.dones,a.infos)
    assert term.all() and not trunc.any() and env.post_capture_coverage_success


def test_r2_changes_only_center_and_pbrs_including_boundary():
    env,_=s.make_env('coverage',19)
    converge_fixture(SimpleNamespace(env=env,observations=env.get_observations()))
    a=env.step([4]*4,[]);b=copy.deepcopy(a);s.shape_reward(b,2.)
    for i,(x,y) in enumerate(zip(a.infos,b.infos)):
        x=x['replay_metadata'];y=y['replay_metadata']
        assert y['reward_ce_control']==x['reward_ce_control']
        assert y['reward_safety']==x['reward_safety']
        assert y['reward_terminal']==x['reward_terminal']
        np.testing.assert_allclose(b.rewards[i]-a.rewards[i],x['reward_ce_center']+x['reward_ce_pbrs'])
        assert y['reward_ce_terminal_correction']==2*x['reward_ce_terminal_correction']


def test_stream_has_no_recovery_and_retains_collector_masks():
    stream=s.SingleTaskStream('capture',19)
    capture_fixture(stream);outcome=stream.step([4]*4)
    assert all(outcome.dones)
    row=stream.finish()
    assert row['captured'] and row['pool_size']==0
    assert stream.task_name=='capture' and len(stream.env.evaders)==1
    assert not hasattr(stream,'recovery_init_pool')
    assert stream.telemetry.count['length']==0


def fake_reports(task,success=0.,rms=.2,cv=.3,hold=0.,collision=.1):
    import copy
    budget=500000 if task=='capture' else 200000
    out=[]
    for step in range(0,budget+1,25000):
        row=dict(captured_rate=success if task=='capture' and step else 0.,
                 ce_success_rate=success if task=='coverage' and step else 0.,collision_rate=collision,
                 ce_rms=dict(p50=rms if step else .2),area_cv=dict(p50=cv if step else .3),
                 strict_max_hold=dict(mean=hold if step else 0.))
        out.append(dict(step=step,summary={m:copy.deepcopy(row) for m in ('argmax','sample')}))
    return out


def test_capture_never_classifies_early_or_best_spike_strong():
    from tools import analyze_forward_final_single_task_20260915 as a
    rs=fake_reports('capture',success=.6)
    assert a.capture_result(rs[:11])['decision'] is None
    assert a.capture_result(rs)['decision']=='STRONG_STABLE_LEARNABILITY'
    rs[-1]['summary']['argmax']['captured_rate']=0.
    assert a.capture_result(rs)['decision']=='WEAK_OR_UNSTABLE_LEARNABILITY'
    rs=fake_reports('capture');rs[10]['summary']['sample']['captured_rate']=.9
    assert a.capture_result(rs)['decision']=='WEAK_OR_UNSTABLE_LEARNABILITY'
    assert a.capture_result(fake_reports('capture'))['decision']=='NO_MEANINGFUL_LEARNABILITY'


def test_coverage_gate_requires_budget_and_uses_sustained_geometry():
    from tools import analyze_forward_final_single_task_20260915 as a
    rs=fake_reports('coverage',rms=.1,cv=.1,hold=4.)
    assert a.coverage_result(rs[:5])['decision'] is None
    assert a.coverage_result(rs)['decision']=='PURE_COVERAGE_LEARNABLE'
    assert a.coverage_result(fake_reports('coverage'))['decision']=='NO_MEANINGFUL_LEARNING'
    rs=fake_reports('coverage');rs[3]['summary']['sample']['ce_success_rate']=1.
    assert a.coverage_result(rs)['decision']=='NO_MEANINGFUL_LEARNING'
    for r in rs[2:5]:r['summary']['sample']['ce_success_rate']=.2
    result=a.coverage_result(rs)
    assert result['decision']=='PURE_COVERAGE_LEARNABLE' and result['later_regression']


def test_supervised_random_backbone_receives_action_gradient():
    import torch.nn.functional as F
    from tools.probe_forward_final_coverage_20260915 import agreement
    actor=s.p.make_actor(s.contract()['scratch'],19,'cpu')
    env,obs=s.make_env('coverage',19)
    local={k:np.stack([o[k] for o in obs]) for k in obs[0]}
    arrays={'local.'+k:v for k,v in local.items()};arrays['action']=np.arange(4)
    score=agreement(actor,arrays,np.arange(4));assert score['rows']==4
    before=s.p.tensor_hash(actor.encoder.state_dict())
    optimizer=torch.optim.Adam(actor.parameters(),lr=.0003)
    loss=F.cross_entropy(actor.distribution(s.tensor_tree(local,torch.device('cpu'))).logits,torch.arange(4))
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in actor.encoder.parameters())
    optimizer.step()
    assert s.p.tensor_hash(actor.encoder.state_dict())!=before
