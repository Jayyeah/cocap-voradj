from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import pytest
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.envs.map_normalization import get_map_normalization_scale, physical_distance_threshold_factor, map_normalization_facts
from cocap_voradj.envs.coverage_ce import normalized_center_costs, ce_transition_reward, normalized_control_cost
from cocap_voradj.envs.z_guided_support import select_support_target, ally_progress
from cocap_voradj.training.trainer import set_global_config, CoCapTrainer
from tools import zsupport_relative_rnorm_20261017 as run
from tools import iqn_z_unified_decay_curriculum_20260919 as old
from tools import iqn_token_matched_20260919 as formal


def make_env(side=None,scene='mixed',stage='stage1',new=True,n=None):
    cfg=formal.resolved(run.CONFIGS[stage]) if new else old.resolved('z05',stage)
    cfg=formal.scene_config(cfg,scene)
    if side is not None:cfg['env'].update(width=side,height=side,x_boundary_left=0,x_boundary_right=side,y_boundary_bottom=0,y_boundary_top=side)
    if n is not None:cfg['env']['num_pursuers']=n
    set_global_config(cfg);env=VorAdjEnv(copy.deepcopy(cfg),seed=2026101801);obs=env.reset()
    return env,obs


def token(x,z):return np.asarray([x,0,0,0,abs(x),0,z],dtype=np.float32)

CASES=[(0,[.25,.25],'support',2),(.125,[.25,.5],'support',2),(.25,[.25,.25],'coverage',0),(.5,[.25,.125],'coverage',0),(0,[.125],'support',1),(0,[],'coverage',0),(0,[0,0],'coverage',0)]
@pytest.mark.parametrize('own,zs,role,target',CASES)
def test_relative_z_matrix_without_any_capture_neighbor(own,zs,role,target):
    env,_=make_env();env._support_policy_snapshots[0]=dict(self_z=own,friend_ids=tuple(range(1,len(zs)+1)),tokens=np.asarray([token(3-i,z) for i,z in enumerate(zs)],dtype=np.float32).reshape(-1,7))
    labels=['coverage']*len(env.pursuers)
    assert env._task_reward_role(0,labels,{'adjacency':{}})==role
    snap=env._support_policy_snapshots[0];s=select_support_target(0,own,snap['friend_ids'],snap['tokens'])
    assert s['target_id']==target
    assert env._task_reward_role(0,['capture']+labels[1:],{})=='capture'


def force_friend_map(env,monkeypatch,ids):
    data=copy.deepcopy(env._capture_voronoi_map());data['adjacency']={('pursuer',i):{('pursuer',j) for j in ids if j!=i} for i in range(len(env.pursuers))}
    monkeypatch.setattr(env,'_capture_voronoi_map',lambda *a,**k:data)
    env._pursuing_flags_initialized=False
    env._pursuing_release_counters=[0]*len(env.pursuers)
    for p in env.pursuers:p.is_pursuing=False
    return data


def test_hidden_neighbor_policy_cap_and_invalid_are_excluded(monkeypatch):
    env,_=make_env(n=11)
    for i,p in enumerate(env.pursuers):p.x=40+i;p.y=50;p.theta=0;p.deactivated=False
    data=force_friend_map(env,monkeypatch,range(1,11))
    env.z_state[:]=0;env.z_state[10]=.5
    env.get_observations();snap=env._support_policy_snapshots[0]
    assert 10 not in snap['friend_ids'] and env._task_reward_role(0,['coverage']*11,data)=='coverage'
    env.z_state[3]=.125;env.get_observations()
    assert env._task_reward_role(0,['coverage']*11,data)=='support'
    s=env._support_policy_snapshots[0];s['tokens'][:]=np.nan
    assert env._task_reward_role(0,['coverage']*11,data)=='coverage'


@pytest.mark.parametrize('zs',[[0,.5,.25,.125],[0,.25,.5,.125],[0,.125,.25,0]])
def test_inactive_target_residual_support_real_step_then_natural_decay(monkeypatch,zs):
    env,_=make_env();env.evaders[0].deactivated=True
    for i,p in enumerate(env.pursuers):p.x=30+15*i;p.y=50;p.theta=0;p.speed=0;p.deactivated=False
    force_friend_map(env,monkeypatch,range(4))
    env.z_state[:]=zs;env.get_observations()
    first=env.step([4]*4,[None]);m=first.infos[0]['replay_metadata']
    assert m['reward_role']=='support' and m['task_label']=='support' and m['support_reward_blend_active']
    assert m['support_reward_capture_weight']==max(zs)
    assert m['coverage_ce_applied_weight']==1-max(zs)
    assert m['z_support_selection']['target_id']==int(np.argmax(zs))
    assert np.isfinite(first.rewards).all()
    assert m['next_task_label']==first.infos[0]['next_task_label']
    # Recovery is real, so four .5-second decisions exist; no synthetic terminal transition.
    assert not all(first.dones)
    for _ in range(3):last=env.step([4]*4,[None])
    assert np.max(env.z_state)==0
    last=env.step([4]*4,[None])
    assert all(i['replay_metadata']['reward_role']=='coverage' for i in last.infos)


@pytest.mark.parametrize('side',[80.,120.,173.])
def test_r_config_not_inner_area_and_square_rejection(side):
    env,_=make_env(side=side)
    assert get_map_normalization_scale(env)==side
    assert env._distance_scale()==side
    facts=map_normalization_facts(env);assert facts['width']==side and facts['units']=='metres'
    env.height=side+1
    with pytest.raises(ValueError,match='square'):get_map_normalization_scale(env)


@pytest.mark.parametrize('side',[80.,120.,173.])
def test_normalized_distance_cost_and_physical_success_boundary(side):
    new,_=make_env(side=side);legacy,_=make_env(side=side,new=False)
    for distance in [side*.01,side*.025,side*.035,side*.06,.05*np.sqrt(2)*side/2*.999999,.05*np.sqrt(2)*side/2*1.000001]:
        oldd,oldc=legacy._ce_center_distance_and_cost(np.array([0,0]),np.array([distance,0]),4)
        newd,newc=new._ce_center_distance_and_cost(np.array([0,0]),np.array([distance,0]),4)
        assert newd==pytest.approx(np.sqrt(2)*oldd);assert newc==pytest.approx(2*oldc)
        def geometry(env):
            return env._ce_coverage_geometry(dict(centroids={('pursuer',i):env._position(p)+np.array([distance,0]) for i,p in enumerate(env.pursuers)},counts={('pursuer',i):100 for i in range(4)}),True)
        a,b=geometry(legacy),geometry(new)
        assert a['geometry_strict']==b['geometry_strict']
        assert a['ce_physical_rms_threshold']==pytest.approx(b['ce_physical_rms_threshold'])
        assert b['ce_center_physical_rms']==pytest.approx(distance)
        assert b['ce_rms_threshold']==pytest.approx(.05*np.sqrt(2))
    assert physical_distance_threshold_factor(new)==pytest.approx(np.sqrt(2))


@pytest.mark.parametrize('scene',['coverage','capture','mixed'])
def test_all_observation_map_columns_and_physics_unchanged(scene):
    a,oa=make_env(scene=scene,new=False);b,ob=make_env(scene=scene)
    for x,y in zip(oa,ob):
        if x is None:assert y is None;continue
        np.testing.assert_array_equal(x['masks'],y['masks'])
        for key,columns in [('self',[2,3,4,6,7]),('pursuers',[0,1,4]),('evaders',[0,1,4]),('obstacles',[0,1,2,3])]:
            expected=x[key].copy();expected[...,columns]*=np.sqrt(2)
            np.testing.assert_allclose(expected,y[key],rtol=3e-6,atol=1e-7)
    for x,y in zip(a.pursuers,b.pursuers):
        assert (x.r,x.max_speed,x.a,x.w,x.dt,x.N)==(y.r,y.max_speed,y.a,y.w,y.dt,y.N)
    assert a.env_cfg['capture_distance']==b.env_cfg['capture_distance']


@pytest.mark.parametrize('stage',['stage1','stage2','stage3'])
@pytest.mark.parametrize('scene',['coverage','capture','mixed'])
def test_train_and_independent_scene_same_r(stage,scene):
    env,_=make_env(stage=stage,scene=scene)
    assert env.width==env.height and get_map_normalization_scale(env)==env.width
    from tools.evaluate_iqn_z05_independent_20260923 import scene_config
    cfg=scene_config(formal.resolved(run.CONFIGS[stage]),scene)
    set_global_config(cfg);ind=VorAdjEnv(copy.deepcopy(cfg),seed=1)
    assert ind._distance_scale()==env._distance_scale()
    run.assert_scientific_config(cfg)


def test_ce_helper_pbrs_dynamic_mix_and_control_units():
    ds,cs=normalized_center_costs(np.array([[0.,0.]]),[np.array([6.,8.])],80.,4)
    assert ds[0]==.25 and cs[0]==.0625
    kw=dict(before_center_cost=.2,after_center_cost=.1,control_cost=.003,reward_scale=10,gamma=.99,pbrs_enabled=True,pbrs_kappa=1)
    reward,terms=ce_transition_reward(**kw)
    assert reward==pytest.approx(10*(-.1-.003-.99*.1+.2))
    env,_=make_env();alpha=.25;correction=(1-alpha)*env._ce_terminal_pbrs_correction(.1)
    terminal,_=ce_transition_reward(**kw,terminal=True)
    assert (1-alpha)*reward+correction==pytest.approx((1-alpha)*terminal)
    ally,_=ally_progress([0,0],[1,0],[2,0]);assert alpha*ally+(1-alpha)*reward==pytest.approx(.25+.75*reward)
    for phase in ['pre_capture','post_capture','pure_coverage']:
        trainer=object.__new__(CoCapTrainer);trainer.replays={k:None for k in ['pursuing','pre_capture_cover','post_capture_real','recovery_pure']}
        assert trainer._voradj_replay_class({'task_label':'support','phase':phase})=={'pre_capture':'pre_capture_cover','post_capture':'post_capture_real','pure_coverage':'recovery_pure'}[phase]


def test_new_defaults_do_not_change_legacy_z05_config():
    cfg=old.resolved('z05','stage1');assert 'normalization' not in cfg
    env,_=make_env(new=False);assert not env._relative_z_support_enabled
    assert env._distance_scale()==pytest.approx(np.hypot(env.width,env.height))
