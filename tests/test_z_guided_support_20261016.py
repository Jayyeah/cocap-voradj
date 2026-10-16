from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.envs.z_guided_support import select_support_target, ally_progress
from cocap_voradj.training.trainer import set_global_config
from tools import iqn_z_unified_decay_curriculum_20260919 as source
from tools import iqn_token_matched_20260919 as matched

BASE = Path('/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919/src/cocap_voradj/envs/voronoi_adjacency.py')

def env_config(enabled=False, scene='mixed', n=None):
    cfg = matched.scene_config(source.resolved('z05', 'stage1'), scene)
    cfg['voradj']['z_guided_support_enabled'] = enabled
    if n is not None:
        cfg['env']['num_pursuers'] = n
    return cfg

def make_env(enabled=False, scene='mixed', n=None, cls=VorAdjEnv, seed=2026101601):
    cfg = env_config(enabled, scene, n)
    set_global_config(cfg)
    env = cls(copy.deepcopy(cfg), seed=seed)
    obs = env.reset()
    return env, obs

def token(x, y, z, vx=0., vy=0.):
    return np.asarray([x, y, vx, vy, np.hypot(x,y), np.arctan2(y,x), z], dtype=np.float32)

CASES = [
    ('zero', 0., [], [], 0, 0.),
    ('self-max', .5, [1], [token(2,0,.25)], 0, 0.),
    ('self-equal', .5, [1], [token(2,0,.5)], 0, 0.),
    ('self-near-equal', .5, [1], [token(2,0,.5000005)], 0, 0.),
    ('unique-high', .25, [1], [token(2,0,1)], 1, 1.),
    ('far-high-vs-near-low', 0., [1,2], [token(1,0,.5),token(8,0,1)], 2, 1.),
    ('highest-nearest', 0., [1,2,3], [token(9,0,1),token(2,0,1),token(1,0,.5)], 2, 1.),
    ('geometry-opposite', 0., [1,2], [token(2,0,.5),token(-2,0,.5)], 2, .5),
    ('geometry-velocity', 0., [1,2], [token(2,0,.5,1),token(2,0,.5,-1)], 2, .5),
    ('stable-id-fallback', 0., [3,2], [token(2,0,.5),token(2,0,.5)], 2, .5),
    ('not-slot-order', 0., [2,1], [token(-8,0,.5),token(2,0,.5)], 1, .5),
    ('invalid-friend', 0., [1,2], [token(float('nan'),0,1),token(2,0,.5)], 2, .5),
]
@pytest.mark.parametrize('name,z,ids,tokens,expected,alpha', CASES, ids=[c[0] for c in CASES])
def test_selection(name,z,ids,tokens,expected,alpha):
    actual = select_support_target(0,z,ids,tokens)
    assert actual['target_id'] == expected
    assert actual['alpha'] == alpha
    reverse = select_support_target(0,z,list(reversed(ids)),list(reversed(tokens)))
    assert reverse['target_id'] == expected
    if name == 'stable-id-fallback':
        assert actual['id_fallback']
    if name == 'geometry-opposite':
        assert actual['distance_tie'] and not actual['id_fallback']
        assert actual['target_id'] != 0  # no averaged anchor at the origin


def equal_tree(left,right):
    if isinstance(left,dict):
        assert set(left) <= set(right)
        for key,value in left.items(): equal_tree(value,right[key])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right): equal_tree(a,b)
    elif isinstance(left,np.ndarray):
        np.testing.assert_array_equal(left,right)
    elif isinstance(left,float) and np.isnan(left):
        assert np.isnan(right)
    else: assert left==right

@pytest.mark.parametrize('scene',['coverage','capture','mixed'])
@pytest.mark.parametrize('seed',[2026101601,2026101602])
def test_t0_historical_fixed_transitions(scene,seed):
    spec=importlib.util.spec_from_file_location('cocap_voradj.envs.historical_z05',BASE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old,oo=make_env(scene=scene,cls=module.VorAdjEnv,seed=seed)
    new,no=make_env(scene=scene,seed=seed)
    equal_tree(oo,no)
    rng=np.random.default_rng(seed)
    for step in range(80):
        actions=rng.integers(0,9,len(old.pursuers)).tolist()
        ea=rng.integers(0,9,len(old.evaders)).tolist()
        a=old.step(actions,ea);b=new.step(actions,ea)
        equal_tree(a.observations,b.observations)
        equal_tree(a.rewards,b.rewards);equal_tree(a.dones,b.dones)
        equal_tree(a.infos,b.infos)
        equal_tree(old.z_state_dict(),new.z_state_dict())
        equal_tree(old.last_capture_events,new.last_capture_events)
        equal_tree(old.last_voradj_metrics,new.last_voradj_metrics)
        if all(a.dones): break


def test_token_mapping_sort_cap_inactive_and_changed_adjacency(monkeypatch):
    env,_=make_env(True,n=11)
    for i,p in enumerate(env.pursuers):
        p.x,p.y,p.theta=60.+i,60.,0.
        p.deactivated=False
    env.pursuers[0].x=60.
    for i in range(1,10): env.pursuers[i].x=45.+i  # physical order != distance
    env.pursuers[10].deactivated=True
    env.z_state[:]=0;env.z_state[9]=1.;env.z_state[7]=.5
    data=env._capture_voronoi_map()
    adjacency={('pursuer',0):{('pursuer',i) for i in range(1,11)}}
    data=copy.deepcopy(data);data['adjacency']=adjacency
    monkeypatch.setattr(env,'_capture_voronoi_map',lambda *args:data)
    obs=env._pack_agent_obs(0)
    snap=env._support_policy_snapshots[0]
    assert snap['friend_ids']==tuple(range(1,9))
    np.testing.assert_array_equal(snap['tokens'],obs['pursuers'][obs['masks'][1:9]])
    choice=select_support_target(0,snap['self_z'],snap['friend_ids'],snap['tokens'])
    assert choice['target_id']==7  # ID9 with higher Z was truncated
    assert 10 not in choice['candidate_ids']
    adjacency[('pursuer',0)]={('pursuer',9)}
    env._pack_agent_obs(0);snap=env._support_policy_snapshots[0]
    assert select_support_target(0,snap['self_z'],snap['friend_ids'],snap['tokens'])['target_id']==9


def test_synchronous_two_three_hop_decay_and_multiple_enemies(monkeypatch):
    env,_=make_env(True)
    data={'adjacency':{('pursuer',i):{('pursuer',j) for j in range(4) if abs(i-j)==1} for i in range(4)}}
    direct={0}
    monkeypatch.setattr(env,'_has_enemy_neighbor',lambda _data,key:key[1] in direct)
    env._reset_z_state()
    for expected in ([1,0,0,0],[1,.5,0,0],[1,.5,.25,0],[1,.5,.25,.125]):
        env._advance_z_state(data);np.testing.assert_array_equal(env.z_state,expected)
    assert env._z_lineage_hops.tolist()==[0,1,2,3]
    direct.clear()
    for peak in [.5,.25,.125,0]:
        env._advance_z_state(data);assert max(env.z_state)==peak
    # Two distinct sources; capture/remove one source retains other evidence.
    direct.update({0,3});env._advance_z_state(data)
    direct.remove(0)
    env._advance_z_state(data)
    assert env.z_state[0]==.5 and env.z_state[3]==1
    for _ in range(4): env._advance_z_state(data)
    np.testing.assert_array_equal(env.z_state,[.125,.25,.5,1])


def test_ally_anchor_frozen_and_own_motion():
    reward,progress=ally_progress([0,0],[1,0],[3,0])
    assert reward==progress==1
    # Friend's post-action position [100,0] is deliberately not an argument.
    assert ally_progress([0,0],[0,0],[3,0])[0]==0
    assert ally_progress([0,0],[100,0],[3,0])[0]==-3
    with pytest.raises(ValueError): ally_progress([0,0],[1,0],[np.nan,0])


def test_support_self_tie_full_ce_and_gating(monkeypatch):
    control,_=make_env(False,scene='coverage')
    treatment,_=make_env(True,scene='coverage')
    # Exercise original support gate while keeping a visible active target.
    for env in (control,treatment):
        env.z_state[:]=.5;env.get_observations()
        monkeypatch.setattr(env,'_task_reward_role',lambda i,*args:'support')
        # Coverage env has no targets, so current post-capture gate must win.
    a=control.step([4]*4,[]);b=treatment.step([4]*4,[])
    np.testing.assert_array_equal(a.rewards,b.rewards)
    for info in b.infos:
        meta=info['replay_metadata'];assert not meta['support_reward_blend_active']
        assert meta['z_support_selection']['alpha']==0


def test_dynamic_ce_scales_ledger_and_terminal_reset(monkeypatch):
    env,_=make_env(True)
    # Fix roles to reach every alpha while preserving real CE and dynamics.
    monkeypatch.setattr(env,'_task_reward_role',lambda i,*args:'support')
    env.z_state[:]=[.25,1,.5,.125];env.get_observations()
    result=env.step([4]*4,[4])
    for i,info in enumerate(result.infos):
        m=info['replay_metadata']
        assert np.isfinite(result.rewards[i])
        if m['support_reward_blend_active']:
            assert m['support_reward_coverage_weight']==1-m['z_support_selection']['alpha']
            assert m['reward_coverage']==pytest.approx(m['reward_ce_center']+m['reward_ce_control']+m['reward_ce_pbrs'])
            assert m['reward_capture']==pytest.approx(m['reward_support_blend_capture'])
        assert result.rewards[i]==pytest.approx(sum(m[k] for k in ['reward_coverage','reward_capture','reward_safety','reward_terminal','reward_speed_repeat','reward_early_speed','reward_deceleration','reward_settled_terminal','reward_motion_penalty']))


def test_new_selector_ignores_hidden_enemy_positions():
    a=select_support_target(0,.25,[1,2],[token(5,0,.5),token(-2,0,.5)])
    # API has no enemy position, target ID, role or collection order input.
    assert a['target_id']==2
    assert a['alpha']==.5

@pytest.mark.parametrize('terminal',[False,True])
def test_real_capture_boundary_and_weighted_reset(monkeypatch,terminal):
    env,_=make_env(True,scene='capture' if terminal else 'mixed')
    for p,pos in zip(env.pursuers,[(67,60),(56.5,66.0621778),(56.5,53.9378222),(95,95)]):
        p.x,p.y=pos;p.speed=0;p.theta=0;p.deactivated=False
    env.evaders[0].x=60;env.evaders[0].y=60;env.evaders[0].speed=0
    env._invalidate_voronoi_cache()
    monkeypatch.setattr(env,'_task_reward_role',lambda i,*args:'support')
    env.z_state[:]=[.25,1,.5,.125];env.get_observations()
    result=env.step([4]*4,[4])
    assert env.last_capture_events and env.evaders[0].deactivated
    assert env.last_capture_events[0]['capture_type']=='loose'
    assert max(env.z_state)==.5  # capture does not clear natural memory
    for info in result.infos:
        m=info['replay_metadata']
        w=m['support_reward_coverage_weight']
        assert m['coverage_ce_pbrs_reset_reason']==('capture_phase_boundary' if w else 'none')
        # Cancellation uses exactly this transition's CE weight, once only.
        assert m['reward_ce_terminal_correction']==pytest.approx(-.99*m['reward_ce_center'])
        assert m['reward_ce_terminal_correction']>=0
        assert m['reward_coverage']==pytest.approx(m['reward_ce_center']+m['reward_ce_control']+m['reward_ce_pbrs'])
    m=result.infos[1]['replay_metadata']
    assert m['z_support_selection']['target_id']==1 and m['support_reward_coverage_weight']==1
    if terminal:
        assert all(result.dones)
    else:
        next_result=env.step([4]*4,[4])
        assert all(not i['replay_metadata']['support_reward_blend_active'] for i in next_result.infos)
        assert max(env.z_state)==.25


def test_inactive_nan_friend_never_packed(monkeypatch):
    env,_=make_env(True)
    env.pursuers[1].deactivated=True
    env.pursuers[1].x=float('nan');env.z_state[1]=1
    data=env._capture_voronoi_map()
    data['adjacency'][('pursuer',0)]={('pursuer',1),('pursuer',2)}
    monkeypatch.setattr(env,'_capture_voronoi_map',lambda *args:data)
    obs=env._pack_agent_obs(0);snap=env._support_policy_snapshots[0]
    assert snap['friend_ids']==(2,)
    assert np.isfinite(obs['pursuers']).all()


def test_two_actual_enemy_sources_capture_one_preserves_other():
    cfg=env_config(True);cfg['env']['num_evaders']=2;set_global_config(cfg)
    env=VorAdjEnv(cfg,seed=2026101603);env.reset()
    for p,pos in zip(env.pursuers,[(15,15),(20,30),(100,90),(105,105)]):
        p.x,p.y=pos;p.speed=0;p.theta=0
    for e,pos in zip(env.evaders,[(12,12),(108,108)]):e.x,e.y=pos;e.speed=0
    env._invalidate_voronoi_cache();env._reset_z_state()
    env._advance_z_state(env._capture_voronoi_map())
    assert env._z_last_direct.tolist()==[True,True,True,True]
    env.evaders[0].deactivated=True;env._invalidate_voronoi_cache()
    for _ in range(4):env._advance_z_state(env._capture_voronoi_map())
    assert not env._z_last_direct[0] and env._z_last_direct[3]
    assert env.z_state[3]==1 and max(env.z_state)==1
    assert env.z_state[0]<1


def test_matched_configs_differ_only_reward_switch_and_operational_name():
    from tools.z_guided_support_20261016 import config
    a=config('control');b=config('treatment')
    diffs=historical_diff=source.diff_rows(source.flattened(a),source.flattened(b))
    assert {r['path'] for r in diffs}=={'run_name','voradj.z_guided_support_enabled'}
    for cfg in (a,b):source._assert_production_contract(cfg,.5)


def test_paired_gate_rejects_capture_and_collision_regressions():
    from tools.z_guided_support_20261016 import compare,MODES,SCENES
    def report(capture,collision):
        return {'summary':{m:{s:{'episodes':50,'normal_capture':{'rate':capture},'collision':{'rate':collision},'support':{'effective_attraction':100}} for s in SCENES} for m in MODES}}
    assert compare(report(.9,.1),report(.9,.1))['status']=='ENGINEERING_PASS_NOT_NONINFERIORITY'
    assert compare(report(.9,.1),report(.7,.1))['status']=='PAUSE_DIAGNOSE'
    assert compare(report(.9,.1),report(.9,.3))['status']=='PAUSE_DIAGNOSE'
