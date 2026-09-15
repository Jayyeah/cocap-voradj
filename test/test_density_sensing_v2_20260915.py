import copy
from types import SimpleNamespace
import numpy as np
import pytest
from tools import forward_final_single_task_20260915 as st
from tools.audit_forward_final_transitions_20260914 import capture_fixture,converge_fixture
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.envs.density_sensing import (enable_v2,runtime_metadata,map_scale,
    resolve_density_normalized_onboard_radius,CALIBRATED_K)


def make(task='capture',n=4,alpha=1.):
    cfg=enable_v2(st.task_config(task));cfg['env']['num_pursuers']=n;cfg['env']['num_obstacles']=n//4
    if alpha!=1:cfg['reward']['static_capture_scale']=alpha
    st.p.set_global_config(cfg);env=VorAdjEnv(cfg,seed=19);env.reset();return env


@pytest.mark.parametrize('n',[4,8,12])
def test_single_resolver_and_surface_semantics(n):
    env=make(n=n);m=runtime_metadata(env);r=m['resolved_onboard_radius']
    assert env._vct_ls_sensing_radius('enemy')==env._vct_ls_sensing_radius('obstacle')==r
    assert m['map_size']==[120.,120.] and m['normalized_radius']==pytest.approx(CALIBRATED_K)
    observer=env.pursuers[0];enemy=env.evaders[0];observer.x=10.;observer.y=50.
    # Outside center radius but inside surface radius: a center-based consumer fails.
    enemy.x=10+r+observer.r+enemy.r-.01;enemy.y=50.
    assert env._vct_ls_direct_enemy_ids_for_pursuer(0)==[0]
    enemy.x+=.02;assert env._vct_ls_direct_enemy_ids_for_pursuer(0)==[]
    obstacle=env.obstacles[0];obstacle.x=10+r+observer.r+obstacle.r-.1;obstacle.y=50.
    visible,clearance=env._vct_ls_obstacle_visible_for_pursuer(0,0)
    assert visible and clearance>r-observer.r-obstacle.r
    obstacle.x+=1.;assert not env._vct_ls_obstacle_visible_for_pursuer(0,0)[0]
    assert all(k[0]=='pursuer' for k in env._capture_voronoi_map()['keys'])


@pytest.mark.parametrize('field,value',[('enemy_sensing_radius',20),('obstacle_sensing_radius',20),('local_sensing_uses_surface_distance',False)])
def test_reject_drifting_fields(field,value):
    env=make();env.config['voradj'][field]=value
    with pytest.raises(ValueError):runtime_metadata(env)


def test_reject_global_enemy_broadcast_and_radius_floor():
    env=make();env.per_cfg['global_evader_visibility']=True
    with pytest.raises(ValueError):runtime_metadata(env)
    area=14364.;n=64;scale=map_scale(n,area)
    assert scale>1 and map_scale(12,area)==1
    with pytest.raises(ValueError):resolve_density_normalized_onboard_radius(n,120,120,area)
    m=resolve_density_normalized_onboard_radius(n,120*scale,120*scale,area*scale**2)
    assert m['resolved_onboard_radius']==pytest.approx(20.)


def test_scale_direct_support_terminal_but_preserve_coverage_safety():
    a=make();b=make(alpha=.5)
    for env in (a,b):
        # Full-Mix support weights, same state and transition.
        env.config['voradj'].pop('single_task_objective')
        env.config['voradj']['support_reward_capture_weight']=.5
        env.config['voradj']['support_reward_coverage_weight']=.5
        capture_fixture(SimpleNamespace(env=env,observations=env.get_observations()))
        env._pursuing_flags_initialized=False
        env._pursuing_release_counters=[0]*4
        env.last_task_labels=env._task_labels_from_map(env._capture_voronoi_map(),update_effective=True)
    x=a.step([7]*4,[4]);y=b.step([7]*4,[4]);assert a.last_capture_events and b.last_capture_events
    assert any(i['replay_metadata']['support_candidate'] for i in x.infos)
    for i,(xx,yy) in enumerate(zip(x.infos,y.infos)):
        xx=xx['replay_metadata'];yy=yy['replay_metadata']
        for key in ('reward_capture','reward_support_blend_capture','reward_terminal'):
            assert yy[key]==pytest.approx(.5*xx[key])
        for key in ('reward_coverage','reward_support_blend_coverage','reward_safety','reward_ce_pbrs','reward_ce_control'):
            assert yy[key]==pytest.approx(xx[key])
        assert y.rewards[i]-x.rewards[i]==pytest.approx(-.5*(xx['reward_capture']+xx['reward_terminal']))


def test_scale_does_not_touch_coverage_terminal_or_allow_online_mutation():
    a=make('coverage');b=make('coverage',alpha=.5)
    for env in (a,b):converge_fixture(SimpleNamespace(env=env,observations=env.get_observations()))
    x=a.step([4]*4,[]);y=b.step([4]*4,[])
    np.testing.assert_array_equal(x.rewards,y.rewards);assert a.post_capture_coverage_success
    b.reward_cfg['static_capture_scale']=.75
    with pytest.raises(ValueError):b._capture_scale()


def test_v2_consumer_cannot_bypass_resolver_with_broadcast_flag():
    env=make();env.per_cfg['global_evader_visibility']=True
    with pytest.raises(ValueError):env._vct_ls_direct_enemy_ids_for_pursuer(0)


def test_new_pure_capture_stream_matches_initial_state_and_collector(tmp_path):
    from cocap_voradj.training.forward_final_v2 import make_pure_capture_stream,make_fullmix_stream
    from cocap_voradj.training.runtime_semantics import initial_state_fingerprint
    old=st.SingleTaskStream('capture',19);new=make_pure_capture_stream(19)
    assert initial_state_fingerprint(old.env)==initial_state_fingerprint(new.env)
    trainer=st.p.make_trainer(st.contract()['scratch'],19,'cpu')
    row,episode=st.collect_transition(trainer,new)
    assert row['active_mask'].sum()==4 and row['local_obs']['self'].shape==(4,9)
    capture_fixture(new);out=new.step([4]*4)
    assert all(out.dones);rec=new.finish();assert rec['captured']
    assert 'onboard_sensing' in new.env.config['voradj']
    mixed=make_fullmix_stream(19,tmp_path)
    capture_fixture(mixed);out=mixed.step([4]*4)
    # A native or explicit reset retains V2 in both scheduled scenes.
    mixed.finish();assert 'onboard_sensing' in mixed.env.config['voradj']


def test_scaled_map_metadata_and_invalid_geometry():
    from cocap_voradj.envs.density_sensing import resolve_density_normalized_map
    for n in (4,8,12):
        m=resolve_density_normalized_map(n,120,120,14360)
        assert m['map_size']==[120.,120.] and m['map_linear_scale']==1
    m=resolve_density_normalized_map(64,120,120,14360)
    assert m['map_size'][0]>120 and m['resolved_onboard_radius']==pytest.approx(20)
    for k in (float('nan'),float('inf'),-1):
        with pytest.raises(ValueError):resolve_density_normalized_onboard_radius(4,120,120,14360,k=k)
