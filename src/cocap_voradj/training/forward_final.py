"""Forward Final contract. Never resolve through the legacy AC capture ladder."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from cocap_voradj.training.trainer import load_config, deep_update, set_global_config
from cocap_voradj.training.runtime_semantics import assert_runtime, assert_only_changed
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/'configs/experiments/forward_final_mappo_20260908/stage1_4v1.yaml'
HISTORICAL = ROOT/'configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml'
CONTRACT = 'forward-final-aw9-4v1-swept-v1'
TEACHER = ROOT/'artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage1_4v1_step_2000000.pt'
TEACHER_SHA = '2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89'


def flatten(value, prefix=''):
    out={}
    for k,v in value.items():
        key=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict):out.update(flatten(v,key))
        else:out[key]=v
    return out


def scene_config(scene, *, historical=False):
    if scene not in ('mixed','coverage'):raise ValueError('Full-task scenes are mixed and coverage; capture is a mixed prefix diagnostic')
    root=load_config(str(HISTORICAL if historical else CONFIG))
    task='voradj_coverage' if scene=='coverage' else 'voradj'
    return deep_update(root,root['tasks'][task])


def check_env(env):
    facts=assert_runtime(env,expected={
        'topology':'friendly_voronoi_comm_v0','enemy_token_rule':'surface_radius',
        'global_enemy_flag':False,'support_capture_weight':.5,'support_coverage_weight':.5,
        'support_blend_enabled':True,'capture_reward_mode':'ring_importance_ms_v0',
        'capture_radius':8.,'capture_k':3,'enemy_radius':20.,'pursuers':4,
        'action_mode':'unicycle_discrete','decision_dt':.5,'physics_dt':.05,
        'a_longitudinal_max':.4,'omega_max':float(np.pi/6),'v_max':3.,
        'drag':.4/3,'collision_semantics':'synchronized_swept_v1'})
    assert env.collision_semantics=='synchronized_swept_v1'
    assert not env._pure_capture_all_capture_enabled()
    assert env._ce_coverage_enabled() and env._ring_importance_ms_enabled()
    assert env._support_reward_capture_component_mode()=='approach_only'
    assert env._support_reward_capture_target_mode()=='neighbor_visible'
    assert env._vct_ls_sensing_radius('obstacle')==20
    assert env._vct_ls_voronoi_obstacle_mode()=='free_mask_projected'
    assert env.episode_max_length==3000
    assert not env.config['voradj'].get('capture_episode_ends_on_capture',False)
    assert env.reward_cfg['post_capture_coverage_window_steps']==500
    assert env.reward_cfg['min_active_pursuers']==4
    assert env._pursuing_release_delay_steps==10 and env._vct_ls_apply_release_delay()
    for mapping in (env._capture_voronoi_map(),env._coverage_voronoi_map()):
        assert all(k[0]=='pursuer' for k in mapping['keys'])
    np.testing.assert_allclose(env.pursuers[0].action_list,[(a,w) for a in (-.4,0,.4) for w in (-np.pi/6,0,np.pi/6)])
    facts.update({'ce_enabled':True,'pure_capture_override':False,'episode_horizon':env.episode_max_length,
                  'capture_terminal':False,'post_capture_window':500,'min_active':4,
                  'ce_speed_weight_at_reset':env.reward_cfg['coverage_ce_speed_weight'],
                  'capture_sites':env._capture_voronoi_map()['keys']})
    return facts


def make_env(scene,seed):
    cfg=scene_config(scene);set_global_config(cfg)
    env=VorAdjEnv(copy.deepcopy(cfg),seed=int(seed));obs=env.reset()
    check_env(env)
    return env,obs


def preflight():
    canonical=json.loads((ROOT/'artifacts/2026-09-08_contract_parity/canonical_main_runtime.json').read_text())
    assert scene_config('mixed',historical=True)==canonical['profiles']['final4']['resolved'], 'canonical Final config drift'
    report={'contract':CONTRACT,'historical_main':canonical['commit'],'scenes':{}}
    for scene in ('mixed','coverage'):
        old=scene_config(scene,historical=True);new=scene_config(scene)
        # Missing collision field in historical config means legacy_end_step.
        before=flatten(old);before.setdefault('env.collision_semantics','legacy_end_step')
        changes=assert_only_changed(before,flatten(new),['env.collision_semantics'])
        set_global_config(old);legacy=VorAdjEnv(copy.deepcopy(old),seed=2026090801);legacy.reset()
        legacy_facts=assert_runtime(legacy)
        env,_=make_env(scene,2026090801)
        forward_facts=assert_runtime(env)
        live_changes=assert_only_changed(legacy_facts,forward_facts,['collision_semantics'])
        report['scenes'][scene]={'resolved':new,'runtime':check_env(env),'resolved_changed':changes,'runtime_changed':live_changes}
    return report
