#!/usr/bin/env python3
"""CPU-only state -> Voronoi -> token -> frozen policy perturbation audit."""
from __future__ import annotations
import copy, hashlib, json, os
from pathlib import Path
import numpy as np
import torch
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.trainer import load_config, deep_update, set_global_config
from cocap_voradj.training.continuous.formal_config import resolve_ladder_config
from cocap_voradj.training.runtime_semantics import runtime_facts, assert_only_changed, assert_config_namespaces
from cocap_voradj.training.small_step_ac import tensor_tree
from tools.run_small_step_ac_migration import configure_environment, make_components

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/2026-09-08_p0/information_audit.json'
FINAL='configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml'
AC='configs/experiments/mappo9_v2_20260830/seed1.yaml'


def configs():
    final=load_config(str(ROOT/FINAL));final=deep_update(copy.deepcopy(final),final.get('tasks',{}).get('voradj',{}))
    ac=configure_environment(resolve_ladder_config(ROOT/AC),'mappo9_v2')
    return {'final':final,'ac':ac}


def setup(cfg):
    set_global_config(cfg);env=VorAdjEnv(copy.deepcopy(cfg),seed=20260908);env.reset()
    assert len(env.pursuers)==4 and len(env.evaders)==1 and len(env.obstacles)==1
    for p,xy in zip(env.pursuers,[(10,10),(10,70),(70,10),(70,70)]):
        p.x,p.y=xy;p.theta=0.;p.speed=0.;p.velocity=np.zeros(2);p.deactivated=False;p.is_pursuing=False
    env.evaders[0].x,env.evaders[0].y=42.,42.;env.evaders[0].velocity=np.zeros(2);env.evaders[0].deactivated=False
    env.obstacles[0].x,env.obstacles[0].y=32.,32.;env.obstacles[0].r=3.
    env._pursuing_release_counters=[0]*4
    return env


def refresh(env):
    env._invalidate_voronoi_cache()
    raw=env._raw_task_labels_from_map(env._capture_voronoi_map())
    env._update_effective_pursuing_flags(raw)
    return env.get_observations()


def vector_policy(policy,obs,iqn):
    batch=tensor_tree({k:np.asarray(v)[None] for k,v in obs.items()},torch.device('cpu'))
    with torch.no_grad():
        if iqn:
            tau=(torch.arange(32)+.5)/32
            return policy(batch,num_tau=32,mode='voradj',tau=tau)['q_values'].mean(1).numpy()[0]
        return policy.distribution(batch).logits.numpy()[0]


def audit():
    torch.set_num_threads(4);torch.manual_seed(20260908)
    profiles=configs();result={'schema':'p0-information-audit-v1','device':'cpu','profiles':{}}
    for name,cfg in profiles.items():
        env=setup(cfg);before=refresh(env)
        if name=='final':
            policy=CoCapIQN.load(str(ROOT/'artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage1_4v1_step_2000000.pt'),device='cpu').eval()
        else:
            trainer,_=make_components(cfg,'mappo9_v2','cpu');policy=trainer.actor
            payload=torch.load(ROOT/'artifacts/2026-09-03_iqn_mappo_bc/distillation/distilled_actor.pt',map_location='cpu',weights_only=True)
            policy.load_state_dict(payload['actor_state_dict'],strict=True);policy.eval()
        base_score=vector_policy(policy,before[0],name=='final')
        row={'runtime':runtime_facts(env),'initial_enemy_surface_clearances':[float(np.linalg.norm([p.x-env.evaders[0].x,p.y-env.evaders[0].y])-p.r-env.evaders[0].r) for p in env.pursuers], 'initial_enemy_token_counts':[int(np.asarray(o['masks'])[np.asarray(o['types'])==2].sum()) for o in before]}
        env.evaders[0].x=48.;after=refresh(env)
        row['hidden_enemy_move']={'max_token_delta_by_agent':[max(float(np.max(np.abs(np.asarray(a[k],dtype=float)-np.asarray(b[k],dtype=float)))) for k in a) for a,b in zip(before,after)],'focal_policy_score_max_delta':float(np.max(np.abs(base_score-vector_policy(policy,after[0],name=='final'))))}
        env.evaders[0].x=42.;refresh(env);env.obstacles[0].x,env.obstacles[0].y=34.,30.;obs=refresh(env)
        row['known_map_outside_sensor_move']={'focal_centroid_feature_delta':(np.asarray(obs[0]['self'])[6:8]-np.asarray(before[0]['self'])[6:8]).tolist(),'focal_obstacle_token_count':int(np.asarray(obs[0]['masks'])[np.asarray(obs[0]['types'])==3].sum())}
        env.evaders[0].x,env.evaders[0].y=18.,18.;visible=refresh(env)
        row['visible_enemy_positive_control_token_count']=int(np.asarray(visible[0]['masks'])[np.asarray(visible[0]['types'])==2].sum())
        result['profiles'][name]=row
    assert result['profiles']['final']['initial_enemy_token_counts']==[0]*4
    assert result['profiles']['final']['hidden_enemy_move']['max_token_delta_by_agent']==[0]*4
    assert any(result['profiles']['ac']['initial_enemy_token_counts'])
    assert result['profiles']['ac']['hidden_enemy_move']['focal_policy_score_max_delta']>0
    assert result['profiles']['final']['visible_enemy_positive_control_token_count']==1
    # A semantic NOOP and the wrong namespace must fail, even when YAML diff passes.
    env=setup(profiles['final']);before=runtime_facts(env)
    candidate=copy.deepcopy(profiles['final']);candidate['voradj']['support_reward_capture_weight']=1.;candidate['voradj']['support_reward_coverage_weight']=1.
    after=runtime_facts(setup(candidate))
    result['valid_support_runtime_diff']=assert_only_changed(before,after,['support_capture_weight','support_coverage_weight'])
    invalid=copy.deepcopy(profiles['final']);invalid['reward']['support_reward_capture_weight']=1.
    try:assert_config_namespaces(invalid)
    except ValueError as e:result['invalid_namespace_rejected']=str(e)
    else:raise AssertionError('wrong namespace accepted')
    result['limits']='Controlled perturbations verify these scenes; static source audit establishes broader information dependencies. No training or GPU use.'
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':audit()
