#!/usr/bin/env python3
"""CPU-only Final/BC lineage, effective consumer and reward-path audit."""
from __future__ import annotations
import copy
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from cocap_voradj.training.trainer import load_config, deep_update, set_global_config
from cocap_voradj.training.continuous.formal_config import scene_config, resolve_ladder_config
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.runtime_semantics import assert_runtime
from tools.collect_iqn_aw_teacher_dataset_20260903 import stable_hash
from tools.evaluate_iqn_corrected_capture import DEFAULT_COLLISION_SEMANTICS
from tools.probe_cf3_policy_temperature import apply_collision_semantics_override
from tools.run_small_step_ac_migration import configure_environment, make_components
from tools.audit_p0_information_20260908 import setup, refresh, vector_policy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/2026-09-08_contract_parity'
AC = ROOT / 'configs/experiments/mappo9_v2_20260830/seed1.yaml'
FINAL = ROOT / 'configs/experiments/cr_ms_support_approach_ce_curriculum_20260802'


def profiles():
    result = {}
    for key, name in [('final4', 'stage1_4p1e1obs_scratch2m.yaml'), ('final8', 'stage2_8p2e2obs_700k.yaml'), ('final12', 'stage3_12p3e3obs_700k.yaml')]:
        root = load_config(str(FINAL / name))
        result[key] = deep_update(root, root.get('tasks', {}).get('voradj', {}))
    raw = load_config(str(AC))
    result['bc_dataset'] = scene_config(apply_collision_semantics_override(raw, DEFAULT_COLLISION_SEMANTICS), 'capture')
    result['mappo_train'] = configure_environment(resolve_ladder_config(AC), 'mappo9_v2')
    result['bc_eval'] = scene_config(apply_collision_semantics_override(resolve_ladder_config(AC), DEFAULT_COLLISION_SEMANTICS), 'capture')
    return result


def make_env(cfg):
    set_global_config(cfg)
    env = VorAdjEnv(copy.deepcopy(cfg), seed=2026090301)
    env.reset()
    return env


def facts(cfg):
    env = make_env(cfg)
    result = assert_runtime(env)
    result.update({
        'bounds': env._bounds(), 'episode_horizon': env.episode_max_length,
        'pre_capture_horizon': env.env_cfg.get('pre_capture_max_length', 0),
        'aw_grid': env.pursuers[0].action_list,
        'capture_voronoi_keys': env._capture_voronoi_map()['keys'],
        'coverage_voronoi_keys': env._coverage_voronoi_map()['keys'],
        'capture_obstacle_mode': env._vct_ls_voronoi_obstacle_mode() if env._vct_ls_enabled() else env._capture_voronoi_obstacle_mode(),
        'coverage_obstacle_mode': env._vct_ls_voronoi_obstacle_mode() if env._vct_ls_enabled() else env._voronoi_obstacle_mode(),
        'ring_ms_enabled': env._ring_importance_ms_enabled(),
        'pure_capture_all_capture_enabled': env._pure_capture_all_capture_enabled(),
        'ce_helper_enabled': env._ce_coverage_enabled(),
        'ce_task_path_reachable': not env._pure_capture_all_capture_enabled(),
        'obstacle_radius_getter': env._vct_ls_sensing_radius('obstacle'),
        'spawn': {k:v for k,v in env.env_cfg.items() if 'spawn' in k or 'initial' in k},
        'reward_consumer_config': env.reward_cfg,
        'voradj_consumer_config': env.config.get('voradj', {}),
        'perception_consumer_config': env.per_cfg,
        'env_consumer_config': env.env_cfg,
        'initial_entities': {name:[{k:getattr(p,k,None) for k in ('x','y','r','speed','theta','dt','N','max_speed','coefficient_water_resistance')} for p in getattr(env,name)] for name in ('pursuers','evaders')},
        'token_shapes': {k:list(np.asarray(v).shape) for k,v in env.get_observations()[0].items()},
    })
    return result


def reward_probe(cfg, terminal=False):
    env = setup(cfg)
    positions = [(40,50),(50,40),(60,50),(10,10)]
    if terminal:
        positions = [(50+7*np.cos(t),50+7*np.sin(t)) for t in (0,2*np.pi/3,4*np.pi/3)] + [(10,10)]
    for p,xy in zip(env.pursuers,positions):
        p.x,p.y=map(float,xy);p.theta=0.;p.speed=0. if terminal else 1.;p.velocity=np.array([p.speed,0.])
    e=env.evaders[0];e.x,e.y=50.,50.;e.theta=0.;e.speed=0.;e.velocity=np.zeros(2)
    refresh(env)
    actions=[np.zeros(2) for _ in env.pursuers] if env.continuous_aw_action else [4]*4
    outcome=env.step(actions,[4])
    return {'rewards':outcome.rewards,'dones':outcome.dones,'capture_events':env.last_capture_events,
            'infos':outcome.infos,'metrics':env.last_voradj_metrics}


def derived_probe(cfg, actor):
    env=setup(cfg);before=refresh(env);base=vector_policy(actor,before[0],False)
    env.evaders[0].x=48.;after=refresh(env)
    clearances=[float(np.linalg.norm([p.x-env.evaders[0].x,p.y-env.evaders[0].y])-p.r-env.evaders[0].r) for p in env.pursuers]
    assert min(clearances)>20
    clamped=copy.deepcopy(after[0]);clamped['evaders']=before[0]['evaders'].copy()
    is_enemy=np.asarray(clamped['types'])==2
    clamped['masks'][is_enemy]=before[0]['masks'][is_enemy]
    delta={k:float(np.max(np.abs(np.asarray(after[0][k],dtype=float)-np.asarray(before[0][k],dtype=float)))) for k in before[0]}
    return {'all_enemy_surface_distances_exceed_20m':True,'after_move_surface_clearances':clearances,'observation_delta':delta,
            'logit_delta_with_explicit_enemy_tokens_held_fixed':float(np.max(np.abs(base-vector_policy(actor,clamped,False)))),
            'interpretation':'Mediation diagnostic: hold explicit enemy rows fixed; other real preprocessing changes still affect frozen BC logits. Not a naturally generated clamped observation.'}


def json_default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,set):return sorted(value)
    raise TypeError(type(value).__name__)


def main():
    torch.set_num_threads(4);torch.manual_seed(20260908);OUT.mkdir(exist_ok=True)
    configs=profiles()
    manifest=json.loads((ROOT/'artifacts/2026-09-03_iqn_mappo_bc/teacher_dataset/manifest.json').read_text())
    actual_hash=stable_hash(configs['bc_dataset'])
    assert actual_hash==manifest['config_hash'], (actual_hash,manifest['config_hash'])
    data={'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'device':'cpu','dataset_config_hash_exact_match':actual_hash,'profiles':{k:facts(v) for k,v in configs.items()},'reward_probes':{k:{'dense':reward_probe(configs[k]),'capture_terminal':reward_probe(configs[k],True)} for k in ('final4','bc_dataset')}}
    trainer,_=make_components(configs['mappo_train'],'mappo9_v2','cpu')
    actor=trainer.actor
    actor.load_state_dict(torch.load(ROOT/'artifacts/2026-09-03_iqn_mappo_bc/distillation/distilled_actor.pt',map_location='cpu',weights_only=True)['actor_state_dict'],strict=True);actor.eval()
    from cocap_voradj.models.iqn import CoCapIQN
    teacher=CoCapIQN.load(str(ROOT/'artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage2_8v2_step_300000.pt'),device='cpu').eval()
    mapped=actor.encoder.map_legacy_iqn_decision_keys(teacher.state_dict())
    mismatches=[k for k,v in actor.encoder.state_dict().items() if not torch.equal(v,mapped[k])]
    data['backbone_checkpoint_parity']={'mapped_keys':len(mapped),'mismatched_keys':mismatches,'bit_exact':not mismatches}
    assert not mismatches
    base=assert_runtime(make_env(configs['bc_dataset']))
    for name in ('mappo_train','bc_eval'):
        assert assert_runtime(make_env(configs[name]))==base, name
    data['bc_dataset_train_eval_runtime_core_equal']=True
    for key in ('dense','capture_terminal'):
        rows=data['reward_probes']['bc_dataset'][key]['infos']
        assert all(r['replay_metadata']['reward_coverage']==0 and r['replay_metadata']['reward_ce_pbrs']==0 for r in rows)
    for name,count in [('final4',3),('bc_dataset',4)]:
        rows=data['reward_probes'][name]['capture_terminal']['infos']
        assert sum(r['replay_metadata']['reward_terminal']>0 for r in rows)==count
    assert data['reward_probes']['final4']['dense']['infos'][3]['replay_metadata']['support_reward_blend_active']
    data['derived_enemy_path']=derived_probe(configs['bc_dataset'],actor)
    assert data['profiles']['final4']['ring_ms_enabled']
    assert not data['profiles']['bc_dataset']['ring_ms_enabled']
    assert data['profiles']['bc_dataset']['pure_capture_all_capture_enabled']
    assert all(data['reward_probes']['bc_dataset']['capture_terminal']['dones'])
    assert not any(data['reward_probes']['final4']['capture_terminal']['dones'])
    assert data['derived_enemy_path']['logit_delta_with_explicit_enemy_tokens_held_fixed']>0
    (OUT/'resolved_profiles.json').write_text(json.dumps(configs,indent=2,default=json_default)+'\n')
    (OUT/'runtime_audit.json').write_text(json.dumps(data,indent=2,default=json_default)+'\n')
    print(json.dumps({'status':'PASS','dataset_hash':actual_hash,'derived_enemy_path':data['derived_enemy_path']},indent=2))

if __name__=='__main__':main()
