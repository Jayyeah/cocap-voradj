#!/usr/bin/env python3
"""Prepare reviewable manifests and fresh initialization hashes. No training CLI."""
import sys,json,copy,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
import yaml
from tools import forward_final_single_task_20260915 as st
from cocap_voradj.training.forward_final import flatten
from cocap_voradj.training.forward_final_v2 import new_scene_config,new_pure_capture_config,make_pure_capture_env,make_fullmix_stream
from cocap_voradj.envs.density_sensing import runtime_metadata,enable_v2,POLICY
OUT=ROOT/'artifacts/2026-09-15_normsense_v2';CFG=ROOT/'configs/experiments/forward_final_normsense_v2_20260915'


def diff(a,b,allowed):
    a,b=flatten(a),flatten(b);rows=[]
    for k in sorted(a.keys()|b.keys()):
        if a.get(k)!=b.get(k):
            assert k in allowed,f'Unexplained difference: {k}'
            rows.append(dict(field=k,before=a.get(k),after=b.get(k),classification=allowed[k]))
    return dict(UNEXPLAINED=0,differences=rows)


def value_norm_hash(value_norm):
    state={k:(v.detach().cpu().tolist() if torch.is_tensor(v) else v) for k,v in value_norm.state_dict().items()}
    return hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()


def main():
    torch.set_num_threads(1)
    geometry=json.loads((OUT/'geometry.json').read_text());visibility=json.loads((OUT/'capture_visibility.json').read_text())
    reward=json.loads((OUT/'reward_balance.json').read_text());coverage=json.loads((OUT/'coverage_transfer.json').read_text())
    alpha=reward['alpha_capture']
    if alpha is None:raise ValueError('No predeclared alpha: insufficient phase evidence')
    parent=st.contract()['scratch'];seed=2026091401
    hashes=[]
    for _ in range(2):
        trainer=st.p.make_trainer(parent,seed,'cpu')
        hashes.append(dict(actor=st.p.tensor_hash(trainer.actor.state_dict()),value=st.p.tensor_hash(trainer.value.state_dict()),value_norm=value_norm_hash(trainer.value_norm)))
    assert hashes[0]==hashes[1]
    shared=dict(schema='forward-final-normsense-run-v2',sensing_policy=POLICY,seed=seed,initial_hashes=hashes[0],ppo=parent['ppo'],actor=parent['actor'],critic=parent['critic'],actor_head=parent['actor_head'],rollout_length=256,reward_clock_offset=2000000,scene_schedule=parent['reset'],eval_seed_base=2026092401,eval_confirmation_seed_base=2026102401,eval_episodes_per_scene_mode=20,formal_training='NOT_STARTED',requires_published_delivery=True,initialization='random',teacher_dependency=0)
    names=('NormSense-Original-FullMix','NormSense-RewardBalanced-FullMix','NormSense-PureCapture')
    runs={};full_configs=[]
    for name,scale in zip(names[:2],(1.,alpha)):
        cfg={scene:new_scene_config(scene,alpha_capture=scale) for scene in ('mixed','coverage')};full_configs.append(cfg)
        stream=make_fullmix_stream(seed,OUT/('preflight_'+name),alpha_capture=scale)
        runs[name]=dict(shared,name=name,alpha_capture=scale,resolved=cfg,sensing_runtime={s:runtime_metadata(env) for s,env in stream.envs.items()})
    full_parity=diff(full_configs[0],full_configs[1],{s+'.reward.static_capture_scale':'STATIC_CAPTURE_SCALE_ONLY' for s in ('mixed','coverage')})
    capcfg=new_pure_capture_config();oldcfg=st.task_config('capture')
    allowed={f'voradj.{key}':'ONBOARD_SENSING_ONLY' for key in ('enemy_sensing_radius','obstacle_sensing_radius','vct_ls_enemy_sensing_radius','vct_ls_obstacle_sensing_radius')}
    allowed.update({f'voradj.onboard_sensing.{key}':'ONBOARD_SENSING_ONLY' for key in ('policy','schema_version','k','radius_floor')})
    cap_parity=diff(oldcfg,capcfg,allowed)
    capseed=st.contract()['spec']['seed'];trainer=st.p.make_trainer(parent,capseed,'cpu');env,_=make_pure_capture_env(capseed)
    old_launch=json.loads((ROOT/'artifacts/2026-09-15_single_task/capture/launch.json').read_text())
    caphashes=dict(actor=st.p.tensor_hash(trainer.actor.state_dict()),value=st.p.tensor_hash(trainer.value.state_dict()),value_norm=value_norm_hash(trainer.value_norm))
    assert caphashes['actor']==old_launch['actor_initial_sha256']
    assert caphashes['value']==old_launch['critic_initial_sha256']
    runs[names[2]]=dict(shared,name=names[2],seed=capseed,initial_hashes=caphashes,alpha_capture=1.,budget=500000,checkpoint_interval=25000,scene_schedule=['capture'],eval_seed_base=st.contract()['spec']['eval_seed_base'],eval_confirmation_seed_base=None,resolved=capcfg,sensing_runtime=runtime_metadata(env),parent_single_task_contract=st.contract(),old_launch_initial=dict(actor=old_launch['actor_initial_sha256'],value=old_launch['critic_initial_sha256']))
    geometry_ok=all(v['worst_blind']<=geometry['blind_tolerance'] for v in geometry['summary'].values()) and not visibility['global_visibility_veto']
    coverage_ok=coverage['verdict']!='COVERAGE_TRANSFER_BREAKS_POLICY'
    for name,run in runs.items():
        blockers=[]
        if not geometry_ok:blockers.append('GEOMETRY_OR_GLOBAL_VISIBILITY_GATE')
        if name!=names[2] and not coverage_ok:blockers.append('MASTER_COVERAGE_V2_TRANSFER_BREAKS_POLICY_REVIEW')
        if name==names[1] and alpha==1.:blockers.append('NO_CAPTURE_DOMINANCE_ALPHA_1_DUPLICATES_ORIGINAL_RECOMMEND_CANCEL')
        run['blockers']=blockers;run['READY']=not blockers;run['publication_gate']='commit + push required before any launch; no launch in this tool'
        spec={k:v for k,v in run.items() if k not in ('resolved','sensing_runtime','old_launch_initial','parent_single_task_contract')}
        spec['resolved_manifest']=f'artifacts/2026-09-15_normsense_v2/{name}.json'
        (CFG/(name+'.yaml')).write_text(yaml.safe_dump(spec,sort_keys=False,allow_unicode=True))
        st.p.write_json(OUT/(name+'.json'),run)
    report=dict(schema='normsense-v2-readiness-v1',runs={k:{key:v[key] for key in ('READY','blockers','initial_hashes','alpha_capture')} for k,v in runs.items()},fullmix_parity=full_parity,pure_capture_parity=cap_parity,fullmix_initial_hashes_matched=True,formal_training_started=False)
    st.p.write_json(OUT/'contracts.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
