from __future__ import annotations
import copy, importlib.util, json, os, random, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
import torch
from cocap_voradj.training.trainer import CoCapTrainer
from tools import iqn_token_matched_20260919 as formal
from tools import z_support_full3stage_20261017 as run
from tools import z_guided_support_20261016 as pilot


def tiny(tmp,steps):
    cfg=run.config('stage1',runtime=tmp,device='cpu')
    cfg.update(train_mode='voradj',total_timesteps=steps)
    cfg['iqn'].update(hidden_dim=32,num_heads=4,num_layers=1,num_quantiles=4,num_cosine_features=8,training_quantile_samples=2,target_quantiles=2,batch_size=4,min_replay_size=4,replay_capacity=128,train_freq=1,target_update_freq=2,checkpoint_freq=6,log_freq_steps=1)
    return cfg

def digest(value):
    if isinstance(value,torch.Tensor):return (str(value.dtype),tuple(value.shape),value.cpu().numpy().tobytes())
    if isinstance(value,np.ndarray):return (str(value.dtype),value.shape,value.tobytes())
    if isinstance(value,dict):return {k:digest(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return tuple(digest(v) for v in value)
    return value


def payload(root):return torch.load(root/'stages/stage1/training/checkpoints/resume_latest.pt',map_location='cpu',weights_only=False)

def test_queue_does_not_block_or_change_training_and_mid_episode_resume(tmp_path):
    torch.set_num_threads(1)
    base=CoCapTrainer(tiny(tmp_path/'base',12));base.train();p0=payload(tmp_path/'base')
    q=run.QueueTrainer(tiny(tmp_path/'async',12));q.train();p1=payload(tmp_path/'async')
    for k in ('model','target_model','optimizer','rng','z_states'):assert digest(p0[k])==digest(p1[k]),k
    assert p1['runtime']['global_step']==12 and p1['runtime']['update_steps']>0
    with run.db(tmp_path/'async') as c:
        jobs=c.execute('SELECT * FROM jobs').fetchall()
        assert len(jobs)==2 and all(j['status']=='pending' for j in jobs)
    assert p1['runtime']['episode_idx']==0 # snapshots happened while first episode was unfinished
    cfg=tiny(tmp_path/'split',12);first=run.QueueTrainer(cfg);first.total_timesteps=6;first.train()
    resumed=copy.deepcopy(cfg);resumed['checkpointing']['resume_path']=str(tmp_path/'split/stages/stage1/training/checkpoints/resume_latest.pt')
    second=run.QueueTrainer(resumed);second.train();p2=payload(tmp_path/'split')
    for k in ('model','target_model','optimizer','rng','z_states'):assert digest(p1[k])==digest(p2[k]),k
    assert digest(p1['runtime']['current_observations'])==digest(p2['runtime']['current_observations'])
    for i,snap in p1['runtime']['envs']['voradj']._support_policy_snapshots.items():assert digest(snap)==digest(p2['runtime']['envs']['voradj']._support_policy_snapshots[i])
    assert (tmp_path/'split/stages/stage1/training/checkpoints/step_6.pt').stat().st_mode&0o222==0
    # A snapshot name can never silently acquire different policy parameters.
    with torch.no_grad():next(second.model.parameters()).add_(1)
    with pytest.raises(RuntimeError,match='immutable'):second._save_checkpoint('step_12.pt')


def test_queue_claim_dedup_and_dead_worker_recovery(tmp_path):
    for _ in range(2):run.enqueue('selection','stage1',{'step':1},'one',tmp_path)
    run.enqueue('render','stage1',{},'two',tmp_path)
    r=run.claim(('selection',),tmp_path);assert r['id']=='one'
    assert run.claim(('selection',),tmp_path) is None
    with run.db(tmp_path) as c:c.execute('UPDATE jobs SET worker_pid=? WHERE id=?',(99999999,'one'))
    r=run.claim(('selection',),tmp_path);assert r['attempts']==1
    with run.db(tmp_path) as c:
        c.execute('UPDATE jobs SET worker_pid=?,attempts=? WHERE id=?',(99999999,3,'one'))
    assert run.claim(('selection',),tmp_path) is None
    with run.db(tmp_path) as c:assert c.execute('SELECT status FROM jobs WHERE id=?',('one',)).fetchone()[0]=='failed'
    assert run.claim(('render',),tmp_path)['id']=='two'


def test_stationary_fallback_keeps_real_threshold_and_type():
    cfg=run.formal.resolved(run.CONFIGS['stage1'])
    env,obs=pilot.env_make(cfg,'capture',2026101709)
    for p,(x,y) in zip(env.pursuers,[(54,60),(66,60),(20,20),(100,100)]):
        p.x=x;p.y=y;p.speed=0;p.deactivated=False
    e=env.evaders[0];e.x=60;e.y=60;e.speed=0;e.deactivated=False
    env._stationary_capture_counters={}
    hold=cfg['reward']['capture_stationary_hold_steps'];assert hold==10
    for _ in range(hold-1):assert env._loose_capture_events()==[]
    event=env._loose_capture_events()[0]
    assert event['capture_type']=='stationary' and event['participants']==[0,1]
    e.speed=.21;env._stationary_capture_counters={}
    for _ in range(hold+1):assert not env._loose_capture_events()


def test_nonfinite_gradient_is_caught_before_optimizer_update():
    p=torch.nn.Parameter(torch.ones(2));o=torch.optim.Adam([p]);o.register_step_pre_hook(run.finite_optimizer_step)
    p.grad=torch.tensor([float('nan'),1.])
    with pytest.raises(FloatingPointError):o.step()
    assert torch.equal(p,torch.ones(2)) and not o.state


def test_native_sample_matches_registered_model_act():
    torch.set_num_threads(1)
    cfg=formal.resolved(run.CONFIGS['stage1']);env,obs=pilot.env_make(cfg,'mixed',2026101710)
    model=run.CoCapIQN(formal.model_config(cfg)).eval();local=[o for o in obs if o is not None]
    torch.manual_seed(31);_,actual=run.sample_q(run.NativeSamplePolicy(model),local,'cpu')
    torch.manual_seed(31);expected=model.act(run.stack_obs(local,'cpu'),mode='voradj',epsilon=0,deterministic_quantiles=False).numpy()
    np.testing.assert_array_equal(actual,expected)


def test_frozen_independent_runner_trajectory_and_render_smoke(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    sd=tmp_path/'stages/stage1';sd.mkdir(parents=True)
    cfg=formal.resolved(run.CONFIGS['stage1']);(sd/'resolved.yaml').write_text(run.yaml.safe_dump(cfg))
    # Read-only historical parent is diagnostic input only, never production initialization.
    selected={'selected':{'checkpoint':str(pilot.PARENT),'checkpoint_sha256':pilot.PARENT_SHA,'step':1300000}}
    run.write(sd/'selection_report.json',selected)
    primary=run.formal_batch('stage1','argmax',tmp_path,episodes=1,smoke_steps=3)
    secondary=run.formal_batch('stage1','sample',tmp_path,episodes=1,smoke_steps=3)
    recovered=run.formal_batch('stage1','argmax',tmp_path,episodes=1,smoke_steps=3)
    assert recovered['summary']==primary['summary']
    for r in (primary,secondary):
        assert len(r['records'])==3
        for row in r['records']:
            assert row['smoke_horizon_override']==3 and Path(row['trajectory']).is_file() and Path(row['events']).is_file()
    run.render('stage1',{'mode':'argmax','scene':'mixed','episode_index':0},tmp_path)
    assert len(list((sd/'independent/argmax/gifs').rglob('*.gif')))==1
    # The adapter must yield the same episode as the unchanged historical runner.
    from tools import run_forward_final_bridge_20260908 as runner
    model=run.CoCapIQN.load(str(pilot.PARENT),device='cpu').eval();holder={}
    row=runner.run_episode(model,'mixed',run.FINAL_SEED+200000,'cpu',max_steps=3,env_factory=lambda scene,seed:run.independent.make_env_factory(cfg,scene,seed,'native',model,holder))
    actual=next(r for r in primary['records'] if r['scene']=='mixed')
    for k in row:assert digest(row[k])==digest(actual[k]),k
