"""Independent reviewer diagnostics. Never launches official smoke or production jobs."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from tools import zsupport_relative_rnorm_20261017 as run
from cocap_voradj.training.trainer import CoCapTrainer, set_global_config
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv

class ProbeStop(RuntimeError):
    pass

class FakeChild:
    pid = 99999999
    returncode = 0
    def poll(self): return 0
    def terminate(self): pass


def harness(tmp_path,monkeypatch):
    runtime=tmp_path/'runtime';runtime.mkdir()
    art=tmp_path/'art';run.write(art/'acceptance_gate.json',{'status':'PASS','engineering_blockers_resolved':True})
    monkeypatch.setattr(run,'ART',art)
    monkeypatch.setattr(run,'require_independent_review',lambda:None)
    monkeypatch.setattr(run,'prepare',lambda r:None)
    monkeypatch.setattr(run,'export',lambda r:None)
    monkeypatch.setattr(run,'reports',lambda r:None)
    monkeypatch.setattr(run,'sync_milestone',lambda r,l:None)
    monkeypatch.setattr(run,'launch_child',lambda *a,**kw:FakeChild())
    return runtime


def after_one_iteration_marker(monkeypatch,runtime,marker='pause_request.json'):
    calls=[]
    def sleep(seconds):
        calls.append(seconds)
        if len(calls)==1:run.write(runtime/marker,{'reason':'independent diagnostic'})
        else:raise ProbeStop('supervisor continued after hold; diagnostic stops after two iterations')
    monkeypatch.setattr(run.time,'sleep',sleep)
    return calls

@pytest.mark.parametrize('marker',['pause_request.json','scientific_failure.json'])
def test_contract_adopting_existing_training_honors_hold(tmp_path,monkeypatch,marker):
    runtime=harness(tmp_path,monkeypatch)
    run.write(runtime/'stages/stage1/training_started.json',{'pid':42,'process_start':'test'})
    monkeypatch.setattr(run,'alive',lambda pid,start=None:pid==42)
    after_one_iteration_marker(monkeypatch,runtime,marker)
    with pytest.raises(run.PauseRequested):run.supervise(runtime,'0')


def test_contract_selection_wait_honors_operator_pause(tmp_path,monkeypatch):
    runtime=harness(tmp_path,monkeypatch)
    run.write(runtime/'stages/stage1/training_complete.json',{'environment_decision_steps':2000000})
    after_one_iteration_marker(monkeypatch,runtime)
    with pytest.raises(run.PauseRequested):run.supervise(runtime,'0')


def test_contract_selection_does_not_handoff_after_operator_pause(tmp_path,monkeypatch):
    runtime=harness(tmp_path,monkeypatch)
    sd=runtime/'stages/stage1';run.write(sd/'training_complete.json',{'environment_decision_steps':2000000})
    ckpt=runtime/'selected.pt';ckpt.write_bytes(b'independent diagnostic only')
    selected={'checkpoint':str(ckpt),'checkpoint_sha256':run.sha256_file(ckpt)}
    run.enqueue('selection','stage1',{},'selection-test',runtime)
    with run.db(runtime) as c:c.execute("UPDATE jobs SET status='complete'")
    monkeypatch.setattr(run.old,'MILESTONES',{'stage1':[100000]})
    def select(stage_dir,milestones):
        payload={'selected':selected,'selection_reason':'diagnostic'}
        run.write(sd/'selection_report.json',payload)
        run.write(runtime/'pause_request.json',{'reason':'arrived while balanced selection ran'})
        return payload
    monkeypatch.setattr(run.old,'select_balanced',select)
    def sync(r,label):raise ProbeStop('reached post-handoff sync')
    monkeypatch.setattr(run,'sync_milestone',sync)
    try:run.supervise(runtime,'0')
    except (run.PauseRequested,ProbeStop):pass
    assert not (runtime/'stages/stage2/handoff.json').exists(),'stage2 handoff was published while operator pause was set'


def test_contract_formal_render_tail_honors_operator_pause(tmp_path,monkeypatch):
    runtime=harness(tmp_path,monkeypatch)
    for stage in run.old.STAGE_ORDER:
        sd=runtime/'stages'/stage
        run.write(sd/'training_complete.json',{'environment_decision_steps':run.BUDGETS[stage]})
        run.write(sd/'selection_report.json',{'selected':{'checkpoint':'diagnostic','checkpoint_sha256':'diagnostic'}})
    after_one_iteration_marker(monkeypatch,runtime)
    with pytest.raises(run.PauseRequested):run.supervise(runtime,'0')

@pytest.mark.parametrize('dependency',['tools/iqn_token_matched_20260919.py','tools/evaluate_iqn_z05_independent_20260923.py','tools/run_forward_final_bridge_20260908.py','configs/experiments/iqn_z_guided_support_full3stage_20261017/stage1_4p1e1obs_2m.yaml'])
def test_contract_source_hash_covers_live_dependencies(tmp_path,monkeypatch,dependency):
    root=tmp_path/'source';config=root/'configs/new';config.mkdir(parents=True)
    (root/'src').mkdir();(root/'src/env.py').write_text('x=1\n')
    for rel in ['tools/zsupport_relative_rnorm_20261017.py','tools/zsupport_relative_rnorm_reports_20261017.py','tools/smoke_relative_z_rnorm_20261017.py','tests/test_relative_z_rnorm_20261017.py',dependency]:
        path=root/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('original\n')
    monkeypatch.setattr(run,'ROOT',root);monkeypatch.setattr(run,'CONFIG_DIR',config)
    before=run.reviewed_source_hash();(root/dependency).write_text('changed live implementation\n')
    assert before!=run.reviewed_source_hash(),f'review approval hash ignores active dependency {dependency}'


def tiny(tmp,steps):
    cfg=run.config('stage1',runtime=tmp,device='cpu');cfg.update(train_mode='voradj',total_timesteps=steps)
    cfg['iqn'].update(hidden_dim=32,num_heads=4,num_layers=1,num_quantiles=4,num_cosine_features=8,training_quantile_samples=2,target_quantiles=2,batch_size=4,min_replay_size=4,replay_capacity=128,train_freq=1,target_update_freq=2,checkpoint_freq=6,log_freq_steps=1)
    return cfg

def digest(x):
    if torch.is_tensor(x):return str(x.dtype),tuple(x.shape),x.cpu().numpy().tobytes()
    if isinstance(x,np.ndarray):return str(x.dtype),x.shape,x.tobytes()
    if isinstance(x,dict):return {k:digest(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return tuple(digest(v) for v in x)
    return x

def load(root):return torch.load(root/'stages/stage1/training/checkpoints/resume_latest.pt',map_location='cpu',weights_only=False)


def test_new_queue_trainer_actual_updates_preserve_baseline_and_exact_resume(tmp_path):
    torch.set_num_threads(1)
    base=CoCapTrainer(tiny(tmp_path/'base',12));base.train();p0=load(tmp_path/'base')
    queued=run.QueueTrainer(tiny(tmp_path/'queued',12));queued.train();p1=load(tmp_path/'queued')
    assert queued.update_steps>0
    for k in ('model','target_model','optimizer','rng','z_states'):assert digest(p0[k])==digest(p1[k]),k
    cfg=tiny(tmp_path/'split',12);first=run.QueueTrainer(cfg);first.total_timesteps=6;first.train()
    resumed=copy.deepcopy(cfg);resumed['checkpointing']['resume_path']=str(tmp_path/'split/stages/stage1/training/checkpoints/resume_latest.pt')
    second=run.QueueTrainer(resumed);second.train();p2=load(tmp_path/'split')
    for k in ('model','target_model','optimizer','rng','z_states'):assert digest(p1[k])==digest(p2[k]),k
    assert digest(p1['runtime']['current_observations'])==digest(p2['runtime']['current_observations'])
    for i,snap in p1['runtime']['envs']['voradj']._support_policy_snapshots.items():assert digest(snap)==digest(p2['runtime']['envs']['voradj']._support_policy_snapshots[i])

@pytest.mark.parametrize('stage',['stage1','stage2','stage3'])
@pytest.mark.parametrize('strict',[True,False])
@pytest.mark.parametrize('side',[100.,173.])
def test_success_rms_and_max_physical_boundary_all_stages(stage,strict,side):
    from tools import iqn_z_unified_decay_curriculum_20260919 as legacy
    envs=[]
    for cfg in [legacy.resolved('z05',stage),run.formal.resolved(run.CONFIGS[stage])]:
        cfg=run.formal.scene_config(cfg,'coverage');cfg['env'].update(width=side,height=side,x_boundary_left=0,x_boundary_right=side,y_boundary_bottom=0,y_boundary_top=side)
        set_global_config(cfg);env=VorAdjEnv(copy.deepcopy(cfg),seed=55);env.reset();envs.append(env)
    a,b=envs;n=len(a.pursuers);prefix='coverage_ce_success' if strict else 'coverage_ce_loose'
    rms=a.reward_cfg.get(f'{prefix}_rms_threshold',a.reward_cfg.get('coverage_ce_success_rms_threshold',.05))*np.sqrt(2)*side/np.sqrt(n)
    mx=a.reward_cfg.get(f'{prefix}_max_threshold',a.reward_cfg.get('coverage_ce_success_max_threshold',.10))*np.sqrt(2)*side/np.sqrt(n)
    for mode,threshold in [('rms',rms),('max',mx)]:
        for factor in [1-1e-7,1+1e-7]:
            distances=np.repeat(threshold*factor,n) if mode=='rms' else np.r_[threshold*factor,np.zeros(n-1)]
            outputs=[]
            for env in envs:
                data={'centroids':{('pursuer',i):env._position(p)+np.array([distances[i],0]) for i,p in enumerate(env.pursuers)},'counts':{('pursuer',i):100 for i in range(n)}}
                outputs.append(env._ce_coverage_geometry(data,strict))
            assert outputs[0]['geometry_strict']==outputs[1]['geometry_strict']
            assert outputs[1]['geometry_strict']==(factor<1), (stage,strict,side,mode,factor,outputs)
            for key in ['ce_physical_rms_threshold','ce_physical_max_threshold','ce_center_physical_rms','ce_center_physical_max']:assert outputs[0][key]==pytest.approx(outputs[1][key])

@pytest.mark.parametrize('diagnostic_error',['ValueError: runtime assertion map_scale_contract: expected side_length_r, got legacy_diagonal','AssertionError: friend z_j token mapping mismatch'])
def test_contract_scientific_worker_failure_is_not_ordinary_retry(tmp_path,monkeypatch,diagnostic_error):
    from types import SimpleNamespace
    runtime=tmp_path/'worker';run.enqueue('selection','stage1',{},'scientific-test',runtime)
    original_claim=run.claim;calls=[]
    def claim(kinds,runtime):
        calls.append(1)
        if len(calls)>1:raise ProbeStop('one failed attempt is sufficient to inspect retry state')
        return original_claim(kinds,runtime)
    def fake_process(*a,stdout=None,**kw):
        stdout.write(diagnostic_error+'\n');stdout.flush()
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(run,'claim',claim);monkeypatch.setattr(run.subprocess,'run',fake_process);monkeypatch.setattr(run,'export',lambda r:None)
    with pytest.raises(ProbeStop):run.worker(('selection',),runtime)
    assert (runtime/'scientific_failure.json').is_file(),'scientific contract assertion was treated as a retryable job crash'
    assert (runtime/'pause_request.json').is_file(),'scientific contract assertion did not hold subsequent work'
