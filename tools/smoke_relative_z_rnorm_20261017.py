#!/usr/bin/env python3
"""Review-gated diagnostics only; no diagnostic state is a production parent."""
from __future__ import annotations
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from tools import zsupport_relative_rnorm_20261017 as run
from cocap_voradj.training.trainer import set_global_config
from cocap_voradj.training.runtime_semantics import assert_runtime


def config(runtime,steps=64,device='cpu'):
    cfg=run.config('stage1',runtime=runtime,device=device)
    cfg.update(total_timesteps=steps)
    cfg['iqn'].update(batch_size=16,min_replay_size=4,replay_capacity=256,train_freq=1,checkpoint_freq=32,log_freq_steps=8,target_update_freq=16)
    cfg['voradj'].update(replay_batch_counts=dict(pursuing=8,pre_capture_cover=2,post_capture_real=4,recovery_pure=2),real_post_min_replay_size=4)
    # Full production network and real original physics/targets; only learning
    # thresholds/buffer/budget shrink for diagnostics, never for formal launch.
    return cfg


def digest(x):
    if torch.is_tensor(x):return (str(x.dtype),tuple(x.shape),x.cpu().numpy().tobytes())
    if isinstance(x,np.ndarray):return (str(x.dtype),x.shape,x.tobytes())
    if isinstance(x,dict):return {k:digest(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return tuple(digest(v) for v in x)
    return x


def fill_real_replays(t):
    """Collect diagnostic physical rollouts, including actual normal capture."""
    env=t.envs['voradj'];records=[]
    for episode in range(8):
        set_global_config(env.config);env.reset()
        for o in env.obstacles:o.x=105.;o.y=105.
        for p,(x,y) in zip(env.pursuers,[(54.,60.),(63.,65.196152),(63.,54.803848),(20.,20.)]):
            p.x=x;p.y=y;p.theta=0.;p.speed=0.;p.deactivated=False;p.collision=False;p.is_pursuing=False
        e=env.evaders[0];e.x=e.y=60.;e.theta=0.;e.speed=0.;e.deactivated=False;e.collision=False
        env._pursuing_release_counters=[0]*4;env._pursuing_flags_initialized=False
        env._invalidate_voronoi_cache();env._reset_z_state();env._advance_z_state(env._capture_voronoi_map());obs=env.get_observations()
        for step in range(8):
            result=env.step([4]*4,[4]);records.extend(env.last_capture_events)
            for i,o in enumerate(obs):
                if o is None:continue
                meta=dict(result.infos[i]['replay_metadata']);meta['scene']='voradj';key=t._voradj_replay_class(meta);meta['buffer_class']=key
                t.replays[key].add(o,4,float(result.rewards[i]),result.observations[i],result.dones[i],metadata=meta)
            obs=result.observations
            if all(result.dones):break
    coverage=t.envs['voradj_coverage'];set_global_config(coverage.config);obs=coverage.reset()
    for _ in range(8):
        result=coverage.step([4]*4,[])
        for i,o in enumerate(obs):
            if o is None:continue
            meta=dict(result.infos[i]['replay_metadata']);meta['scene']='voradj_coverage';key=t._voradj_replay_class(meta);meta['buffer_class']=key
            t.replays[key].add(o,4,float(result.rewards[i]),result.observations[i],result.dones[i],metadata=meta)
        obs=result.observations
        if all(result.dones):break
    assert any(e.get('capture_type')=='loose' for e in records),records
    assert all(len(b)>=4 for b in t.replays.values()),{k:len(b) for k,b in t.replays.items()}
    return dict(replay_sizes={k:len(b) for k,b in t.replays.items()},normal_events=len(records))


def payload(runtime):return torch.load(runtime/'stages/stage1/training/checkpoints/resume_latest.pt',map_location='cpu',weights_only=False)


def configure_probe_evidence(runtime):
    # Subprocess-local diagnostic output remap; production globals/configs stay frozen.
    from tools import zsupport_relative_rnorm_reports_20261017 as reports
    run.ART=runtime/'probe_artifacts';run.DOC=runtime/'probe_reports'
    reports.ART=run.ART;reports.DOC=run.DOC


def run_child(runtime,device,resume=False,steps=64,probe=False):
    cfg=config(runtime,steps,device)
    if resume:cfg['checkpointing']['resume_path']=str(runtime/'stages/stage1/training/checkpoints/resume_latest.pt')
    if probe:
        # Prevent scheduled selection jobs in the tiny real coordinator probe.
        cfg['iqn']['checkpoint_freq']=100000
        configure_probe_evidence(runtime)
        class ProbeTrainer(run.QueueTrainer):
            def _select_actions(self,task,obs_list):
                run.write(runtime/'probe_progress.json',dict(pid=os.getpid(),step=self.global_step,updates=self.update_steps))
                return super()._select_actions(task,obs_list)
        t=ProbeTrainer(cfg)
        run.write(runtime/'stages/stage1/training_started.json',dict(pid=os.getpid(),process_start=run.process_start(os.getpid()),start_step=t.global_step,diagnostic_only=True))
    else:t=run.QueueTrainer(cfg)
    if not resume:fill_real_replays(t)
    t.train()


def supervisor_probe(runtime,device):
    gate=run.require_independent_review();configure_probe_evidence(runtime)
    run.write(run.ART/'independent_review_gate.json',gate)  # exact existing reviewer signature
    run.write(run.ART/'acceptance_gate.json',dict(status='PASS',engineering_blockers_resolved=True,diagnostic_only=True))
    original_launch=run.launch_child
    def launch(args,log,runtime,gpu=''):
        if args[0]=='train':
            command=[sys.executable,__file__,'--child','--probe-trainer','--runtime',str(runtime),'--device',device,'--steps','1000']
            if (runtime/'stages/stage1/training/checkpoints/resume_latest.pt').is_file():command.append('--resume')
            log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as f:
                child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        else:
            command=[sys.executable,__file__,'--probe-worker',args[args.index('--kinds')+1],'--runtime',str(runtime)]
            log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as f:
                child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        with (runtime/'probe_launches.jsonl').open('a') as f:f.write(json.dumps(dict(args=args,pid=child.pid,at=run.now()))+'\n')
        return child
    run.launch_child=launch
    run.sync_milestone=lambda rt,label:run.write(rt/'probe_git_sync.json',dict(diagnostic_only=True,label=label,reason='isolated probe does not publish its assets'))
    run.supervise(runtime,'0')  # actual persistent production coordinator body


def low_z_dynamic_evidence():
    rows=[]
    for alpha in (.5,.25,.125):
        cfg=run.formal.scene_config(run.formal.resolved(run.CONFIGS['stage1']),'mixed')
        set_global_config(cfg);env=run.VorAdjEnv(copy.deepcopy(cfg),seed=2026101804);env.reset()
        env.evaders[0].deactivated=True
        for o in env.obstacles:o.x=o.y=105.
        for i,p in enumerate(env.pursuers):
            p.x=30.+15*i;p.y=50.;p.theta=0.;p.speed=.5;p.deactivated=False;p.is_pursuing=False
        env._pursuing_release_counters=[0]*4;env._pursuing_flags_initialized=False
        env._invalidate_voronoi_cache();env.z_state[:]=[0,alpha,alpha/2,0];env.get_observations()
        result=env.step([4]*4,[None]);meta=result.infos[0]['replay_metadata']
        assert meta['reward_role']=='support' and meta['support_reward_blend_active']
        assert meta['support_reward_capture_weight']==alpha and meta['coverage_ce_applied_weight']==1-alpha
        assert all(i['replay_metadata']['effective_task_label']!='capture' for i in result.infos)
        assert np.isfinite(result.rewards).all()
        decay=[float(env.z_state.max())]
        for _ in range(3):
            if all(result.dones):break
            result=env.step([4]*4,[None]);decay.append(float(env.z_state.max()))
        assert decay[-1]==0
        rows.append(dict(alpha=alpha,role=meta['reward_role'],selected=meta['z_support_selection'],ce_weight=meta['coverage_ce_applied_weight'],decay=decay,no_capture_neighbors=True))
    return rows


def real_supervisor_pause_resume_probe(runtime,device):
    runtime.mkdir(parents=True)
    log=runtime/'supervisor_probe.log'
    def supervisor():
        with log.open('a') as f:return subprocess.Popen([sys.executable,__file__,'--probe-supervisor','--runtime',str(runtime),'--device',device],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
    def wait_until(predicate,process,limit=180):
        deadline=time.time()+limit
        while not predicate():
            assert process.poll() is None,log.read_text()[-4000:]
            if time.time()>deadline:raise TimeoutError('real supervisor probe deadline')
            time.sleep(.1)
    def progress():return run.read(runtime/'probe_progress.json',{})
    proc=supervisor()
    try:
        wait_until(lambda:progress().get('step',0)>=8,proc)
        run.write(runtime/'pause_request.json',dict(reason='actual supervisor owned-child pause smoke'))
        assert proc.wait(timeout=180)!=0
        first=payload(runtime);n1=first['runtime']['global_step'];u1=first['runtime']['update_steps']
        assert n1>=8 and u1>0
        launches=(runtime/'probe_launches.jsonl').read_text();time.sleep(.5)
        assert launches==(runtime/'probe_launches.jsonl').read_text()
        assert (runtime/'pause_request.json').is_file()
        # Resume exact state as a real externally started trainer, then exercise
        # the coordinator's adopting-existing process path (no duplicate launch).
        (runtime/'pause_request.json').unlink()
        with log.open('a') as f:
            trainer=subprocess.Popen([sys.executable,__file__,'--child','--probe-trainer','--resume','--runtime',str(runtime),'--device',device,'--steps','1000'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        wait_until(lambda:run.read(runtime/'stages/stage1/training_started.json',{}).get('pid')==trainer.pid,trainer)
        proc=supervisor()
        wait_until(lambda:run.read(runtime/'status.json',{}).get('phase')=='adopting_existing_training',proc)
        run.write(runtime/'pause_request.json',dict(reason='actual supervisor adopted trainer pause smoke'))
        assert proc.wait(timeout=180)!=0;assert trainer.wait(timeout=180)!=0
        second=payload(runtime);n2=second['runtime']['global_step'];u2=second['runtime']['update_steps']
        assert n2>n1 and u2>u1
        trace=[json.loads(line) for line in (runtime/'probe_launches.jsonl').read_text().splitlines()]
        assert sum(x['args'][0]=='train' for x in trace)==1,'adoption launched duplicate trainer'
        assert not (runtime/'stages/stage2/handoff.json').exists()
        for x in trace:assert not run.alive(x['pid']),x
        run.enqueue('selection','stage1',{},'durable-smoke',runtime)
        assert run.claim(('selection',),runtime) is None
        with run.db(runtime) as c:assert c.execute("SELECT status FROM jobs WHERE id='durable-smoke'").fetchone()[0]=='pending'
        return dict(actual_supervise_process=True,first_paused_step=n1,resumed_adopted_paused_step=n2,updates_before=u1,updates_after=u2,no_duplicate_train_or_worker_restart=True,no_handoff=True,pause_marker_retained=True,sqlite_pending_durable=True)
    finally:
        run.write(runtime/'pause_request.json',dict(reason='diagnostic end; keep isolated probe held'))
        if proc.poll() is None:run.wait_for_safe_pause(proc)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--child',action='store_true');ap.add_argument('--resume',action='store_true');ap.add_argument('--steps',type=int,default=64);ap.add_argument('--probe-trainer',action='store_true');ap.add_argument('--probe-supervisor',action='store_true');ap.add_argument('--probe-worker');args=ap.parse_args();torch.set_num_threads(1)
    run.require_independent_review()
    runtime=args.runtime
    if args.probe_supervisor:return supervisor_probe(runtime,args.device)
    if args.probe_worker:
        configure_probe_evidence(runtime);return run.worker(tuple(args.probe_worker.split(',')),runtime)
    if args.child:return run_child(runtime,args.device,args.resume,args.steps,args.probe_trainer)
    assert not runtime.exists(),'smoke diagnostics must use a fresh directory'
    runtime.mkdir(parents=True)
    results={}
    results['low_z_dynamic_evidence']=low_z_dynamic_evidence()
    # Actual CPU transitions and every production stage/scene live R contract.
    facts=[]
    for stage in run.old.STAGE_ORDER:
        cfg=run.formal.resolved(run.CONFIGS[stage])
        for scene in run.SCENES:
            ec=run.formal.scene_config(cfg,scene);set_global_config(ec);env=run.VorAdjEnv(copy.deepcopy(ec),seed=2026101801);env.reset();assert_runtime(env)
            r=env.step([4]*len(env.pursuers),[4]*len(env.evaders));assert np.isfinite(r.rewards).all()
            facts.append(dict(stage=stage,scene=scene,**run.map_normalization_facts(env)))
    results['cpu_runtime_facts']=facts
    # Full network CUDA optimizer updates and exact resume, compared with uninterrupted.
    assert args.device.startswith('cuda') and torch.cuda.is_available()
    base=runtime/'continuous';t=run.QueueTrainer(config(base,64,args.device));results['real_replay_fill']=fill_real_replays(t);t.train();p0=payload(base)
    assert t.update_steps>=32 and all(len(b)>0 for b in t.replays.values())
    results['cuda_training']=dict(steps=t.global_step,optimizer_updates=t.update_steps,loss_ema=t.loss_ema,q_values=t._last_q,gradient_norm=t._last_grad_norm,replay_sizes={k:len(b) for k,b in t.replays.items()},batch_counts=t.last_batch_buffer_counts,finite=True,initialization='fresh random, no historical weights')
    assert all(v>0 for v in t.last_batch_buffer_counts.values())
    del t;torch.cuda.empty_cache()
    split=runtime/'split';first=run.QueueTrainer(config(split,64,args.device));fill_real_replays(first);first.total_timesteps=32;first.train();del first;torch.cuda.empty_cache()
    resumed=config(split,64,args.device);resumed['checkpointing']['resume_path']=str(split/'stages/stage1/training/checkpoints/resume_latest.pt');second=run.QueueTrainer(resumed);second.train();p1=payload(split)
    for k in ('model','target_model','optimizer','rng','z_states'):assert digest(p0[k])==digest(p1[k]),k
    assert digest(p0['runtime']['current_observations'])==digest(p1['runtime']['current_observations'])
    results['exact_resume']='PASS: model,target,optimizer,RNG,Z,current observation identical to uninterrupted, with real updates'
    # Evaluate and render the newly trained diagnostic checkpoint, never old weights.
    checkpoint=second.ckpt_dir/'step_64.pt';sd=split/'stages/stage1'
    (sd/'resolved.yaml').write_text(run.yaml.safe_dump(run.formal.resolved(run.CONFIGS['stage1'])))
    run.write(sd/'selection_report.json',dict(selected=dict(checkpoint=str(checkpoint),checkpoint_sha256=run.sha256_file(checkpoint),step=64)))
    del second;torch.cuda.empty_cache()
    for mode in ('argmax','sample'):run.formal_batch('stage1',mode,split,episodes=1,smoke_steps=4)
    started=time.time();run.render('stage1',dict(mode='argmax',scene='mixed',episode_index=0),split)
    gifs=list((sd/'independent/argmax/gifs').rglob('*.gif'));assert gifs
    from PIL import Image
    with Image.open(gifs[0]) as im:assert im.n_frames>0
    results['evaluation_renderer']=dict(episodes=6,horizon=4,gifs=len(gifs),render_service_seconds=time.time()-started,checkpoint_sha256=run.sha256_file(checkpoint))
    # Actual Supervisor process/real workers/owned trainer + exact resume/adoption.
    # The permanent regressions execute the actual supervise body at selection,
    # handoff, scientific adoption and tail states with explicit process doubles.
    control_log=runtime/'supervisor_control_path_tests.txt'
    with control_log.open('w') as f:
        result=subprocess.run([sys.executable,'-m','pytest','tests/test_relative_z_supervisor_20261017.py','-q','--tb=short'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'CUDA_VISIBLE_DEVICES':''})
    assert result.returncode==0,control_log.read_text()[-5000:]
    results['supervisor_control_paths']=dict(status='PASS',log=str(control_log),actual_supervise_body=True,selection_handoff_tail_use_explicit_process_doubles=True)
    results['supervisor_pause_resume_queue']=real_supervisor_pause_resume_probe(runtime/'real_supervisor',args.device)
    run.write(runtime/'report.json',dict(status='PASS',diagnostic_only=True,results=results,created_at=run.now()))
    run.write(run.ART/'smoke_test_results.json',dict(status='PASS',runtime=str(runtime),results=results,created_at=run.now()))
    print(json.dumps(dict(status='PASS',runtime=str(runtime),results=results),ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
