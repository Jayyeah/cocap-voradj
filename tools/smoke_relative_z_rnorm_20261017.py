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


def run_child(runtime,device,resume=False,steps=64):
    cfg=config(runtime,steps,device)
    if resume:cfg['checkpointing']['resume_path']=str(runtime/'stages/stage1/training/checkpoints/resume_latest.pt')
    t=run.QueueTrainer(cfg)
    if not resume:fill_real_replays(t)
    t.train()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--child',action='store_true');ap.add_argument('--resume',action='store_true');ap.add_argument('--steps',type=int,default=64);args=ap.parse_args();torch.set_num_threads(1)
    run.require_independent_review()
    runtime=args.runtime
    if args.child:return run_child(runtime,args.device,args.resume,args.steps)
    assert not runtime.exists(),'smoke diagnostics must use a fresh directory'
    runtime.mkdir(parents=True)
    results={}
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
    # Actual process boundary pause: original supervisor helper waits for the
    # child to publish exact state, blocks claims, then resumes the saved child.
    pause=runtime/'pause_controller';log=pause/'controller.log';pause.mkdir()
    def child(resume=False):
        command=[sys.executable,__file__,'--child','--runtime',str(pause),'--device',args.device,'--steps','128']
        if resume:command.append('--resume')
        with log.open('a') as f:return subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
    proc=child();deadline=time.time()+180
    resume_path=pause/'stages/stage1/training/checkpoints/resume_latest.pt'
    while not resume_path.exists():
        assert proc.poll() is None,log.read_text()[-3000:]
        if time.time()>deadline:raise TimeoutError('smoke child did not publish checkpoint')
        time.sleep(.2)
    run.write(pause/'pause_request.json',dict(reason='authorized supervisor smoke'))
    run.wait_for_safe_pause(proc)
    assert proc.returncode!=0
    p=payload(pause);assert 0<p['runtime']['global_step']<128
    paused_step=p['runtime']['global_step']
    run.enqueue('selection','stage1',{},'durable-smoke',pause)
    assert run.claim(('selection',),pause) is None
    (pause/'pause_request.json').unlink();proc=child(True);code=proc.wait(timeout=180);assert code==0,log.read_text()[-3000:]
    assert payload(pause)['runtime']['global_step']==128
    claimed=run.claim(('selection',),pause);assert claimed['id']=='durable-smoke'
    with run.db(pause) as c:c.execute("UPDATE jobs SET status='complete' WHERE id='durable-smoke'")
    results['supervisor_pause_resume_queue']=dict(paused_step=paused_step,resumed_step=128,durable_claim_hold=True,exact_resume=True)
    run.write(runtime/'report.json',dict(status='PASS',diagnostic_only=True,results=results,created_at=run.now()))
    run.write(run.ART/'smoke_test_results.json',dict(status='PASS',runtime=str(runtime),results=results,created_at=run.now()))
    print(json.dumps(dict(status='PASS',runtime=str(runtime),results=results),ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
