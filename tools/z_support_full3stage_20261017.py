#!/usr/bin/env python3
"""Persistent Treatment curriculum, independent CPU queues and artifact ledger.

The production trainer and historical selection/evaluation physics are reused.
No evaluation or rendering is called by the training process.
"""
from __future__ import annotations
import argparse, copy, fcntl, gzip, hashlib, json, math, os, random, shutil, shlex
import sqlite3, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
import yaml
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.trainer import CoCapTrainer
from cocap_voradj.training.runtime_semantics import assert_runtime
from tools import iqn_token_matched_20260919 as formal
from tools import iqn_z_unified_decay_curriculum_20260919 as old
from tools import evaluate_iqn_z05_independent_20260923 as independent
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file, stack_obs
from tools.rollout_voradj_visual import snapshot_env, render_gif

RUNTIME=Path('/home/yjq/rl/CoCap1/iqn-z-guided-support-full3stage-20261017-runtime')
ART=ROOT/'artifacts/2026-10-17_z_support_full3stage'
DOC=ROOT/'docs/experiments/z_support_full3stage_20261017'
CONFIG_DIR=ROOT/'configs/experiments/iqn_z_guided_support_full3stage_20261017'
CONFIGS={s:CONFIG_DIR/f'{name}.yaml' for s,name in zip(old.STAGE_ORDER,('stage1_4p1e1obs_2m','stage2_8p2e2obs_700k','stage3_12p3e3obs_700k'))}
HIST_TRAIN=Path('/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages')
BUDGETS={'stage1':2000000,'stage2':700000,'stage3':700000}
SCENES=('coverage','capture','mixed')
# Freeze before training: same historical selection family; disjoint final family.
SELECTION_SEED=2026091900
FINAL_SEED=2026102700
MAX_ATTEMPTS=3

def now():return old.now_local()
def write(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(formal.json_finite(payload),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)
def read(path,default=None):return json.loads(Path(path).read_text()) if Path(path).is_file() else default
def head():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def process_start(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,TypeError,ValueError,IndexError):return None

def alive(pid,start=None):
    try:
        os.kill(int(pid),0)
        return start is None or str(start)==process_start(pid)
    except (OSError,TypeError,ValueError):return False

def db(runtime=RUNTIME):
    runtime.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(runtime/'queue.sqlite3',timeout=30)
    c.row_factory=sqlite3.Row
    c.execute('PRAGMA journal_mode=WAL');c.execute('PRAGMA synchronous=FULL')
    c.executescript('''
    CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,kind TEXT,stage TEXT,payload TEXT,status TEXT DEFAULT 'pending',attempts INTEGER DEFAULT 0,worker_pid INTEGER,created_at TEXT,updated_at TEXT,error TEXT);
    CREATE TABLE IF NOT EXISTS checkpoints(path TEXT PRIMARY KEY,stage TEXT,step INTEGER,sha256 TEXT,bytes INTEGER,updates INTEGER,episode INTEGER,config_sha256 TEXT,created_at TEXT);
    CREATE TABLE IF NOT EXISTS artifacts(path TEXT PRIMARY KEY,stage TEXT,kind TEXT,sha256 TEXT,bytes INTEGER,metadata TEXT,created_at TEXT);
    CREATE TABLE IF NOT EXISTS failures(id INTEGER PRIMARY KEY,job_id TEXT,attempt INTEGER,error TEXT,created_at TEXT);
    ''')
    if 'worker_start' not in {r[1] for r in c.execute('PRAGMA table_info(jobs)')}:
        c.execute('ALTER TABLE jobs ADD COLUMN worker_start TEXT');c.commit()
    return c

def enqueue(kind,stage,payload,key,runtime=RUNTIME):
    with db(runtime) as c:
        c.execute('INSERT OR IGNORE INTO jobs(id,kind,stage,payload,created_at,updated_at) VALUES(?,?,?,?,?,?)',(key,kind,stage,json.dumps(payload),now(),now()))

def register(path,stage,kind,metadata=None,runtime=RUNTIME):
    path=Path(path)
    with db(runtime) as c:
        c.execute('INSERT OR REPLACE INTO artifacts VALUES(?,?,?,?,?,?,?)',(str(path),stage,kind,sha256_file(path),path.stat().st_size,json.dumps(dict(source_head=head(),command=shlex.join([sys.executable,*sys.argv]),**(metadata or {})),ensure_ascii=False),now()))

def claim(kinds,runtime=RUNTIME):
    c=db(runtime)
    try:
        c.execute('BEGIN IMMEDIATE')
        for row in c.execute("SELECT * FROM jobs WHERE status='running'").fetchall():
            if not alive(row['worker_pid'],row['worker_start']):
                status='failed' if row['attempts']>=MAX_ATTEMPTS else 'pending'
                c.execute('UPDATE jobs SET status=?,error=?,updated_at=? WHERE id=?',(status,'worker exited before commit; recovered',now(),row['id']))
                c.execute('INSERT INTO failures(job_id,attempt,error,created_at) VALUES(?,?,?,?)',(row['id'],row['attempts'],'worker exited before commit; recovered',now()))
        marks=','.join('?' for _ in kinds)
        row=c.execute(f"SELECT * FROM jobs WHERE status='pending' AND kind IN ({marks}) ORDER BY CASE kind WHEN 'selection' THEN 0 WHEN 'formal' THEN 1 WHEN 'reference' THEN 2 ELSE 3 END,created_at,id LIMIT 1",kinds).fetchone()
        if row:
            c.execute("UPDATE jobs SET status='running',worker_pid=?,worker_start=?,attempts=attempts+1,updated_at=? WHERE id=?",(os.getpid(),process_start(os.getpid()),now(),row['id']))
        c.commit()
        return dict(row) if row else None
    finally:c.close()

def config(stage,warm=None,runtime=RUNTIME,device='cuda:0'):
    cfg=formal.resolved(CONFIGS[stage]);old._assert_production_contract(cfg,.5)
    cfg.update(output_root=str(runtime/'stages'/stage),run_name='training',device=device)
    if stage=='stage1':
        assert not (cfg.get('pretrained',{}) or {}).get('path'),'scratch cannot load pretrained policy'
    else:
        if warm is None or not Path(warm).is_file():raise ValueError('selected preceding stage model is required')
        cfg['pretrained']={'path':str(warm),'compatibility_mode':'shape_compatible'}
    return cfg

def finite_grad(grad):
    if not torch.isfinite(grad).all():raise FloatingPointError('non-finite IQN gradient before optimizer.step')
    return grad

def finite_optimizer_step(optimizer,args,kwargs):
    grads=[p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
    if grads and not torch.isfinite(torch.stack(torch._foreach_norm(grads))).all():
        raise FloatingPointError('non-finite IQN gradient before optimizer.step')

class QueueTrainer(CoCapTrainer):
    """Only operational snapshot/telemetry changes; same transition/update loop."""
    def __init__(self,cfg):
        super().__init__(cfg)
        self.optimizer.register_step_pre_hook(finite_optimizer_step)
        self._telemetry_start_time=time.time();self._telemetry_start_step=self.global_step

    def _select_actions(self,task,obs_list):
        runtime=Path(self.config['output_root']).parents[1]
        if (runtime/'pause_request.json').is_file():
            if self.global_step and getattr(self,'_train_runtime_ready',False):self._save_full_resume()
            raise RuntimeError('scientific coordinator requested safe pause; exact decision state preserved')
        # At the next decision boundary the previous loop has committed every
        # current_* field. Never save from _update, which precedes that commit.
        if self.global_step==1 or (self.global_step%25000==0 and self.global_step%100000!=0):
            if getattr(self,'_train_runtime_ready',False):self._save_full_resume()
        return super()._select_actions(task,obs_list)

    def _save_full_resume(self):
        # Atomic replacement temporarily needs both old and new payloads.
        estimate=max(8*1024**3,int(self.full_resume_path.stat().st_size*1.5) if self.full_resume_path.is_file() else 0)
        if shutil.disk_usage(self.run_dir).free<20*1024**3+estimate:
            raise OSError('resource safety: insufficient space for atomic full resume plus 20 GiB reserve')
        path=super()._save_full_resume()
        runtime=Path(self.config['output_root']).parents[1]
        stage=Path(self.config['output_root']).name
        register(path,stage,'full_resume_latest',dict(step=self.global_step,updates=self.update_steps,retention='latest_only'),runtime)
        return path

    def _reset_task(self,task):
        obs=super()._reset_task(task);assert_runtime(self.envs[task]);return obs

    def _distributional_iqn_td_loss(self,batch,task):
        if not torch.isfinite(batch['rewards']).all():raise FloatingPointError('non-finite sampled reward')
        loss=super()._distributional_iqn_td_loss(batch,task)
        if not torch.isfinite(loss):raise FloatingPointError('non-finite IQN loss before backward')
        return loss

    def _update(self,task):
        # These are the transitions just generated, before any learning update.
        for b in self.replays.values():
            if b.data and not math.isfinite(b.data[-1][2]):raise FloatingPointError('non-finite environment reward')
        value=super()._update(task)
        if self.global_step%1000==0:
            if shutil.disk_usage(self.run_dir).free<20*1024**3:raise OSError('resource safety: less than 20 GiB free')
            mem_available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
            if mem_available<8*1024**3:raise OSError('resource safety: less than 8 GiB available RAM')
            write(self.run_dir/'progress.json',dict(steps_per_second=(self.global_step-self._telemetry_start_step)/max(time.time()-self._telemetry_start_time,1e-6),eta_seconds=(self.total_timesteps-self.global_step)/max((self.global_step-self._telemetry_start_step)/max(time.time()-self._telemetry_start_time,1e-6),1e-6),pid=os.getpid(),environment_decision_steps=self.global_step,optimizer_updates=self.update_steps,episodes=self.episode_idx,loss_ema=self.loss_ema,replay_sizes={k:len(v) for k,v in self.replays.items()},episode_step=self.envs[task].episode_step,updated_at=now()))
        return value

    def _save_checkpoint(self,name):
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):raise FloatingPointError('non-finite model snapshot')
        path=self.ckpt_dir/name
        extra=dict(global_step=self.global_step,episode=self.episode_idx,train_mode=self.train_mode,coverage_ce_speed_weight=self.current_coverage_ce_speed_weight,optimizer_updates=self.update_steps)
        if path.exists():
            existing=torch.load(path,map_location='cpu',weights_only=False)
            if existing['extra']['global_step']!=self.global_step or any(not torch.equal(v.cpu(),existing['state_dict'][k]) for k,v in self.model.state_dict().items()):
                raise RuntimeError(f'refusing to replace immutable snapshot: {path}')
        else:
            tmp=path.with_name(f'.{name}.tmp-{os.getpid()}')
            self.model.save(str(tmp),extra=extra);tmp.chmod(0o444);tmp.replace(path)
        if self.full_resume_enabled and getattr(self,'_train_runtime_ready',False):self._save_full_resume()
        runtime=Path(self.config['output_root']).parents[1]
        stage=Path(self.config['output_root']).name
        register(path,stage,'model_checkpoint',dict(step=self.global_step,updates=self.update_steps),runtime)
        if name.startswith('step_'):
            register_checkpoint(path,stage,self.global_step,self.update_steps,self.episode_idx,old.stable_hash(self.config),runtime)
        write(self.run_dir/'snapshot_status.json',dict(path=str(path),step=self.global_step,updates=self.update_steps,full_resume=str(self.full_resume_path),updated_at=now()))
        return path

def register_checkpoint(path,stage,step,updates,episode,cfg_hash,runtime=RUNTIME):
    path=Path(path);sha=sha256_file(path)
    payload=dict(checkpoint=str(path),sha256=sha,config=str(runtime/'stages'/stage/'resolved.yaml'),step=step)
    with db(runtime) as c:
        prior=c.execute('SELECT sha256 FROM checkpoints WHERE path=?',(str(path),)).fetchone()
        if prior and prior['sha256']!=sha:raise RuntimeError('registered immutable checkpoint was modified')
        c.execute('INSERT OR IGNORE INTO checkpoints VALUES(?,?,?,?,?,?,?,?,?)',(str(path),stage,step,sha,path.stat().st_size,updates,episode,cfg_hash,now()))
        c.execute('INSERT OR IGNORE INTO jobs(id,kind,stage,payload,created_at,updated_at) VALUES(?,?,?,?,?,?)',(f'selection/{stage}/{step:09d}','selection',stage,json.dumps(payload),now(),now()))

def train_stage(stage,runtime,device):
    training_lock=(runtime/'training.lock').open('w')
    try:fcntl.flock(training_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('an identical training process is already alive; refusing duplicate')
    stage_dir=runtime/'stages'/stage
    handoff=read(stage_dir/'handoff.json',{})
    warm=handoff.get('checkpoint')
    if warm and sha256_file(Path(warm))!=handoff['sha256']:raise RuntimeError('warm-start handoff hash mismatch')
    cfg=config(stage,warm,runtime,device)
    resume=stage_dir/'training/checkpoints/resume_latest.pt'
    if resume.is_file():cfg['checkpointing']['resume_path']=str(resume)
    if (stage_dir/'training_started.json').exists() and not resume.is_file():raise RuntimeError('previous training started but exact resume missing; refusing fresh restart')
    t=QueueTrainer(cfg)
    # Recover a crash between atomic resume publication and queue registration.
    for snapshot in sorted((stage_dir/'training/checkpoints').glob('step_*.pt')):
        payload=torch.load(snapshot,map_location='cpu',weights_only=False)
        extra=payload['extra'];step=int(extra['global_step'])
        if step<=t.global_step:
            register_checkpoint(snapshot,stage,step,int(extra['optimizer_updates']),int(extra['episode']),old.stable_hash(cfg),runtime)
    if t.global_step==0:
        if warm:
            expected=torch.load(warm,map_location='cpu',weights_only=False)['state_dict']
            assert set(expected)==set(t.model.state_dict())
            for key,value in t.model.state_dict().items():
                assert torch.equal(value.detach().cpu(),expected[key]),f'partial warm-start: {key}'
        gate=old._fresh_stage_gate(t,stage,Path(warm) if warm else None)
        gate.update(model_initial_sha256=formal.state_hash(t.model),seed=cfg['seed'],source_head=head())
        write(stage_dir/'fresh_stage_gate.json',gate)
    write(stage_dir/'training_started.json',dict(pid=os.getpid(),process_start=process_start(os.getpid()),stage=stage,head=head(),config_hash=old.stable_hash(cfg),start_step=t.global_step,updated_at=now()))
    started=time.time();start=t.global_step
    checkpoint=t.train()
    write(stage_dir/'training_complete.json',dict(stage=stage,environment_decision_steps=t.global_step,optimizer_updates=t.update_steps,episodes=t.episode_idx,final_checkpoint=str(checkpoint),final_sha256=sha256_file(checkpoint),source_head=head(),elapsed_segment_seconds=time.time()-started,segment_start_step=start,completed_at=now()))

class NativeSamplePolicy:
    """Only the secondary IQN inference path differs; same episode runner."""
    def __init__(self,model):self.model=model;self.config=model.config
    def __getattr__(self,name):return getattr(self.model,name)

@torch.no_grad()
def sample_q(model,local,device):
    q=model.model(stack_obs(local,device),num_tau=32,mode='voradj')['q_values'].mean(dim=1)
    return q.cpu().numpy().astype(np.float32),q.argmax(dim=-1).cpu().numpy()

# Native q_values uses [batch, quantiles, actions], matching model.act.
def formal_batch(stage,mode,runtime,episodes=20,smoke_steps=None,reference=False):
    assert mode in ('argmax','sample')
    stage_dir=runtime/'stages'/stage
    selected=read((HIST_TRAIN if reference else runtime/'stages')/stage/'selection_report.json')['selected']
    checkpoint=Path(selected['checkpoint'])
    if sha256_file(checkpoint)!=selected['checkpoint_sha256']:raise RuntimeError('selected model changed')
    cfg_path=stage_dir/('reference_resolved.yaml' if reference else 'resolved.yaml')
    cfg=formal.resolved(cfg_path);model=CoCapIQN.load(str(checkpoint),device='cpu').eval()
    output=stage_dir/('reference' if reference else 'independent')/mode;output.mkdir(parents=True,exist_ok=True)
    rows=[];capacity=None
    for si,scene in enumerate(SCENES):
        for ep in range(episodes):
            seed=FINAL_SEED+si*100000+ep
            dest=output/'episodes'/f'{scene}_{ep:03d}.json'
            if dest.is_file():
                cached=read(dest)
                assert cached['checkpoint_sha256']==selected['checkpoint_sha256'] and cached['seed']==seed
                for name in ('trajectory','events','token_audit'):
                    assert sha256_file(Path(cached[name]))==cached[name+'_sha256'],'cached artifact missing or changed'
                with gzip.open(cached['token_audit'],'rt') as h:saved_audit=json.load(h)
                if capacity is None:
                    capacity=independent.CapacityAudit.__new__(independent.CapacityAudit)
                    capacity.samples=[];capacity.episodes=[];capacity._episode_samples=[]
                    capacity.env_caps=saved_audit['env_caps'];capacity.model_caps=saved_audit['model_caps']
                capacity.samples.extend(saved_audit['samples']);capacity.episodes.extend(saved_audit['episodes'])
                rows.append(cached);continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            holder={};frames=[];events=[];audit=None
            def factory(s,sd):
                nonlocal audit
                env,obs=independent.make_env_factory(cfg,s,sd,'native',model,holder)
                audit=independent.CapacityAudit(env,model);audit.start_episode()
                return env,obs
            def snapshot(env,step):
                audit.observe(env)
                phase='coverage' if scene=='coverage' or (scene=='mixed' and env.evaders and all(e.deactivated and not e.collision for e in env.evaders)) else 'capture'
                frames.append(snapshot_env(env,scene,phase,int(env.post_capture_step if phase=='coverage' else 0),step,step))
                events.append(dict(step=step,capture=copy.deepcopy(env.last_capture_events),collision=copy.deepcopy(env.last_collision_events),z=env.z_state.tolist(),active_pursuers=sum(not p.deactivated for p in env.pursuers)))
            diag=formal.ZDiagnostics(scene)
            def progress(step):write(output/'progress.json',dict(scene=scene,episode_index=ep,episode_step=step,completed=len(rows),total=episodes*3,updated_at=now()))
            from tools import run_forward_final_bridge_20260908 as runner
            original=runner.fixed_midpoint_q
            if mode=='sample':runner.fixed_midpoint_q=lambda m,local,device:sample_q(NativeSamplePolicy(m),local,device)
            try:
                row=runner.run_episode(model,scene,seed,'cpu',on_snapshot=snapshot,on_transition=diag.transition,on_progress=progress,env_factory=factory,max_steps=smoke_steps)
            finally:runner.fixed_midpoint_q=original
            env=holder['env'];independent.add_episode_metrics(row,env,diag,scene)
            row.update(normal_capture_definition='historical: captured and at least one loose event',all_targets_normal=bool(row['captured'] and row['capture_types'] and all(t=='loose' for t in row['capture_types'])),mode=mode,resolver=holder['resolver'],checkpoint_sha256=selected['checkpoint_sha256'],alive_agents=sum(not p.deactivated for p in env.pursuers))
            row['token_occupancy_episode']=audit.finish_episode()
            if capacity is None:capacity=audit
            else:capacity.samples.extend(audit.samples);capacity.episodes.extend(audit.episodes)
            trajectory=output/'trajectories'/f'{scene}_{ep:03d}.json.gz';trajectory.parent.mkdir(parents=True,exist_ok=True)
            tmp=trajectory.with_suffix('.tmp')
            with gzip.open(tmp,'wt') as h:json.dump({'scene':scene,'seed':seed,'frames':frames},h,allow_nan=False)
            tmp.replace(trajectory)
            event_path=output/'events'/f'{scene}_{ep:03d}.json';write(event_path,events)
            token_path=output/'token_audits'/f'{scene}_{ep:03d}.json.gz';token_path.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(token_path,'wt') as h:json.dump(dict(samples=audit.samples,episodes=audit.episodes,env_caps=audit.env_caps,model_caps=audit.model_caps),h)
            row.update(token_audit=str(token_path),token_audit_sha256=sha256_file(token_path))
            row.update(trajectory=str(trajectory),trajectory_sha256=sha256_file(trajectory),events=str(event_path),events_sha256=sha256_file(event_path))
            write(dest,row);rows.append(row)
            for path,kind in ((trajectory,'trajectory'),(event_path,'event_log'),(token_path,'token_audit'),(dest,'episode')):register(path,stage,kind,dict(mode=mode,scene=scene,seed=seed),runtime)
    # summarize_records uses the frozen historical statistics, including all failures.
    if 'env' not in locals():
        holder={};env,_=independent.make_env_factory(cfg,'mixed',FINAL_SEED+200000,'native',model,holder)
    if capacity is None:capacity=independent.CapacityAudit(env,model)
    summary=independent.summarize_records(rows,capacity)
    summary['token_occupancy']=capacity.summary(env)
    for scene in SCENES:
        summary[scene]['alive_agents']=formal.numeric_summary([r['alive_agents'] for r in rows if r['scene']==scene])
        if scene!='coverage':summary[scene]['all_targets_normal_rate']=sum(r['all_targets_normal'] for r in rows if r['scene']==scene)/episodes
    report=dict(status='complete',stage=stage,method='historical_z05_selected' if reference else 'z_guided_support',mode=mode,primary=mode=='argmax',episodes_per_scene=episodes,seed_base=FINAL_SEED,checkpoint=str(checkpoint),checkpoint_sha256=selected['checkpoint_sha256'],config=str(cfg_path),config_sha256=sha256_file(cfg_path),summary=summary,records=rows,smoke_steps=smoke_steps,source_head=head(),completed_at=now())
    write(output/'report.json',report)
    (output/'episodes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    register(output/'report.json',stage,'formal_report',dict(mode=mode),runtime)
    counterpart=read(stage_dir/('independent' if reference else 'reference')/mode/'report.json')
    if counterpart:
        pairs={(r['scene'],r['seed']):r['initial_state_fingerprint'] for r in counterpart['records']}
        assert pairs=={(r['scene'],r['seed']):r['initial_state_fingerprint'] for r in rows},'reference/treatment evaluation initial-state mismatch'
        write(stage_dir/'evaluation_pairing.json',dict(status='PASS',pairs=len(rows),mode=mode,training_seed_matched=False,completed_at=now()))
    if smoke_steps is None and not reference:
        # Fixed even indices, defined before outcomes. Includes failures without filtering.
        for scene in SCENES:
            for ep in range(0,episodes,2):enqueue('render',stage,dict(mode=mode,scene=scene,episode_index=ep),f'render/{stage}/{mode}/{scene}/{ep:03d}',runtime)
    return report

def render(stage,payload,runtime):
    output=runtime/'stages'/stage/'independent'/payload['mode'];scene=payload['scene'];ep=payload['episode_index']
    row=read(output/'episodes'/f'{scene}_{ep:03d}.json')
    trajectory=Path(row['trajectory'])
    assert sha256_file(trajectory)==row['trajectory_sha256']
    with gzip.open(trajectory,'rt') as h:frames=json.load(h)['frames']
    gif=output/'gifs'/scene/f'rollout_{scene}_seed_{row["seed"]}.gif'
    record=render_gif(frames,gif,300,70,100,scene,False,False,False,scene=='coverage')
    from PIL import Image
    with Image.open(gif) as im:im.seek(im.n_frames-1);im.load()
    meta=dict(**payload,seed=row['seed'],checkpoint_sha256=row['checkpoint_sha256'],trajectory_sha256=row['trajectory_sha256'],selection_reason='pre-registered even episode index from all 20; outcome-independent',renderer=dict(max_frames=300,trail_window=70,frame_duration_ms=100,draw_neighbor_edges=False,draw_trails=False,draw_sensing_circles=False,draw_ce_targets=scene=='coverage'),render_record=record)
    write(gif.with_suffix('.json'),meta);register(gif,stage,'gif',meta,runtime);register(gif.with_suffix('.json'),stage,'renderer_manifest',meta,runtime)

def job(job_id,runtime):
    with db(runtime) as c:r=dict(c.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone())
    # The actual job PID owns the lease, so it survives its worker exiting.
    with db(runtime) as c:c.execute('UPDATE jobs SET worker_pid=?,worker_start=?,updated_at=? WHERE id=?',(os.getpid(),process_start(os.getpid()),now(),job_id))
    p=json.loads(r['payload']);stage=r['stage']
    if r['kind']=='selection':
        checkpoint=Path(p['checkpoint'])
        if sha256_file(checkpoint)!=p['sha256']:raise RuntimeError('immutable checkpoint SHA changed')
        output=runtime/'stages'/stage/'evaluations'/f'step_{p["step"]:09d}'
        existing=read(output/'report.json',{})
        if existing.get('status')=='complete' and existing.get('checkpoint_sha256')==p['sha256']:
            report=existing
        else:
            report=formal.evaluate_checkpoint(checkpoint,output,'cpu',20,SELECTION_SEED,'z',config_path=Path(p['config']))
        assert report['episodes_per_scene']==20 and report['max_steps_smoke_only'] is None
        register(output/'report.json',stage,'selection_report',dict(step=p['step']),runtime)
    elif r['kind'] in ('formal','reference'):formal_batch(stage,p['mode'],runtime,reference=r['kind']=='reference')
    elif r['kind']=='render':render(stage,p,runtime)
    else:raise ValueError(r['kind'])
    with db(runtime) as c:c.execute("UPDATE jobs SET status='complete',error=NULL,updated_at=? WHERE id=?",(now(),job_id))

def worker(kinds,runtime):
    torch.set_num_threads(1)
    while True:
        r=claim(kinds,runtime)
        if not r:
            if read(runtime/'orchestration_complete.json'):
                return
            time.sleep(5);continue
        log=runtime/'logs'/(r['id'].replace('/','_')+f'_attempt{r["attempts"]+1}.log');log.parent.mkdir(parents=True,exist_ok=True)
        try:
            with log.open('a') as h:
                result=subprocess.run([sys.executable,__file__,'job','--runtime',str(runtime),'--job-id',r['id']],stdout=h,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
            if result.returncode:
                details=log.read_text()[-20000:]
                fatal=any(x in details for x in ('initial-state mismatch','immutable checkpoint SHA changed','selected model changed','runtime policy drift','policy/environment friend token cap mismatch','non-finite'))
                if fatal:
                    evidence=dict(job=r['id'],log=str(log),error=details[-5000:],updated_at=now())
                    write(runtime/'scientific_failure.json',evidence);write(runtime/'pause_request.json',evidence)
                raise RuntimeError(f'job exited {result.returncode}; log={log}')
            with db(runtime) as c:c.execute("UPDATE jobs SET status='complete',error=NULL,updated_at=? WHERE id=?",(now(),r['id']))
        except Exception:
            error=traceback.format_exc()
            with db(runtime) as c:
                attempts=c.execute('SELECT attempts FROM jobs WHERE id=?',(r['id'],)).fetchone()[0]
                c.execute('UPDATE jobs SET status=?,error=?,updated_at=? WHERE id=?',('failed' if attempts>=MAX_ATTEMPTS else 'pending',error,now(),r['id']))
                c.execute('INSERT INTO failures(job_id,attempt,error,created_at) VALUES(?,?,?,?)',(r['id'],attempts,error,now()))
        export(runtime)

def prepare(runtime):
    for stage in old.STAGE_ORDER:
        cfg=formal.resolved(CONFIGS[stage]);baseline=old.resolved('z05',stage)
        differences=formal._diff_rows(baseline,cfg)
        allowed={'seed','run_name','voradj.z_guided_support_enabled','experiment_metadata.series_label'}
        assert all(d['path'] in allowed for d in differences),differences
        assert cfg['total_timesteps']==BUDGETS[stage]
        old._assert_production_contract(cfg,.5)
        path=runtime/'stages'/stage;path.mkdir(parents=True,exist_ok=True)
        dest=path/'resolved.yaml'
        text=yaml.safe_dump(cfg,sort_keys=False)
        if dest.exists() and dest.read_text()!=text:raise RuntimeError('frozen config changed')
        dest.write_text(text)
        reference=path/'reference_resolved.yaml'
        reference_text=yaml.safe_dump(baseline,sort_keys=False)
        if reference.exists() and reference.read_text()!=reference_text:raise RuntimeError('frozen reference config changed')
        reference.write_text(reference_text)
        write(ART/f'{stage}_contract_diff.json',dict(stage=stage,differences=differences,allowed=sorted(allowed),config_sha256=sha256_file(dest),selection_seed=SELECTION_SEED,final_seed=FINAL_SEED))
    write(runtime/'execution_contract.json',dict(budgets=BUDGETS,seeds=[2026101701,2026101702,2026101703],selection_seed=SELECTION_SEED,final_seed=FINAL_SEED,primary='argmax fixed midpoint32',secondary='sample uniform32 IQN mean-Q argmax',source_head=head(),runtime=str(runtime),full_resume_retention='latest_only_per_stage',checkpoint_interval=100000,formal_episodes_per_scene_mode=20,gifs_per_scene_mode=10,max_attempts=MAX_ATTEMPTS))
    db(runtime).close();export(runtime)

def export(runtime):
    ART.mkdir(parents=True,exist_ok=True)
    with db(runtime) as c:
        tables={t:[dict(r) for r in c.execute(f'SELECT * FROM {t}').fetchall()] for t in ('jobs','checkpoints','artifacts','failures')}
    write(ART/'checkpoint_registry.json',tables['checkpoints'])
    write(ART/'evaluation_queue.json',tables['jobs'])
    write(ART/'artifact_registry.json',tables['artifacts'])
    write(ART/'failure_retry_registry.json',tables['failures'])
    export_curves(runtime)
    write(ART/'training_state_manifest.json',dict(status=read(runtime/'status.json',{'status':'prepared'}),stages={s:dict(progress=read(runtime/'stages'/s/'training/progress.json'),training=read(runtime/'stages'/s/'training_complete.json'),selection=read(runtime/'stages'/s/'selection_report.json'),fresh_gate=read(runtime/'stages'/s/'fresh_stage_gate.json')) for s in old.STAGE_ORDER},updated_at=now(),runtime=str(runtime)))

def export_curves(runtime):
    import csv
    rows=[]
    for stage in old.STAGE_ORDER:
        path=runtime/'stages'/stage/'training/metrics.jsonl'
        if not path.is_file():continue
        for line in path.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            rows.append(dict(stage=stage,**{k:r.get(k) for k in ('global_step','update_steps','loss_ema','epsilon','learning_rate','recent_capture_rate','recent_collision_rate','recent_post_capture_coverage_rate')}))
    if rows:
        dest=ART/'training_curves.csv';tmp=dest.with_name(f'.{dest.name}.tmp-{os.getpid()}')
        with tmp.open('w') as h:
            writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        tmp.replace(dest)

def plot_curves(runtime):
    import csv
    path=ART/'training_curves.csv'
    if not path.is_file():return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader(path.open()));fig,axs=plt.subplots(1,2,figsize=(10,3))
    for stage in old.STAGE_ORDER:
        rr=[r for r in rows if r['stage']==stage]
        if not rr:continue
        x=[int(r['global_step']) for r in rr]
        for ax,key in zip(axs,('loss_ema','update_steps')):
            pairs=[(a,float(r[key])) for a,r in zip(x,rr) if r[key] not in ('','None')]
            if pairs:ax.plot(*zip(*pairs),label=stage)
            ax.set_xlabel('Environment decision steps');ax.set_ylabel(key);ax.legend()
    fig.tight_layout();fig.savefig(ART/'training_curves.png',dpi=130);plt.close(fig)

def launch_child(args,log,runtime,gpu=''):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as h:
        return subprocess.Popen([sys.executable,__file__,*args,'--runtime',str(runtime)],cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MPLBACKEND':'Agg'},start_new_session=False)

def supervise(runtime,gpu):
    runtime.mkdir(parents=True,exist_ok=True)
    lock=(runtime/'supervisor.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('an identical curriculum supervisor is already running')
    acceptance=read(ART/'acceptance_gate.json',{})
    if acceptance.get('status') not in ('PASS','CONDITIONAL_PASS') or not acceptance.get('engineering_blockers_resolved'):raise RuntimeError('acceptance gate is not signed')
    if (runtime/'scientific_failure.json').is_file():raise RuntimeError('unresolved scientific failure marker; refusing automatic restart')
    prepare(runtime)
    workers=[]
    for idx,kinds in enumerate(('selection','selection','formal,reference,render')):
        workers.append((kinds,launch_child(['worker','--kinds',kinds],runtime/'logs'/f'worker_{idx}.log',runtime)))
    try:
        for stage in old.STAGE_ORDER:
            stage_dir=runtime/'stages'/stage
            prior=read(stage_dir/'training_started.json',{})
            while alive(prior.get('pid'),prior.get('process_start')) and not (stage_dir/'training_complete.json').is_file():
                write(runtime/'status.json',dict(status='running',phase='adopting_existing_training',stage=stage,training_pid=prior['pid'],updated_at=now()))
                export(runtime);reports(runtime);time.sleep(20)
            if not (stage_dir/'training_complete.json').is_file():
                for attempt in range(1,MAX_ATTEMPTS+1):
                    # GPU is shared; only inspect and wait for sufficient free memory.
                    free=int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    if free<6000 or shutil.disk_usage(runtime).free<30*1024**3:raise OSError('resource safety: insufficient GPU or disk headroom')
                    write(runtime/'status.json',dict(status='running',phase='training',stage=stage,supervisor_pid=os.getpid(),attempt=attempt,updated_at=now()))
                    child=launch_child(['train','--stage',stage,'--device','cuda:0'],runtime/'logs'/f'{stage}_train_attempt{attempt}.log',runtime,gpu)
                    while child.poll() is None:
                        if (runtime/'scientific_failure.json').is_file():
                            write(runtime/'pause_request.json',read(runtime/'scientific_failure.json'))
                            try:child.wait(timeout=60)
                            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=30)
                            raise RuntimeError('scientific failure in evaluation; training paused at a decision boundary')
                        write(runtime/'live_processes.json',dict(supervisor_pid=os.getpid(),training_pid=child.pid,workers=[p.pid for _,p in workers],gpu=gpu,stage=stage,updated_at=now()))
                        export(runtime);reports(runtime)
                        with db(runtime) as c:count=c.execute('SELECT COUNT(*) FROM checkpoints').fetchone()[0]
                        last=read(runtime/'last_synced_milestone.json',{}).get('count',0)
                        if count>last:
                            sync_milestone(runtime,f'{stage}-snapshot-{count}')
                            write(runtime/'last_synced_milestone.json',dict(count=count,updated_at=now()))
                        for i,(k,p) in enumerate(workers):
                            if p.poll() is not None:workers[i]=(k,launch_child(['worker','--kinds',k],runtime/'logs'/f'worker_{i}.log',runtime))
                        time.sleep(20)
                    if child.returncode==0:break
                    with db(runtime) as c:c.execute('INSERT INTO failures(job_id,attempt,error,created_at) VALUES(?,?,?,?)',(f'train/{stage}',attempt,f'exit {child.returncode}',now()))
                    log=(runtime/'logs'/f'{stage}_train_attempt{attempt}.log').read_text()
                    if any(s in log for s in ('FloatingPointError','resource safety:','contract mismatch','AssertionError','immutable snapshot')):raise RuntimeError('scientific, numerical or resource failure; inspect training log')
                    if attempt==MAX_ATTEMPTS:raise RuntimeError('training exhausted finite retries')
                completion=read(stage_dir/'training_complete.json')
                assert completion['environment_decision_steps']==BUDGETS[stage]
            # All required selection batches are needed only at the stage boundary.
            while not (stage_dir/'selection_report.json').is_file():
                if (runtime/'scientific_failure.json').is_file():raise RuntimeError('scientific failure; stage handoff refused')
                with db(runtime) as c:
                    rows=c.execute("SELECT * FROM jobs WHERE kind='selection' AND stage=?",(stage,)).fetchall()
                if any(r['status']=='failed' for r in rows):raise RuntimeError('selection job exhausted retries; cannot perform scientific handoff')
                if len(rows)==len(old.MILESTONES[stage]) and all(r['status']=='complete' for r in rows):
                    selection=old.select_balanced(stage_dir,old.MILESTONES[stage])
                    assert sha256_file(Path(selection['selected']['checkpoint']))==selection['selected']['checkpoint_sha256']
                    write(stage_dir/'handoff_gate.json',dict(status='PASS',selection_reason=selection['selection_reason'],selected=selection['selected'],completed_at=now()))
                    break
                write(runtime/'status.json',dict(status='running',phase='stage_boundary_selection_pending',stage=stage,completed_batches=sum(r['status']=='complete' for r in rows),required=len(old.MILESTONES[stage]),updated_at=now()))
                export(runtime);reports(runtime)
                for i,(k,p) in enumerate(workers):
                    if p.poll() is not None:workers[i]=(k,launch_child(['worker','--kinds',k],runtime/'logs'/f'worker_{i}.log',runtime))
                time.sleep(20)
            selection=read(stage_dir/'selection_report.json')['selected']
            for mode in ('argmax','sample'):enqueue('formal',stage,dict(mode=mode),f'formal/{stage}/{mode}',runtime)
            enqueue('reference',stage,dict(mode='argmax'),f'reference/{stage}/argmax',runtime)
            if stage!='stage3':
                next_stage=old.STAGE_ORDER[old.STAGE_ORDER.index(stage)+1]
                write(runtime/'stages'/next_stage/'handoff.json',dict(checkpoint=selection['checkpoint'],sha256=selection['checkpoint_sha256'],previous_stage=stage,gate=str(stage_dir/'handoff_gate.json'),selection_report=str(stage_dir/'selection_report.json'),created_at=now()))
            reports(runtime);sync_milestone(runtime,stage)
        write(runtime/'training_curriculum_complete.json',dict(status='three_training_budgets_complete',completed_at=now()))
        while True:
            with db(runtime) as c:rows=c.execute('SELECT status,kind,id FROM jobs').fetchall()
            pending=[r['id'] for r in rows if r['status'] in ('pending','running')]
            if not pending:break
            write(runtime/'status.json',dict(status='running',phase='remaining_formal_and_gifs',pending=len(pending),updated_at=now()))
            export(runtime);reports(runtime)
            for i,(k,p) in enumerate(workers):
                if p.poll() is not None:workers[i]=(k,launch_child(['worker','--kinds',k],runtime/'logs'/f'worker_{i}.log',runtime))
            time.sleep(20)
        if (runtime/'scientific_failure.json').is_file():raise RuntimeError('scientific failure; final acceptance refused')
        missing=[r['id'] for r in rows if r['status']=='failed']
        write(runtime/'orchestration_complete.json',dict(status='COMPLETE' if not missing else 'COMPLETE_WITH_MISSING_ARTIFACTS',missing=missing,completed_at=now()))
        write(runtime/'status.json',read(runtime/'orchestration_complete.json'));export(runtime);reports(runtime);sync_milestone(runtime,'final')
    except Exception:
        write(runtime/'status.json',dict(status='SAFE_PAUSE',error=traceback.format_exc(),updated_at=now()))
        export(runtime);reports(runtime)
        raise

# Reports are refreshed by the supervisor without claiming unfinished stages are complete.
def reports(runtime):
    from tools.z_support_full3stage_reports_20261017 import generate
    generate(runtime)

def sync_milestone(runtime,label):
    export(runtime);reports(runtime);plot_curves(runtime)
    paths=[str(ART.relative_to(ROOT)),str(DOC.relative_to(ROOT))]
    try:
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True,text=True)
        dirty=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode
        if dirty:
            subprocess.run(['git','commit','-m',f'artifact: Z support full curriculum {label} status'],cwd=ROOT,check=True,capture_output=True,text=True)
            subprocess.run(['git','push','origin','HEAD:refs/heads/experiment/iqn-z-guided-support-full3stage-20261017'],cwd=ROOT,check=True,capture_output=True,text=True,timeout=60)
        remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/experiment/iqn-z-guided-support-full3stage-20261017'],cwd=ROOT,text=True,timeout=30).split()[0]
        if remote!=head():raise RuntimeError('remote HEAD verification mismatch')
        write(runtime/'git_sync.json',dict(status='synced',head=head(),remote_head=remote,label=label,updated_at=now()))
    except Exception as e:write(runtime/'git_sync.json',dict(status='pending',label=label,error=str(e),updated_at=now()))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','train','supervise','worker','job','formal','export','reports'));p.add_argument('--runtime',type=Path,default=RUNTIME);p.add_argument('--stage',choices=old.STAGE_ORDER);p.add_argument('--device',default='cpu');p.add_argument('--gpu',default='0');p.add_argument('--kinds',default='selection');p.add_argument('--job-id');p.add_argument('--mode',default='argmax');p.add_argument('--smoke-steps',type=int);p.add_argument('--episodes',type=int,default=20);a=p.parse_args();torch.set_num_threads(1)
    if a.command=='prepare':prepare(a.runtime)
    elif a.command=='train':train_stage(a.stage,a.runtime,a.device)
    elif a.command=='supervise':supervise(a.runtime,a.gpu)
    elif a.command=='worker':worker(tuple(a.kinds.split(',')),a.runtime)
    elif a.command=='job':job(a.job_id,a.runtime)
    elif a.command=='formal':formal_batch(a.stage,a.mode,a.runtime,a.episodes,a.smoke_steps)
    elif a.command=='export':export(a.runtime)
    elif a.command=='reports':reports(a.runtime)
if __name__=='__main__':main()
