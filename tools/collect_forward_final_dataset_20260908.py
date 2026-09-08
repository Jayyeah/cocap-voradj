#!/usr/bin/env python3
"""Gate-checked full-task teacher data, with phase audit before scaling."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.forward_final import CONTRACT,TEACHER,TEACHER_SHA,preflight
from tools.run_forward_final_bridge_20260908 import run_episode,atomic_json,summarize
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file
from tools.run_continuous_ctde_training import _split_termination_flags

PHASES={'pre_capture':0,'post_capture':1,'pure_coverage':2}
ROLES={'direct':0,'pursuing_memory':1,'support':2,'coverage':3}


class EpisodeRows:
    def __init__(self,episode):
        self.episode=episode;self.rows=defaultdict(list);self.global_rows=defaultdict(list)
        self.seen=set();self.agent_seen=defaultdict(set);self.capture_events=[]

    def __call__(self,env,local,global_state,q,greedy,active,state,phase,step,outcome):
        n=len(active)
        for k in local[0]:self.rows['local_obs.'+k].append(np.stack([o[k] for o in local]))
        for k,v in global_state.items():self.global_rows['global_state.'+k].append(np.asarray(v))
        term,trunc=_split_termination_flags(outcome.dones,outcome.infos)
        all_direct=set().union(*state['direct'].values()) if state['direct'] else set()
        undiscovered=bool(phase=='pre_capture' and not self.seen and not all_direct)
        near_capture=any(len(t['participants'])>=2 for t in state['targets'].values())
        self.seen.update(all_direct)
        direct=[];support=[];roles=[];targets=[];direct_mask=[];informed_mask=[];hazards=[];detection=[]
        for row,i in enumerate(active):
            own=state['direct'].get(i,set())
            informed=set().union(*(state['direct'].get(f,set()) for f in state['friends'].get(i,set())))
            metadata=outcome.infos[i]['replay_metadata']
            pursuing=bool(metadata['effective_pursuing'])
            is_support=bool(metadata['support_candidate'])
            roles.append(0 if own else 1 if pursuing else 2 if is_support else 3)
            direct.append(bool(own));support.append(is_support)
            candidates=own or informed;targets.append(min(candidates) if candidates else -1)
            direct_mask.append([j in own for j in range(8)]);informed_mask.append([j in informed for j in range(8)])
            detection.append(bool(own-self.agent_seen[i]));self.agent_seen[i].update(own)
            # Local, pre-action hazard diagnostic; never use after-step geometry as an input label.
            obs=local[row];scale=env._distance_scale()
            clearances=[float(obs['self'][2])*scale,float(np.linalg.norm(obs['self'][3:5]))*scale-env.pursuers[i].r]
            mask=np.asarray(obs['masks'])[np.asarray(obs['types'])==1].astype(bool)
            clearances.extend(float(np.linalg.norm(p[:2]))*scale-2*env.pursuers[i].r for p in obs['pursuers'][mask])
            hazards.append(min(clearances))
        events=list(env.last_capture_events)
        self.capture_events.extend({'step':step,**e} for e in events)
        scalars={'teacher_q':q,'greedy_action':greedy,'episode':np.full(n,self.episode,np.int32),
            'timestep':np.full(n,step,np.int32),'agent':np.asarray(active,np.int16),
            'transition_index':np.full(n,step-1,np.int32),'phase_id':np.full(n,PHASES[phase],np.uint8),
            'role_id':np.asarray(roles,np.uint8),'direct':np.asarray(direct,bool),'support':np.asarray(support,bool),
            'pursuing':np.asarray([outcome.infos[i]['replay_metadata']['effective_pursuing'] for i in active],bool),
            'target_id':np.asarray(targets,np.int16),'direct_target_mask':np.asarray(direct_mask,bool),
            'neighbor_target_mask':np.asarray(informed_mask,bool),'undiscovered':np.full(n,undiscovered,bool),
            'first_local_detection':np.asarray(detection,bool),'near_capture':np.full(n,near_capture,bool),
            'capture_transition':np.full(n,bool(events),bool),'reward':np.asarray(outcome.rewards)[active].astype(np.float32),
            'done':np.asarray(outcome.dones)[active],'terminated':term[active],'truncated':trunc[active],
            'near_hazard_local':np.asarray(hazards)<4,'local_hazard_clearance':np.asarray(hazards,np.float32),
            'collision_transition':np.full(n,bool(env.last_collision_events),bool)}
        for k,v in scalars.items():self.rows[k].append(v)

    def arrays(self,failed):
        arrays={k:np.concatenate(v) for k,v in self.rows.items()}
        arrays.update({k:np.stack(v) for k,v in self.global_rows.items()})
        arrays['episode_failed']=np.full(len(arrays['episode']),failed,bool)
        assert np.array_equal(arrays['teacher_q'].argmax(1),arrays['greedy_action'])
        assert arrays['transition_index'].max()<len(arrays['global_state.active_mask'])
        return arrays


def distribution_audit(shards,root):
    phase=Counter();roles=Counter();actions=np.zeros(9,np.int64);flags=Counter();total=0;q_gaps=[]
    by_phase={name:np.zeros(9,np.int64) for name in PHASES}
    for shard in shards:
        with np.load(root/shard['path'],allow_pickle=False) as x:
            n=len(x['episode']);total+=n;actions+=np.bincount(x['greedy_action'],minlength=9)
            for name,i in PHASES.items():
                mask=x['phase_id']==i;phase[name]+=int(mask.sum());by_phase[name]+=np.bincount(x['greedy_action'][mask],minlength=9)
            for name,i in ROLES.items():roles[name]+=int((x['role_id']==i).sum())
            for k in ('undiscovered','first_local_detection','near_capture','capture_transition','near_hazard_local','collision_transition','episode_failed'):flags[k]+=int(x[k].sum())
            ordered=np.sort(x['teacher_q'],axis=1);q_gaps.extend((ordered[:,-1]-ordered[:,-2]).tolist())
    checks={'each_phase_at_least_5pct':all(v/max(total,1)>=.05 for v in phase.values()) and len(phase)==3,
        'direct_at_least_1pct':roles['direct']/max(total,1)>=.01,'support_at_least_0_2pct':roles['support']/max(total,1)>=.002,
        'has_undiscovered':flags['undiscovered']>=4,'has_detection':flags['first_local_detection']>=4,
        'has_capture_transitions':flags['capture_transition']>=4,'has_near_capture':flags['near_capture']>=4,
        'has_failure_or_near_hazard':flags['episode_failed']+flags['near_hazard_local']>=4}
    return {'status':'PASS' if all(checks.values()) else 'STOP_ADJUST_COLLECTION','checks':checks,'rows':total,
        'phase_counts':dict(phase),'phase_fraction':{k:v/max(total,1) for k,v in phase.items()},'role_counts':dict(roles),
        'action_histogram':actions.tolist(),'action_histogram_by_phase':{k:v.tolist() for k,v in by_phase.items()},
        'state_counts':dict(flags),'q_gap_percentiles':np.percentile(q_gaps,[10,50,90]).tolist() if q_gaps else [],
        'limits':'Presence/proportion gate, not proof of exhaustive distribution coverage. near_hazard_local<4m is a proxy, not an actual failure.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--c0-report',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=12);p.add_argument('--seed',type=int,default=2026093101);p.add_argument('--device',default='cuda:0')
    p.add_argument('--pilot-audit',type=Path);args=p.parse_args()
    gate=json.loads(args.c0_report.read_text());assert gate['decision']=='PASS' and gate['contract']==CONTRACT and gate['teacher_sha256']==TEACHER_SHA
    if args.pairs>12:
        if args.pilot_audit is None:raise ValueError('Bulk collection requires pilot distribution audit')
        assert json.loads(args.pilot_audit.read_text())['status']=='PASS'
    out=args.output_root
    if (out/'manifest.json').exists():raise ValueError('Use fresh output; do not overwrite shards')
    atomic_json(out/'runtime_preflight.json',preflight());assert sha256_file(TEACHER)==TEACHER_SHA
    model=CoCapIQN.load(str(TEACHER),device=args.device).eval();shards=[];records=[];start=time.monotonic()
    manifest={'schema':'forward-final-teacher-dataset-v1','contract':CONTRACT,'teacher_sha256':TEACHER_SHA,'seed_base':args.seed,
        'pairs':args.pairs,'status':'running','shards':shards,'row_count':0,'phase_ids':PHASES,'role_ids':ROLES,
        'global_state_layout':'global_state arrays are per transition; agent rows reference episode-local transition_index; never concatenate as agent rows',
        'target_id_semantics':'lowest-ID locally/neighbor observed target, -1 if unresolved; evaluator association, not actor input or optimal assignment',
        'sampling':'alternating mixed/pure coverage, complete teacher trajectories, no success filtering',
        'c0_report':str(args.c0_report),'c0_report_sha256':sha256_file(args.c0_report)}
    atomic_json(out/'manifest.json',manifest)
    for pair in range(args.pairs):
        for scene in ('mixed','coverage'):
            episode=len(records);storage=EpisodeRows(episode)
            row=run_episode(model,scene,args.seed+pair+(100000 if scene=='coverage' else 0),args.device,on_transition=storage)
            arrays=storage.arrays(not row['safe_complete']);path=out/'shards'/f'episode_{episode:05d}.npz';path.parent.mkdir(exist_ok=True)
            tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
            os.replace(tmp,path)
            shards.append({'path':str(path.relative_to(out)),'sha256':sha256_file(path),'rows':len(arrays['episode']),'transitions':row['length'],'episode':episode,'scene':scene})
            manifest['row_count']+=len(arrays['episode']);records.append(row)
            atomic_json(out/'episodes'/f'episode_{episode:05d}.json',{'record':row,'capture_events':storage.capture_events})
            atomic_json(out/'manifest.json',manifest)
            elapsed=time.monotonic()-start
            atomic_json(out/'progress.json',{'status':'running','completed':len(records),'total':args.pairs*2,'rows':manifest['row_count'],'elapsed_seconds':elapsed,'eta_seconds':elapsed/len(records)*(args.pairs*2-len(records))})
    audit=distribution_audit(shards,out);audit['episode_outcomes']=summarize(records)
    atomic_json(out/'distribution_audit.json',audit)
    manifest['status']='complete';manifest['distribution_gate']=audit['status'];atomic_json(out/'manifest.json',manifest)
    atomic_json(out/'progress.json',{'status':'complete','completed':len(records),'rows':manifest['row_count'],'eta_seconds':0,'distribution_gate':audit['status']})
    print(json.dumps({k:v for k,v in audit.items() if k!='episode_outcomes'},indent=2))

if __name__=='__main__':main()
