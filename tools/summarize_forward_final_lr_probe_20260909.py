#!/usr/bin/env python3
"""Evaluate one predeclared LR diagnostic; never starts a training job."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from tools.evaluate_forward_final_actor_20260908 import paired_delta
from tools.run_forward_final_bridge_20260908 import atomic_json
from tools.train_forward_final_ppo_20260909 import BC_SHA


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    gate=json.loads((root/'task_gate.json').read_text());parity=json.loads((root/'parity.json').read_text());assert parity['status']=='PASS'
    parent=json.loads((root.parent/'smoke_task_eval20/bc_parent/report.json').read_text())
    low=json.loads((root/'low_lr_eval20/report.json').read_text())
    assert parent['status']==low['status']=='complete' and parent['actor_sha256']==BC_SHA
    training=json.loads((root/'low_lr/report.json').read_text())
    assert low['actor_sha256']==training['checkpoints']['actor_sha256']
    assert low['contract']==parent['contract'] and low['seed_base']==parent['seed_base']
    assert low['bc_parent_sha256']==BC_SHA and low['training_steps']==512
    assert low['evaluator_sha256']==parent['evaluator_sha256']
    paired=paired_delta(parent['records'],low['records']);checks={}
    for scene in ('mixed','coverage'):
        a=parent['summary'][scene];b=low['summary'][scene]
        assert a['episodes']==b['episodes']>=gate['minimum_per_scene']
        for k in ('safe_complete_rate','ce_success_rate')+(('captured_rate',) if scene=='mixed' else ()):
            checks[f'{scene}/{k}']=b[k]>=gate['min_reliability']-1e-9 and a[k]-b[k]<=gate['max_reliability_drop']+1e-9
        checks[f'{scene}/collision']=b['collision_rate']-a['collision_rate']<=gate['max_collision_increase']+1e-9
        for statistic,limit in [('mean','max_mean_mission_ratio'),('p50','max_p50_mission_ratio'),('p90','max_p90_mission_ratio')]:
            before=a['full_mission_seconds_completed'][statistic];after=b['full_mission_seconds_completed'][statistic]
            checks[f'{scene}/{statistic}']=before is not None and after is not None and after<=before*gate[limit]
    updates={m:json.loads((root/m/'updates.json').read_text()) for m in ('original_lr','low_lr')}
    output={'decision':'ELIGIBLE_FOR_SHORT_BUDGET_NOT_AUTOSTARTED' if all(checks.values()) else 'HOLD_PPO_SCALE','checks':checks,'gate':gate,'paired':paired,'summary':{'bc_parent':parent['summary'],'low_lr':low['summary']},'updates':updates,'parity':parity,'interpretation':'Small fixed diagnostic seed family; whole-update KL reduction alone is not evidence of task improvement.'}
    atomic_json(root/'comparison.json',output);print(json.dumps(output,indent=2))

if __name__=='__main__':main()
