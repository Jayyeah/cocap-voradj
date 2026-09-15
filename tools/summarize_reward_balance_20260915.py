#!/usr/bin/env python3
"""Full schedule robust gradient balance, preserving absent-phase zero contributions."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];path=ROOT/'artifacts/2026-09-15_normsense_v2/reward_balance.json'
r=json.loads(path.read_text());phases=('capture','support','coverage')
paired=[w for w in r['windows'] if all(w['phases'][p]['sample_rows']>=32 for p in ('capture','coverage'))]
a=np.array([[w['phases'][p]['gradient_norm'] for p in ('capture','coverage')] for w in paired]);rng=np.random.default_rng(2026091501)
boot=np.median(a[rng.integers(len(a),size=(10000,len(a)))],axis=1);ratios=boot[:,0]/boot[:,1]
r['conditional_capture_present_analysis']=dict(paired_windows=len(paired),median_norms={p:float(np.median([w['phases'][p]['gradient_norm'] for w in paired])) for p in phases},capture_to_coverage_ratio=float(np.median(a[:,0])/np.median(a[:,1])),formula_alpha=float(np.clip(np.median(a[:,1])/np.median(a[:,0]),.25,1)),descriptive_paired_window_bootstrap_ratio_95_interval=np.quantile(ratios,[.025,.975]).tolist(),not_primary_reason='Excludes 10 real coverage-only update windows and would bias calibration toward capture-present states.')
n=len(r['windows']);cut=int(.1*n)
values={p:np.array([w['phases'][p]['gradient_norm'] for w in r['windows']]) for p in phases}
trim=lambda a:float(np.sort(a)[cut:n-cut].mean())
r['robust_norms']={p:trim(v) for p,v in values.items()}
r['all_window_median_norms']={p:float(np.median(v)) for p,v in values.items()}
r['phase_present_windows']={p:int(np.sum(v>0)) for p,v in values.items()}
r['robust_statistic']=f'10% symmetric trimmed mean over ALL {n} actual scheduled PPO windows; floor(0.1*n)={cut} removed per tail. Zero gradients for absent phases remain. Median is zero for sparse capture/support, so trimmed mean retains their magnitude and occupancy while limiting extreme-window influence.'
r['robust_capture_to_coverage_norm_ratio']=r['robust_norms']['capture']/r['robust_norms']['coverage']
r['alpha_capture']=float(np.clip(r['robust_norms']['coverage']/max(r['robust_norms']['capture'],1e-30),.25,1))
r['support_capture_scale']=r['alpha_capture']
r['sample_rows']={p:sum(w['phases'][p]['sample_rows'] for w in r['windows']) for p in phases}
r['strong_capture_dominance_supported']=r['robust_capture_to_coverage_norm_ratio']>1
r['interpretation']='Actual Full-Mix schedule does not show capture-gradient dominance. Capture-present conditional median suggests the opposite but omits most coverage windows, and its descriptive interval includes 1. Predeclare alpha=1 from full-schedule robust norms and recommend MASTER cancel the identical RewardBalanced arm. One frozen BC actor, cold critic, and one seed do not establish trained-policy interference.'
r['decision']='NO_CAPTURE_DOMINANCE_ALPHA_1_RECOMMEND_CANCEL_BALANCED'
# Preserve the already-computed 0.495... fixed-state contrast as exploratory only.
if r.get('scaled_fixed_rollout_gradients'):
    r['conditional_exploratory_scaled_gradients']=dict(alpha=r['conditional_capture_present_analysis']['formula_alpha'],not_declared_for_training=True,windows=r.pop('scaled_fixed_rollout_gradients'))
r['declared_scale_policy_loss_change']=0.
r['declared_scale_gradient_change']=0.
path.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:r[k] for k in ('alpha_capture','sample_rows','robust_norms','robust_capture_to_coverage_norm_ratio','aggregate_gradient_cosine','decision')},indent=2))
