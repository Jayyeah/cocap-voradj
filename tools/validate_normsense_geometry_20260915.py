#!/usr/bin/env python3
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
from scipy.spatial.distance import cdist
from tools.forward_final_single_task_20260915 import p,task_config
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.envs.coverage_ce import _free_space_mask
from cocap_voradj.envs.density_sensing import enable_v2,runtime_metadata,map_scale
path=ROOT/'artifacts/2026-09-15_normsense_v2/geometry.json';report=json.loads(path.read_text())
for row in report['layouts']:
    cfg=task_config('coverage');cfg['env']['num_pursuers']=row['n'];cfg['env']['num_obstacles']=row['num_obstacles'];cfg=enable_v2(cfg)
    p.set_global_config(cfg);env=VorAdjEnv(cfg,seed=row['seed']);env.reset();meta=runtime_metadata(env)
    assert meta['k']==report['calibrated_k']
    radius=env._vct_ls_sensing_radius('enemy');assert radius==env._vct_ls_sensing_radius('obstacle')
    for robot,xy in zip(env.pursuers,row['sites']):robot.x,robot.y=xy
    env._invalidate_voronoi_cache()
    ce=env._voradj_coverage_geometry(env._coverage_voronoi_map(),strict=True)
    row['runtime_ce_geometry']={k:ce[k] for k in ('ce_center_rms','ce_center_max','area_cv','converged_now','ce_component_count_max','ce_disconnected_ratio_max')}
    assert ce['converged_now'], 'Calibration layout is not runtime strict CE geometry'
    xx,yy=np.meshgrid(np.linspace(0,120,241),np.linspace(0,120,241));pts=np.c_[xx.ravel(),yy.ravel()]
    # Preserve the authoritative coarse-mask physical inflation; only quadrature gets finer.
    mask=_free_space_mask(pts,row['obstacles'],env.pursuers[0].r,env._masked_voronoi_grid_margin(60));pts=pts[mask]
    distances=cdist(pts,row['sites'])-env.pursuers[0].r;counts=(distances<=radius).sum(1)
    row['fine_grid_blind_fraction']=float(np.mean(counts==0));row['resolved_runtime']=meta
    adjacency=cdist(row['sites'],row['sites'])<=2*(radius+env.pursuers[0].r);np.fill_diagonal(adjacency,False)
    seen={0}
    while True:
        nxt=seen|{j for i in seen for j in np.flatnonzero(adjacency[i])}
        if nxt==seen:break
        seen=nxt
    row['all_disks_connected']=len(seen)==row['n']
    assert row['all_disks_connected']
    row['isolated_disks']=int(np.sum(adjacency.sum(1)==0));row['overlap_pairs']=int(adjacency.sum()//2)
    row['overlap_free_fraction']=float(np.mean(counts>=2));row['mean_detector_count']=float(counts.mean())
    assert row['fine_grid_blind_fraction']<=report['blind_tolerance'] and row['isolated_disks']==0
report['summary']={str(n):dict(radius_min=min(r['radius'] for r in report['layouts'] if r['n']==n),radius_max=max(r['radius'] for r in report['layouts'] if r['n']==n),worst_blind=max(r['fine_grid_blind_fraction'] for r in report['layouts'] if r['n']==n),mean_detectors=float(np.mean([r['mean_detector_count'] for r in report['layouts'] if r['n']==n])),isolates=0) for n in (4,8,12)}
report['floor_source']='20m historical physical onboard range retained as lower bound; not an optimized hardware minimum'
report['floor_threshold_N']=report['calibrated_k']**2*report['layouts'][0]['area']/20**2
report['fine_validation']='241x241 grid (0.5m), same known obstacle inflation as Final 60x60 free mask'
path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary'],indent=2))
