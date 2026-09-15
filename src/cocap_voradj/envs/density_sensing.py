"""Density-normalized onboard sensing, measured in surface clearance metres."""
from __future__ import annotations
import copy
import hashlib
import math
import numpy as np
from .coverage_ce import _free_space_mask

POLICY = 'forward-final-density-normalized-sensing-v2'
SCHEMA = 2
# Filled from the offline geometry sweep; no per-entity radius fields in V2.
CALIBRATED_K = 0.8715
RADIUS_FLOOR = 20.0
BLIND_TOLERANCE = 0.001


def resolve_density_normalized_onboard_radius(num_pursuers, width, height,
        free_space_area, *, k=CALIBRATED_K, radius_floor=RADIUS_FLOOR):
    """Resolve a surface radius; large-N maps must first satisfy scale rule."""
    if (not all(math.isfinite(x) and x > 0 for x in (num_pursuers,width,height,free_space_area,k,radius_floor))
            or int(num_pursuers) != num_pursuers or free_space_area > width * height):
        raise ValueError('Invalid density sensing geometry')
    raw = float(k * math.sqrt(free_space_area / num_pursuers))
    required_scale = max(1., radius_floor / raw)
    if required_scale > 1. + 1e-9:
        raise ValueError(f'Map must scale by {required_scale}; refusing a silently clamped sensor')
    return dict(policy=POLICY, schema_version=SCHEMA, k=float(k),
        resolved_onboard_radius=raw, num_pursuers=int(num_pursuers),
        map_size=[float(width), float(height)], free_space_area=float(free_space_area),
        normalized_radius=raw / math.sqrt(free_space_area / num_pursuers),
        radius_floor=float(radius_floor), distance_semantics='surface_clearance',
        blind_area_metric=dict(tolerance=BLIND_TOLERANCE, population='representative CVT free mask',
                               measured_fraction=({4: 0.0, 8: 0.0, 12: 0.0}.get(int(num_pursuers)) if k == CALIBRATED_K and width == height == 120 else None),
                               applies_to='offline 72-layout calibration, not a guarantee for current agent positions', artifact='artifacts/2026-09-15_normsense_v2/geometry.json'))


def map_scale(num_pursuers, base_free_area, *, k=CALIBRATED_K, radius_floor=RADIUS_FLOOR):
    """Scale all map/obstacle coordinates; remeasure mask and resolve before use."""
    return max(1., radius_floor * math.sqrt(num_pursuers / base_free_area) / k)


def enable_v2(config, *, k=CALIBRATED_K):
    cfg = copy.deepcopy(config)
    v = cfg['voradj']
    for key in ('enemy_sensing_radius','obstacle_sensing_radius','vct_ls_enemy_sensing_radius','vct_ls_obstacle_sensing_radius'):
        v.pop(key, None)
    v['onboard_sensing'] = dict(policy=POLICY, schema_version=SCHEMA, k=float(k), radius_floor=RADIUS_FLOOR)
    if not v.get('local_sensing_uses_surface_distance', True) or cfg.get('perception', {}).get('global_evader_visibility', False):
        raise ValueError('V2 requires surface distance and forbids global enemy visibility')
    return cfg


def runtime_metadata(env):
    v = env.config.get('voradj', {})
    spec = v['onboard_sensing']
    if spec['policy'] != POLICY or spec['schema_version'] != SCHEMA:
        raise ValueError('Unknown onboard policy/schema')
    if any(key in v for key in ('enemy_sensing_radius','obstacle_sensing_radius','vct_ls_enemy_sensing_radius','vct_ls_obstacle_sensing_radius')):
        raise ValueError('Independent enemy/obstacle radii forbidden in V2')
    if not env._vct_ls_use_surface_distance() or env.per_cfg.get('global_evader_visibility',False):
        raise ValueError('V2 requires surface clearance and no global enemy broadcast')
    width,height = float(env.env_cfg['width']),float(env.env_cfg['height'])
    grid_n = max(8,int(env.reward_cfg.get('voradj_grid_size',env.reward_cfg.get('distribution_voronoi_grid_size',60))))
    obstacles = tuple((float(o.x),float(o.y),float(o.r)) for o in env.obstacles)
    robot_radius = max(float(p.r) for p in env.pursuers)
    signature = (len(env.pursuers),width,height,obstacles,robot_radius,grid_n,float(spec['k']),float(spec['radius_floor']))
    cache = getattr(env,'_density_sensing_cache',None)
    if cache is not None and cache[0] == signature: return copy.deepcopy(cache[1])
    xl,xr,yb,yt = env._bounds()
    xx,yy = np.meshgrid(np.linspace(xl,xr,grid_n),np.linspace(yb,yt,grid_n))
    points = np.c_[xx.ravel(),yy.ravel()]
    mask = _free_space_mask(points,obstacles,robot_radius,env._masked_voronoi_grid_margin(grid_n))
    result = resolve_density_normalized_onboard_radius(len(env.pursuers),width,height,width*height*float(mask.mean()),k=spec['k'],radius_floor=spec['radius_floor'])
    result.update(free_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),free_mask_grid_n=grid_n,
                  free_area_method='Final free_mask_projected grid fraction times map area',
                  friendly_information='historical global-friendly VorAdj unchanged; local-friendly deferred to V3')
    env._density_sensing_cache = (signature,result)
    return copy.deepcopy(result)


def resolve_density_normalized_map(num_pursuers, base_width, base_height, base_free_space_area,
                                   *, k=CALIBRATED_K, radius_floor=RADIUS_FLOOR):
    """Plan geometrically similar map scaling, with complete resolved metadata.

    Larger-N launchers must instantiate/re-measure the physical free mask and
    call runtime_metadata again; this plan is not permission to run them.
    """
    scale = map_scale(num_pursuers,base_free_space_area,k=k,radius_floor=radius_floor)
    width,height = base_width*scale,base_height*scale
    metadata = resolve_density_normalized_onboard_radius(num_pursuers,width,height,
        base_free_space_area*scale**2,k=k,radius_floor=radius_floor)
    metadata.update(base_map_size=[base_width,base_height],map_linear_scale=scale,
        map_area_scale=scale**2,base_free_space_area=base_free_space_area,
        floor_start_N=k**2*base_free_space_area/radius_floor**2,
        formula='s=max(1,R_floor*sqrt(N/A0)/k); W=sW0; H=sH0; A=s^2 A0; R=k sqrt(A/N)',
        geometry_assumption='same free-space fraction; instantiate and remeasure known obstacle mask before runtime',
        larger_N_experiment_authorized=False)
    return metadata
