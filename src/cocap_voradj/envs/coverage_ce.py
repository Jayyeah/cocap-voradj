from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage


SiteKey = Tuple[str, int]
BLOCKED_OWNER = -1


def _free_space_mask(
    grid_points: np.ndarray,
    obstacles: Sequence[Tuple[float, float, float]],
    robot_radius: float,
    grid_margin: float,
) -> np.ndarray:
    free = np.ones(len(grid_points), dtype=bool)
    for x, y, radius in obstacles:
        center = np.asarray([x, y], dtype=float)
        inflated = max(float(radius) + robot_radius + grid_margin, 0.0)
        free &= np.linalg.norm(grid_points - center, axis=1) > inflated
    return free


def _component_target(
    owner_grid: np.ndarray,
    grid_points: np.ndarray,
    site: np.ndarray,
    site_id: int,
    obstacles: Sequence[Tuple[float, float, float]],
    robot_radius: float,
    grid_margin: float,
) -> Tuple[np.ndarray, np.ndarray, int, float, float]:
    owned_mask = owner_grid == site_id
    owned_count = int(np.count_nonzero(owned_mask))
    if owned_count == 0:
        fallback = np.asarray(site, dtype=float).copy()
        return fallback, fallback, 0, 0.0, 0.0

    structure = ndimage.generate_binary_structure(2, 1)
    labels, component_count = ndimage.label(owned_mask, structure=structure)
    owned_rows, owned_cols = np.nonzero(owned_mask)
    grid_n = owner_grid.shape[0]
    flat_indices = owned_rows * grid_n + owned_cols
    owned_points = grid_points[flat_indices]
    seed_local = int(np.argmin(np.sum((owned_points - site) ** 2, axis=1)))
    seed_label = int(labels[owned_rows[seed_local], owned_cols[seed_local]])
    component_mask = labels == seed_label
    component_rows, component_cols = np.nonzero(component_mask)
    component_indices = component_rows * grid_n + component_cols
    component_points = grid_points[component_indices]
    raw_centroid = component_points.mean(axis=0)

    raw_free = True
    for x, y, radius in obstacles:
        inflated = max(float(radius) + robot_radius + grid_margin, 0.0)
        if float(np.linalg.norm(raw_centroid - np.asarray([x, y], dtype=float))) <= inflated:
            raw_free = False
            break
    nearest_local = int(np.argmin(np.sum((component_points - raw_centroid) ** 2, axis=1)))
    nearest_point = component_points[nearest_local]
    raw_grid_index = int(np.argmin(np.sum((grid_points - raw_centroid) ** 2, axis=1)))
    raw_row, raw_col = divmod(raw_grid_index, grid_n)
    raw_in_component = bool(component_mask[raw_row, raw_col])
    target = raw_centroid if raw_free and raw_in_component else nearest_point
    projection_distance = float(np.linalg.norm(raw_centroid - target))
    disconnected_ratio = float(1.0 - len(component_points) / max(owned_count, 1))
    return target, raw_centroid, int(component_count), disconnected_ratio, projection_distance


def build_masked_voronoi_map(
    *,
    keys: List[SiteKey],
    sites: np.ndarray,
    grid_points: np.ndarray,
    grid_n: int,
    obstacles: Sequence[Tuple[float, float, float]],
    robot_radius: float,
    grid_margin: float,
) -> Dict[str, Any]:
    """Build an obstacle-masked Euclidean Voronoi map on a fixed grid."""
    free_mask = _free_space_mask(grid_points, obstacles, robot_radius, grid_margin)
    owners = np.full(len(grid_points), BLOCKED_OWNER, dtype=np.int64)
    if keys and np.any(free_mask):
        free_points = grid_points[free_mask]
        dist_sq = np.sum((free_points[:, None, :] - sites[None, :, :]) ** 2, axis=2)
        owners[free_mask] = np.argmin(dist_sq, axis=1).astype(np.int64)
    owner_grid = owners.reshape(grid_n, grid_n)

    counts: Dict[SiteKey, int] = {}
    centroids: Dict[SiteKey, np.ndarray] = {}
    raw_centroids: Dict[SiteKey, np.ndarray] = {}
    component_counts: Dict[SiteKey, int] = {}
    disconnected_ratios: Dict[SiteKey, float] = {}
    projection_distances: Dict[SiteKey, float] = {}
    for site_id, key in enumerate(keys):
        counts[key] = int(np.count_nonzero(owners == site_id))
        target, raw, component_count, disconnected_ratio, projection_distance = _component_target(
            owner_grid,
            grid_points,
            sites[site_id],
            site_id,
            obstacles,
            robot_radius,
            grid_margin,
        )
        centroids[key] = target
        raw_centroids[key] = raw
        component_counts[key] = component_count
        disconnected_ratios[key] = disconnected_ratio
        projection_distances[key] = projection_distance

    adjacency: Dict[SiteKey, set[SiteKey]] = {key: set() for key in keys}
    shared_counts: Dict[Tuple[SiteKey, SiteKey], int] = {}

    def add_edges(a: np.ndarray, b: np.ndarray) -> None:
        valid = (a >= 0) & (b >= 0) & (a != b)
        for sid_a, sid_b in zip(a[valid].ravel(), b[valid].ravel()):
            key_a = keys[int(sid_a)]
            key_b = keys[int(sid_b)]
            adjacency[key_a].add(key_b)
            adjacency[key_b].add(key_a)
            pair = tuple(sorted((key_a, key_b)))
            shared_counts[pair] = shared_counts.get(pair, 0) + 1

    add_edges(owner_grid[:, :-1], owner_grid[:, 1:])
    add_edges(owner_grid[:-1, :], owner_grid[1:, :])

    obstacle_adjacent_sites: Dict[int, set[SiteKey]] = {}
    if grid_n > 1:
        reshaped = grid_points.reshape(grid_n, grid_n, 2)
        dx = float(np.linalg.norm(reshaped[0, 1] - reshaped[0, 0]))
        dy = float(np.linalg.norm(reshaped[1, 0] - reshaped[0, 0]))
        neighbor_margin = float(np.hypot(dx, dy))
    else:
        neighbor_margin = 0.0
    for obstacle_id, (x, y, radius) in enumerate(obstacles):
        center = np.asarray([x, y], dtype=float)
        ring_radius = float(radius) + robot_radius + grid_margin + neighbor_margin
        near = free_mask & (np.linalg.norm(grid_points - center, axis=1) <= ring_radius)
        owner_ids = {int(owner) for owner in owners[near] if int(owner) >= 0}
        obstacle_adjacent_sites[obstacle_id] = {keys[owner_id] for owner_id in owner_ids}

    return {
        "keys": keys,
        "sites": sites,
        "points": grid_points,
        "owners": owners,
        "owner_grid": owner_grid,
        "free_mask": free_mask,
        "free_grid": free_mask.reshape(grid_n, grid_n),
        "free_grid_count": int(np.count_nonzero(free_mask)),
        "blocked_grid_count": int(np.count_nonzero(~free_mask)),
        "counts": counts,
        "centroids": centroids,
        "raw_centroids": raw_centroids,
        "centroid_component_counts": component_counts,
        "centroid_disconnected_ratios": disconnected_ratios,
        "centroid_projection_distances": projection_distances,
        "adjacency": adjacency,
        "shared_counts": shared_counts,
        "obstacle_adjacent_sites": obstacle_adjacent_sites,
        "grid_n": grid_n,
        "voronoi_obstacle_mode": "free_mask_projected",
    }


def normalized_center_costs(
    positions: np.ndarray,
    centroids: Sequence[np.ndarray],
    map_diagonal: float,
    participant_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scale = float(np.sqrt(max(participant_count, 1)) / max(map_diagonal, 1e-6))
    distances = np.asarray(
        [float(np.linalg.norm(positions[idx] - np.asarray(centroid, dtype=float)) * scale) for idx, centroid in enumerate(centroids)],
        dtype=float,
    )
    return distances, distances ** 2


def normalized_control_cost(
    *,
    speed: float,
    acceleration: float,
    angular_velocity: float,
    max_speed: float,
    max_acceleration: float,
    max_angular_velocity: float,
    speed_weight: float,
    acceleration_weight: float,
    angular_velocity_weight: float,
) -> Tuple[float, Dict[str, float]]:
    speed_term = (float(speed) / max(abs(float(max_speed)), 1e-6)) ** 2
    acceleration_term = (float(acceleration) / max(abs(float(max_acceleration)), 1e-6)) ** 2
    angular_velocity_term = (float(angular_velocity) / max(abs(float(max_angular_velocity)), 1e-6)) ** 2
    total = (
        float(speed_weight) * speed_term
        + float(acceleration_weight) * acceleration_term
        + float(angular_velocity_weight) * angular_velocity_term
    )
    return float(total), {
        "speed": float(speed_term),
        "acceleration": float(acceleration_term),
        "angular_velocity": float(angular_velocity_term),
    }


def ce_transition_reward(
    *,
    before_center_cost: float,
    after_center_cost: float,
    control_cost: float,
    reward_scale: float,
    gamma: float,
    pbrs_enabled: bool,
    pbrs_kappa: float,
    terminal: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """Centroid-energy reward with Potential-Based Reward Shaping (PBRS)."""
    base = -float(after_center_cost) - float(control_cost)
    shaping = 0.0
    if pbrs_enabled:
        phi_before = -float(pbrs_kappa) * float(before_center_cost)
        phi_after = 0.0 if terminal else -float(pbrs_kappa) * float(after_center_cost)
        shaping = float(gamma) * phi_after - phi_before
    scale = float(reward_scale)
    return scale * (base + shaping), {
        "base_center": scale * -float(after_center_cost),
        "control": scale * -float(control_cost),
        "pbrs": scale * shaping,
    }
