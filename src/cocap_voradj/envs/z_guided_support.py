"""Single-friend greedy selection using the exact float32 policy tokens."""
from __future__ import annotations

import numpy as np


TIE_EPS = 1e-6


def select_support_target(agent_id, own_z, friend_ids, friend_tokens):
    """Return selection and diagnostics; IDs are only a final stable fallback.

    Tokens contain relative position, observable velocity, distance, bearing, Z.
    Distance and geometry are read from policy precision, never hidden enemies.
    """
    if not np.isfinite(own_z) or not 0 <= own_z <= 1:
        raise ValueError("invalid self Z in support policy snapshot")
    valid = []
    invalid = 0
    for j, token in zip(friend_ids, friend_tokens, strict=True):
        token = np.asarray(token)
        if token.shape != (7,) or not np.all(np.isfinite(token)) or not 0 <= token[6] <= 1 or token[4] < 0:
            invalid += 1
            continue
        valid.append((int(j), token))
    zmax = max([float(own_z)] + [float(t[6]) for _, t in valid])
    highest = [(j, t) for j, t in valid if float(t[6]) >= zmax - TIE_EPS]
    self_best = own_z >= zmax - TIE_EPS
    result = dict(target_id=int(agent_id), alpha=0.0, zmax=zmax,
                  neighbor_zmax=max([0.0] + [float(t[6]) for _, t in valid]),
                  self_tie=bool(self_best and highest), friend_tie=len(highest) > 1,
                  distance_tie=False, id_fallback=False, selected_distance_normalized=0.0,
                  invalid_tokens=invalid, candidate_ids=[j for j, _ in valid])
    if self_best:
        return result
    distance = min(float(t[4]) for _, t in highest)
    nearest = [(j, t) for j, t in highest if float(t[4]) == distance]
    geometry = lambda item: tuple(float(x) for x in item[1][:6])
    best_geometry = min(geometry(item) for item in nearest)
    indistinguishable = [(j, t) for j, t in nearest if geometry((j, t)) == best_geometry]
    chosen, token = min(indistinguishable, key=lambda item: item[0])
    result.update(target_id=chosen, alpha=float(token[6]),
                  distance_tie=len(nearest) > 1, id_fallback=len(indistinguishable) > 1,
                  selected_distance_normalized=distance)
    return result


def ally_progress(own_before, own_after, anchor_before, *, weight=1.0, clip=3.0):
    points = np.asarray([own_before, own_after, anchor_before], dtype=float)
    if not np.all(np.isfinite(points)) or not np.isfinite(weight) or not np.isfinite(clip) or clip <= 0:
        raise ValueError("non-finite support anchor/progress contract")
    progress = float(np.linalg.norm(points[0] - points[2]) - np.linalg.norm(points[1] - points[2]))
    return float(weight * np.clip(progress, -clip, clip)), progress
