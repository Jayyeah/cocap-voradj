# 原排序、截断与新候选构造代码

以下排序/截断代码来自固定基线提交，当前保持相同语义。

```python
        friend_ids = [nk[1] for nk in adjacency if nk[0] == "pursuer" and not self.pursuers[nk[1]].deactivated]
        def friend_sort(j: int) -> Tuple[int, int, float]:
            other_key = ("pursuer", j)
            return (
                0 if self._effective_is_pursuing(j, self._has_enemy_neighbor(data, other_key)) else 1,
                -self._shared_count(data, key, other_key),
                float(np.linalg.norm(self._position(self.pursuers[j]) - self._position(pursuer))),
            )
        friend_ordering_mode = str(
            self.per_cfg.get(
                "friend_ordering_mode",
                "role_then_physical" if include_is_pursuing else "physical_only",
            )
        ).strip().lower()
        if friend_ordering_mode not in {"role_then_physical", "physical_only"}:
            raise ValueError(
                "perception.friend_ordering_mode must be role_then_physical or physical_only"
            )
        if include_is_pursuing and friend_ordering_mode == "role_then_physical":
            friend_ids = sorted(friend_ids, key=friend_sort)
        else:
            # Physical-only ordering can be required even when a role/z token
            # is present so evidence cannot affect truncation priority.
            def role_free_friend_sort(j: int) -> Tuple[float, ...]:
                other = self.pursuers[j]
                if world_frame:
                    pos_r = self._position(other) - self._position(pursuer)
                    vel_r = np.asarray(other.velocity, dtype=float)
                else:
                    pos_r = self._robot_frame(pursuer, self._position(other), False)
                    vel_r = self._robot_frame(pursuer, other.velocity, True)
                dist = float(np.linalg.norm(pos_r))
                ang = float(np.arctan2(pos_r[1], pos_r[0]))
                return (
                    float(pos_r[0]),
                    float(pos_r[1]),
                    float(vel_r[0]),
                    float(vel_r[1]),
                    dist,
                    ang,
                    float(j),
                )
            friend_ids = sorted(friend_ids, key=role_free_friend_sort)
        pursuer_feats: List[List[float]] = []
        for j in friend_ids[:max_p]:
            other = self.pursuers[j]
            if world_frame:
                pos_r = self._position(other) - self._position(pursuer)
                vel_r = np.asarray(other.velocity, dtype=float)
            else:
                pos_r = self._robot_frame(pursuer, self._position(other), False)
                vel_r = self._robot_frame(pursuer, other.velocity, True)
            dist = float(np.linalg.norm(pos_r))
            ang = float(np.arctan2(pos_r[1], pos_r[0]))
            friend_feature = [
                float(pos_r[0] / distance_scale),
                float(pos_r[1] / distance_scale),
                float(vel_r[0]),
                float(vel_r[1]),
                float(dist / distance_scale),
                ang,
            ]
            if include_is_pursuing:
                friend_feature.append(
                    float(self._effective_is_pursuing(j, self._has_enemy_neighbor(data, ("pursuer", j))))
                )
            if include_z_state:
                friend_feature.append(float(self.z_state[j]) if j < len(self.z_state) else 0.0)
            pursuer_feats.append(friend_feature)

```

策略真实 token 的内部映射（不进入 policy 输入）：

```python
        if include_z_state and not include_is_pursuing:
            # Internal mapping only: do not add IDs or change policy tensors.
            self._support_policy_snapshots[idx] = {
                "friend_ids": tuple(int(j) for j in friend_ids[:max_p]),
                "tokens": np.asarray(pursuer_feats, dtype=np.float32).reshape(-1, friend_dim).copy(),
                "self_z": float(np.float32(self_feat[-1])),
            }
```

最大 Z、自身优先、最近距离、几何与 ID fallback，以及冻结锚点进展的完整实现：

```python
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
```
