# Paper-derived MAADPG specification

Spec version: `maadpg-paper-v1`
Paper SHA-256: `956fb59036dcba020b75bda0bc12d75c50204aaf6cfe49bf53c1eaf48d28091f`

This file records only paper-derived facts. An absent value remains absent here;
the executable choice is recorded separately in `assumptions.yaml`.

## Scenario and dynamics

| Field | Paper statement | Source |
|---|---|---|
| Workspace | Square, 2 km by 2 km | Tables 2-3 / simulation setup |
| Pursuers | Three homogeneous USVs | problem formulation and figures |
| Target | One faster evading USV | problem formulation and Tables 2-3 |
| Obstacles | 0-3 static circular obstacles; radius U(0.10, 0.15) km | Table 2 |
| Pursuer initial velocity | zero | Table 3 |
| Pursuer limits | `vmax=0.010 km/s`, `amax=0.004 km/s^2` | Table 3 |
| Target initial velocity | zero | Table 3 |
| Target limits | `vmax=0.011 km/s`, `amax=0.005 km/s^2` | Table 3 |
| Integrator | velocity update followed by position update using the new velocity | Eq. 12, Eqs. 26-27 |
| Bounds | position is clipped to the map; velocity is norm-projected | Eq. 12 |
| Policy action | normalized world-frame acceleration `[a_x,a_y]`; network output clipped to `[-1,1]^2`, then scaled by `amax` | Eq. 12 and Algorithm 1 |
| Physical constraint | paper separately writes `||a_i,t|| <= amax` | Eq. 13 |

The displayed position update is semi-implicit Euler even though the prose calls
it first-order explicit Euler. The component-wise action box also permits norm
`sqrt(2)*amax`, which conflicts with Eq. 13. Both are registered ambiguities.

## Observation

The pursuer observation in Eq. 24 is:

```text
[x_i/L, y_i/L, vx_i/vmax, vy_i/vmax,
 x_j/L, y_j/L for each of the other two pursuers,
 lidar_1..lidar_16,
 ||p_target-p_i||/d_norm, atan2(dy,dx)]
```

This totals 26 scalars: 4 self + 4 teammate + 16 lidar + 2 target.
Teammate locations are absolute, while target information is relative distance
and bearing. The target observation in Eq. 25 also totals 26 scalars: 4 self,
16 lidar, and relative distance/bearing to each of three pursuers.

The lidar has 16 uniformly spaced rays around the current body heading, detects
map boundaries and obstacle intersections, has maximum range 0.2 km, returns the
maximum range on no hit, and is normalized to `[0,1]`.

The paper later states an actor input width of 28. No two additional features are
defined, so 28 conflicts with Eqs. 24-25.

## CTDE architecture

The MADDPG formulation uses a deterministic local actor `mu_i(o_i)` and a
centralized critic `Q_i(s,a_1,...,a_N)` for each pursuer. The critic observes the
joint state/observation representation and the joint action during training;
actors execute independently from local observations.

The paper describes actor and critic MLPs with 128-unit hidden layers and ReLU,
with a two-dimensional actor output. The text gives `28x128x128x2` for the actor
and `28x128x128x1` for the critic. Table 1 also lists four hidden layers, 128
hidden units, and 512 total hidden neurons. The critic input width 28 conflicts
with the paper's joint-critic definition.

Paper hyperparameters from Table 1:

| Parameter | Value |
|---|---:|
| Discount `gamma` | 0.95 |
| Actor learning rate | 0.0005 |
| Critic learning rate | 0.001 |
| Target soft-update `tau` | 0.01 |
| Replay capacity | 1,000,000 |
| Batch size | 1,024 |
| Hidden units per listed layer | 128 |
| Training episodes | 1,500 |
| Gate denominator stabilizer `lambda` | 1e-6 |
| Gate threshold `beta` | 0.1 |

Optimizer, warm-up, exploration noise, update cadence, number of gradient steps,
gradient clipping, horizon, number of seeds, evaluation rollout count, and
checkpoint selection rule are not stated.

## Capture success

Eq. 18 defines the strict geometric success predicate. All conditions must hold:

1. the target lies in the pursuer convex hull;
2. every pursuer-target radius is within `[rho_min,rho_max]`;
3. the largest angular gap between adjacent pursuers around the target is no
   greater than `Gamma_max`;
4. workspace boundary constraints hold;
5. every pursuer-pursuer separation is at least `d_min`;
6. obstacle clearance is at least `d_obs`.

The paper does not provide numerical values for `rho_min`, `Gamma_max`, `d_min`,
or `d_obs`. Table 3 gives a capture radius of 0.15 km, which is the only disclosed
candidate for `rho_max`.

Eq. 33 instead uses a terminal indicator based on a sum of triangular areas and
maximum pursuer-target distance. The accompanying prose again says the target
must lie inside the pursuer triangle. Eq. 18 and Eq. 33 are therefore not a
complete identical predicate without additional definitions.

## Reward

The paper reward is the sum of five shaped terms and a terminal bonus:

- `R_prog`: pursuer velocity projected toward the target, normalized by `vmax`;
- `R_safe`: `-c_coll` on collision, otherwise
  `(min(lidar)-L_sens)/L_sens`;
- `R_stage`: a three-branch curriculum based on summed triangle areas, summed
  pursuer-target distance, minimum/maximum target distance, and previous-step
  distance;
- `R_sep`: pairwise teammate-separation penalty;
- `R_hit`: soft/hard target-collision penalty;
- terminal `c_succ` when the area/distance success indicator holds.

The displayed `R_stage` branches are:

```text
-sum(d_k)/d_max
    if sum(s_k)>S4 and sum(d_k)>=d_lim and min(d_k)>=d_cap
-(1/3)*log(sum(s_k)-S4+1)
    if sum(s_k)>S4 and (sum(d_k)<d_lim or min(d_k)<d_cap)
exp((sum(d_prev)-sum(d_now))/(3*vmax))
    if sum(s_k)<=S4 and max(d_k)>d_cap
```

The paper does not disclose the reward weights/coefficient values or numerical
values for `S4`, `d_lim`, `d_max`, separation radii, soft/hard hit radii,
`c_coll`, or `c_succ`. It also does not explicitly state the reward if none of
the displayed `R_stage` conditions matches.

## Potential-field module (PFM)

Eq. 19 defines an unnormalized force for pursuer `i`:

```text
F_i = k_t / ||p_t-p_i||^2 * unit(p_t-p_i)
    + sum_o k_o / d_io^2 * unit(p_i-p_o)
    + sum_j!=i k_r / ||p_i-p_j||^2 * unit(p_i-p_j)
```

The obstacle distance definition (centre distance versus signed surface
clearance), the gains `k_t,k_o,k_r`, singularity handling, and obstacle influence
cutoff are not numerically specified. Eq. 20 normalizes `F_i` by
`max(epsilon,||F_i||)` and clips the result element-wise to the action bounds.

## Adaptive difference gate

The prose and Eq. 21 define the relative one-step reward improvement:

```text
Delta_i = (r_i^P - r_i^pi) / (abs(r_i^P) + lambda) * 100%
```

Eq. 22 adopts PFM guidance when `Delta_i >= beta`. The paper says both candidate
actions are evaluated through equivalent one-step environment rollouts. Formal
evaluation later executes the trained actor by ordinary forward propagation
without Peek/PFM guidance.

Algorithm 1 contains swapped/inconsistent labels: its line called “PFM action”
uses the actor expression, its line called “actor action” uses `mu_i(s_i)`, and
its displayed difference reverses Eq. 21. The implementation contract follows
the prose and Eqs. 21-22, not these inconsistent pseudocode labels.

## Target and evaluation

The target is described as a stochastic, nondeterministic deep-RL policy based
on MADDPG, but no target-policy code, architecture, checkpoint, training data,
or checkpoint-selection method is released in the available paper. This is a
material non-recoverable dependency.

The paper reports random initial positions and randomized obstacles. Episodes
end on success, boundary collision, or a maximum step count, but the maximum is
not disclosed. Evaluation is actor-only. Seed lists, evaluation sample count,
and confidence intervals are not disclosed.

## Paper-primary invariants

Regardless of unresolved numeric choices, any Paper-primary implementation must
preserve all of the following:

- exactly three fixed pursuer slots and one target;
- local independent pursuer actors;
- one genuinely joint critic per pursuer;
- normalized continuous world-frame acceleration;
- 16-ray obstacle/boundary lidar;
- no Voronoi, communication-topology, ORCA, action shield, safety projection,
  IQN, SAC entropy, CoCap oracle, observation compression, or curriculum port;
- PFM/gate used only as a training-time action-adoption mechanism;
- actor-only formal validation and test rollouts.
