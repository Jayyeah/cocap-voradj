"""Paper Eq. 19-20 potential-field guidance in normalized action space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dynamics import canonical_normalized_action
from .geometry import obstacle_surface_clearance, unit_vector
from .guidance_config import PotentialFieldConfig


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PotentialFieldResult:
    actions: FloatArray
    target_components: FloatArray
    obstacle_components: FloatArray
    teammate_components: FloatArray
    resultant_forces: FloatArray


def _inverse_square(distance: float, floor: float) -> float:
    return 1.0 / max(distance, floor) ** 2


def potential_field_actions(
    pursuer_positions: ArrayLike,
    target_position: ArrayLike,
    obstacles: Iterable[tuple[ArrayLike, float]],
    config: PotentialFieldConfig | None = None,
) -> PotentialFieldResult:
    values = config or PotentialFieldConfig()
    values.validate()
    pursuers = np.asarray(pursuer_positions, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    if pursuers.shape != (3, 2) or target.shape != (2,):
        raise ValueError("PFM requires pursuers (3,2) and target (2,)")
    obstacle_list = [
        (np.asarray(centre, dtype=np.float64), float(radius))
        for centre, radius in obstacles
    ]
    target_components = np.zeros((3, 2), dtype=np.float64)
    obstacle_components = np.zeros((3, 2), dtype=np.float64)
    teammate_components = np.zeros((3, 2), dtype=np.float64)
    for index, position in enumerate(pursuers):
        target_offset = target - position
        target_distance = float(np.linalg.norm(target_offset))
        target_components[index] = (
            values.target_gain
            * _inverse_square(
                target_distance, values.inverse_square_distance_floor
            )
            * unit_vector(target_offset)
        )
        for centre, radius in obstacle_list:
            clearance = obstacle_surface_clearance(position, centre, radius)
            if clearance >= values.obstacle_influence_cutoff:
                continue
            obstacle_components[index] += (
                values.obstacle_gain
                * _inverse_square(
                    clearance, values.inverse_square_distance_floor
                )
                * unit_vector(position - centre)
            )
        for other, teammate in enumerate(pursuers):
            if other == index:
                continue
            offset = position - teammate
            distance = float(np.linalg.norm(offset))
            teammate_components[index] += (
                values.teammate_gain
                * _inverse_square(distance, values.inverse_square_distance_floor)
                * unit_vector(offset)
            )
    resultant = target_components + obstacle_components + teammate_components
    actions = np.stack(
        [
            canonical_normalized_action(
                force / max(values.epsilon, float(np.linalg.norm(force)))
            )
            for force in resultant
        ]
    )
    return PotentialFieldResult(
        actions=actions,
        target_components=target_components,
        obstacle_components=obstacle_components,
        teammate_components=teammate_components,
        resultant_forces=resultant,
    )
