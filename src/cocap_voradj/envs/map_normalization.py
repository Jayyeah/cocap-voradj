"""Auditable map scales; the R contract uses the final environment world size."""
from __future__ import annotations
import math


def uses_map_side_length(env):
    mode = (env.config.get("normalization", {}) or {}).get("map_scale", "legacy_diagonal")
    if mode not in {"legacy_diagonal", "side_length_r"}:
        raise ValueError(f"unknown map normalization contract: {mode}")
    return mode == "side_length_r"


def legacy_bounds(env):
    if hasattr(env, "_bounds"):
        return env._bounds()
    return env.inner_x_min, env.inner_x_max, env.inner_y_min, env.inner_y_max


def legacy_diagonal(env, bounds=None):
    xl, xr, yb, yt = legacy_bounds(env) if bounds is None else bounds
    return max(math.hypot(xr-xl, yt-yb), 1e-6)


def get_map_normalization_scale(env, *, legacy_geometry_bounds=None):
    """Metres. In the new contract R=env.width=env.height, never the inner area."""
    if not uses_map_side_length(env):
        return legacy_diagonal(env, legacy_geometry_bounds)
    width, height = float(env.width), float(env.height)
    if not all(math.isfinite(x) and x > 0 for x in (width, height)):
        raise ValueError("R normalization requires finite positive world dimensions")
    if not math.isclose(width, height, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError(f"R normalization requires a square world; got {width} x {height}")
    return width


def physical_distance_threshold_factor(env, *, legacy_geometry_bounds=None):
    """Convert a legacy dimensionless distance threshold, preserving metres."""
    if not uses_map_side_length(env):
        return 1.0
    contract = (env.config.get("normalization", {}) or {}).get("success_threshold_contract")
    if contract != "preserve_legacy_physical":
        raise ValueError("R normalization requires explicit preserve_legacy_physical thresholds")
    return legacy_diagonal(env, legacy_geometry_bounds) / get_map_normalization_scale(env)


def map_normalization_facts(env):
    scale = get_map_normalization_scale(env)
    return dict(contract="side_length_r" if uses_map_side_length(env) else "legacy_diagonal",
                source="resolved env.width/env.height" if uses_map_side_length(env) else "legacy geometry bounds",
                units="metres", width=float(env.width), height=float(env.height),
                normalization_scale=scale, bounds=list(legacy_bounds(env)),
                distance_threshold_factor=physical_distance_threshold_factor(env))
