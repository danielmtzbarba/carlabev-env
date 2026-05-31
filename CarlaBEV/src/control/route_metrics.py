from __future__ import annotations

import numpy as np

from CarlaBEV.envs.geometry import SURFACE_METERS_PER_PIXEL


def compute_smoothed_route_direction_fractions(
    cx,
    cy,
    cyaw,
    *,
    turn_rate_thresh_rad_per_m: float = 0.12,
) -> dict[str, float]:
    """
    Compute straight/left/right route fractions from a smoothed hero route.

    Fractions are measured by route arc length, not waypoint count.
    Positive heading-rate is treated as a left turn and negative as a right turn.
    """
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    cyaw = np.unwrap(np.asarray(cyaw, dtype=float))

    if cx.size < 2 or cy.size < 2 or cyaw.size < 2:
        return {
            "straight_fraction": 1.0,
            "left_turn_fraction": 0.0,
            "right_turn_fraction": 0.0,
        }

    dx = np.diff(cx)
    dy = np.diff(cy)
    ds_m = np.hypot(dx, dy) * SURFACE_METERS_PER_PIXEL

    valid = ds_m > 1e-6
    if not np.any(valid):
        return {
            "straight_fraction": 1.0,
            "left_turn_fraction": 0.0,
            "right_turn_fraction": 0.0,
        }

    dtheta = np.diff(cyaw)
    dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi

    ds_valid = ds_m[valid]
    dtheta_valid = dtheta[valid]
    turn_rate = dtheta_valid / ds_valid
    turning = np.abs(turn_rate) > turn_rate_thresh_rad_per_m

    total_length_m = float(ds_valid.sum())
    if total_length_m <= 1e-9:
        return {
            "straight_fraction": 1.0,
            "left_turn_fraction": 0.0,
            "right_turn_fraction": 0.0,
        }

    straight_length_m = float(ds_valid[~turning].sum())
    left_length_m = float(ds_valid[turning & (turn_rate > 0.0)].sum())
    right_length_m = float(ds_valid[turning & (turn_rate < 0.0)].sum())

    return {
        "straight_fraction": straight_length_m / total_length_m,
        "left_turn_fraction": left_length_m / total_length_m,
        "right_turn_fraction": right_length_m / total_length_m,
    }
