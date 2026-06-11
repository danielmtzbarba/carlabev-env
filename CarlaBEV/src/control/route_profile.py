from __future__ import annotations

from typing import Any

import numpy as np

from CarlaBEV.envs.geometry import SURFACE_METERS_PER_PIXEL
from CarlaBEV.src.control.utils import smooth_and_compute


ROUTE_PROFILES = {
    "any",
    "mostly_straight",
    "single_left",
    "single_right",
    "multi_turn",
    "mixed",
}


def _normalize_turn_segments(
    turn_labels: np.ndarray,
    ds_valid: np.ndarray,
    *,
    min_turn_segment_m: float,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current_sign = 0
    current_length = 0.0

    for sign, seg_length in zip(turn_labels, ds_valid):
        sign = int(sign)
        if sign == 0:
            if current_sign != 0 and current_length >= min_turn_segment_m:
                segments.append({"sign": current_sign, "length_m": current_length})
            current_sign = 0
            current_length = 0.0
            continue

        if sign == current_sign:
            current_length += float(seg_length)
            continue

        if current_sign != 0 and current_length >= min_turn_segment_m:
            segments.append({"sign": current_sign, "length_m": current_length})
        current_sign = sign
        current_length = float(seg_length)

    if current_sign != 0 and current_length >= min_turn_segment_m:
        segments.append({"sign": current_sign, "length_m": current_length})

    return segments


def compute_route_profile_metrics(
    ax,
    ay,
    *,
    turn_rate_thresh_rad_per_m: float = 0.12,
    min_turn_segment_m: float = 4.0,
) -> dict[str, Any]:
    cx, cy, cyaw, _, _ = smooth_and_compute(ax, ay, window=11, poly=3)

    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    cyaw = np.unwrap(np.asarray(cyaw, dtype=float))

    if cx.size < 2 or cy.size < 2 or cyaw.size < 2:
        return {
            "straight_fraction": 1.0,
            "left_turn_fraction": 0.0,
            "right_turn_fraction": 0.0,
            "turn_count": 0,
            "has_left_turn": False,
            "has_right_turn": False,
            "intersection_like": False,
            "route_profile": "mostly_straight",
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
            "turn_count": 0,
            "has_left_turn": False,
            "has_right_turn": False,
            "intersection_like": False,
            "route_profile": "mostly_straight",
        }

    dtheta = np.diff(cyaw)
    dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi

    ds_valid = ds_m[valid]
    dtheta_valid = dtheta[valid]
    turn_rate = dtheta_valid / ds_valid
    turn_labels = np.where(
        turn_rate > turn_rate_thresh_rad_per_m,
        1,
        np.where(turn_rate < -turn_rate_thresh_rad_per_m, -1, 0),
    )

    total_length_m = float(ds_valid.sum())
    if total_length_m <= 1e-9:
        return {
            "straight_fraction": 1.0,
            "left_turn_fraction": 0.0,
            "right_turn_fraction": 0.0,
            "turn_count": 0,
            "has_left_turn": False,
            "has_right_turn": False,
            "intersection_like": False,
            "route_profile": "mostly_straight",
        }

    straight_length_m = float(ds_valid[turn_labels == 0].sum())
    left_length_m = float(ds_valid[turn_labels == 1].sum())
    right_length_m = float(ds_valid[turn_labels == -1].sum())
    turn_segments = _normalize_turn_segments(
        turn_labels,
        ds_valid,
        min_turn_segment_m=min_turn_segment_m,
    )
    turn_count = len(turn_segments)
    has_left_turn = any(segment["sign"] > 0 for segment in turn_segments)
    has_right_turn = any(segment["sign"] < 0 for segment in turn_segments)
    intersection_like = turn_count >= 2 or (has_left_turn and has_right_turn)

    straight_fraction = straight_length_m / total_length_m
    left_fraction = left_length_m / total_length_m
    right_fraction = right_length_m / total_length_m

    if turn_count == 0 or straight_fraction >= 0.9:
        route_profile = "mostly_straight"
    elif turn_count == 1 and left_fraction >= right_fraction:
        route_profile = "single_left"
    elif turn_count == 1 and right_fraction > left_fraction:
        route_profile = "single_right"
    elif turn_count >= 2:
        route_profile = "multi_turn"
    else:
        route_profile = "mixed"

    return {
        "straight_fraction": straight_fraction,
        "left_turn_fraction": left_fraction,
        "right_turn_fraction": right_fraction,
        "turn_count": turn_count,
        "has_left_turn": has_left_turn,
        "has_right_turn": has_right_turn,
        "intersection_like": intersection_like,
        "route_profile": route_profile,
    }


def matches_route_profile(
    metrics: dict[str, Any],
    *,
    route_profile: str | None = None,
    min_turns: int | None = None,
    max_turns: int | None = None,
    intersection_required: bool | None = None,
) -> bool:
    if route_profile is not None and route_profile != "any":
        if metrics.get("route_profile") != route_profile:
            return False
    turn_count = int(metrics.get("turn_count", 0))
    if min_turns is not None and turn_count < min_turns:
        return False
    if max_turns is not None and turn_count > max_turns:
        return False
    if intersection_required is True and not bool(metrics.get("intersection_like", False)):
        return False
    if intersection_required is False and bool(metrics.get("intersection_like", False)):
        return False
    return True
