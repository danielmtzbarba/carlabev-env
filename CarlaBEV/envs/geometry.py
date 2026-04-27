from __future__ import annotations

import numpy as np


RAW_TO_SURFACE_SCALE = 8.0
SURFACE_SIZE_REF = 128.0
VISIBLE_METERS_IN_BEV = 40.0
SURFACE_METERS_PER_PIXEL = VISIBLE_METERS_IN_BEV / SURFACE_SIZE_REF
RAW_METERS_PER_PIXEL = SURFACE_METERS_PER_PIXEL / RAW_TO_SURFACE_SCALE


def as_xy(position) -> tuple[float, float]:
    return float(position[0]), float(position[1])


def raw_to_surface(position):
    x_raw, y_raw = as_xy(position)
    return np.array([x_raw / RAW_TO_SURFACE_SCALE, y_raw / RAW_TO_SURFACE_SCALE], dtype=float)


def surface_to_raw(position):
    x_s, y_s = as_xy(position)
    return np.array([x_s * RAW_TO_SURFACE_SCALE, y_s * RAW_TO_SURFACE_SCALE], dtype=float)


def surface_to_meters(position):
    x_s, y_s = as_xy(position)
    return np.array([x_s * SURFACE_METERS_PER_PIXEL, y_s * SURFACE_METERS_PER_PIXEL], dtype=float)


def meters_to_surface(position):
    x_m, y_m = as_xy(position)
    return np.array([x_m / SURFACE_METERS_PER_PIXEL, y_m / SURFACE_METERS_PER_PIXEL], dtype=float)


def raw_to_meters(position):
    return surface_to_meters(raw_to_surface(position))


def meters_to_raw(position):
    return surface_to_raw(meters_to_surface(position))


def distance_surface_to_meters(distance: float) -> float:
    return float(distance) * SURFACE_METERS_PER_PIXEL


def distance_meters_to_surface(distance: float) -> float:
    return float(distance) / SURFACE_METERS_PER_PIXEL


def speed_mps_to_surface(speed_mps: float) -> float:
    return distance_meters_to_surface(speed_mps)


def speed_surface_to_mps(speed_surface: float) -> float:
    return distance_surface_to_meters(speed_surface)


def route_length_meters(rx, ry) -> float:
    if len(rx) != len(ry):
        raise ValueError("Route coordinates mismatch.")
    total = 0.0
    for i in range(1, len(rx)):
        dx = float(rx[i]) - float(rx[i - 1])
        dy = float(ry[i]) - float(ry[i - 1])
        total += np.hypot(dx, dy)
    return distance_surface_to_meters(total)
