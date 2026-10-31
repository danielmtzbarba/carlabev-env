"""Geometry frames and metric conversions.

Three frames exist (see docs/geometry_and_metric_frame.md):

* **raw** asset coordinates (1024-scale graph/anchor assets),
* **surface** runtime coordinates (the raster the simulation runs on),
* **metric** coordinates in meters (SI; used by dynamics and reward).

:class:`MetricFrame` is the single place where the relation between them is
defined. The surface frame is defined for exactly one native raster size
(``surface_size = 128``): every map asset, route, scenario anchor and authored
scene is expressed at that scale, and the field of view of ``surface_size``
pixels spans ``visible_meters`` (40 m), giving ``0.3125 m/px``. Other raster
sizes are *not* a different world scale; the runtime refuses them until render
resolution is decoupled from the surface frame (see
docs/simulation_clock_and_units.md, U-02).

Module-level constants and functions are thin delegates to
:data:`DEFAULT_METRIC_FRAME` kept for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricFrame:
    """Declares the surface raster scale and its metric interpretation."""

    surface_size: int = 128
    """Native raster size (px) the surface frame is defined on."""

    raw_size: int = 1024
    """Raster size (px) of the raw asset frame (graph assets, anchors)."""

    visible_meters: float = 40.0
    """Physical extent (m) spanned by ``surface_size`` pixels (the BEV field of view)."""

    def __post_init__(self):
        if self.surface_size <= 0 or self.raw_size <= 0 or self.visible_meters <= 0:
            raise ValueError("MetricFrame parameters must be positive")

    # --- scales -----------------------------------------------------------
    @property
    def meters_per_pixel(self) -> float:
        return self.visible_meters / self.surface_size

    @property
    def pixels_per_meter(self) -> float:
        return self.surface_size / self.visible_meters

    @property
    def raw_to_surface_scale(self) -> float:
        return self.raw_size / self.surface_size

    @property
    def raw_meters_per_pixel(self) -> float:
        return self.meters_per_pixel / self.raw_to_surface_scale

    # --- validation -------------------------------------------------------
    @classmethod
    def for_size(cls, size: int, *, raw_size: int = 1024, visible_meters: float = 40.0) -> "MetricFrame":
        """Metric frame for a surface raster of ``size`` px (FOV always spans ``visible_meters``)."""
        return cls(surface_size=int(size), raw_size=raw_size, visible_meters=visible_meters)

    @staticmethod
    def supports_size(size: int) -> bool:
        return int(size) in SUPPORTED_SURFACE_SIZES

    @classmethod
    def require_size(cls, size: int) -> None:
        if not cls.supports_size(size):
            raise ValueError(
                f"size={size} is not supported: expected one of {SUPPORTED_SURFACE_SIZES} "
                "(raster assets exist for these surface scales). "
                "See docs/simulation_clock_and_units.md (F-01)."
            )

    # --- cross-frame surface conversions ----------------------------------
    LEGACY_SURFACE_SIZE = 128
    """Surface scale every v1/v2 authored asset and legacy anchor is expressed in."""

    def surface_scale_from(self, source_size: int) -> float:
        """Factor converting surface px of a ``source_size`` raster into this frame."""
        return self.surface_size / float(source_size)

    def rescale_surface_from(self, value, source_size: int):
        """Convert a surface-px scalar/array expressed at ``source_size`` into this frame."""
        factor = self.surface_scale_from(source_size)
        if factor == 1.0:
            return value
        if isinstance(value, (list, tuple)):
            return type(value)(float(v) * factor for v in value)
        return np.asarray(value, dtype=float) * factor if hasattr(value, "__len__") else float(value) * factor

    # --- position conversions ---------------------------------------------
    @staticmethod
    def _as_xy(position) -> tuple[float, float]:
        return float(position[0]), float(position[1])

    def raw_to_surface(self, position):
        x_raw, y_raw = self._as_xy(position)
        s = self.raw_to_surface_scale
        return np.array([x_raw / s, y_raw / s], dtype=float)

    def surface_to_raw(self, position):
        x_s, y_s = self._as_xy(position)
        s = self.raw_to_surface_scale
        return np.array([x_s * s, y_s * s], dtype=float)

    def surface_to_meters(self, position):
        x_s, y_s = self._as_xy(position)
        mpp = self.meters_per_pixel
        return np.array([x_s * mpp, y_s * mpp], dtype=float)

    def meters_to_surface(self, position):
        x_m, y_m = self._as_xy(position)
        mpp = self.meters_per_pixel
        return np.array([x_m / mpp, y_m / mpp], dtype=float)

    def raw_to_meters(self, position):
        return self.surface_to_meters(self.raw_to_surface(position))

    def meters_to_raw(self, position):
        return self.surface_to_raw(self.meters_to_surface(position))

    # --- scalar conversions -----------------------------------------------
    def distance_surface_to_meters(self, distance: float) -> float:
        return float(distance) * self.meters_per_pixel

    def distance_meters_to_surface(self, distance: float) -> float:
        return float(distance) / self.meters_per_pixel

    def speed_mps_to_surface(self, speed_mps: float) -> float:
        return self.distance_meters_to_surface(speed_mps)

    def speed_surface_to_mps(self, speed_surface: float) -> float:
        return self.distance_surface_to_meters(speed_surface)

    def route_length_meters(self, rx, ry) -> float:
        if len(rx) != len(ry):
            raise ValueError("Route coordinates mismatch.")
        total = 0.0
        for i in range(1, len(rx)):
            dx = float(rx[i]) - float(rx[i - 1])
            dy = float(ry[i]) - float(ry[i - 1])
            total += np.hypot(dx, dy)
        return self.distance_surface_to_meters(total)

    def map_extent_meters(self, width_px: int, height_px: int) -> tuple[float, float]:
        """Physical extent (m) of a surface-frame raster."""
        return width_px * self.meters_per_pixel, height_px * self.meters_per_pixel


SUPPORTED_SURFACE_SIZES: tuple[int, ...] = (64, 128, 256, 512, 1024)
"""Raster sizes the runtime supports: each must have ``<Map>-<size>-sem/rgb.png`` assets.

Every coordinate source is frame-aware (F-01): graph assets are raw-frame,
procedural anchors are raw-frame, authored scenes and scene-library payloads
carry a frame tag and are rescaled on load. ``128`` is the reference frame all
legacy data are expressed in.
"""

DEFAULT_METRIC_FRAME = MetricFrame()

# Backward-compatible constants (all derived from the default metric frame).
RAW_TO_SURFACE_SCALE = DEFAULT_METRIC_FRAME.raw_to_surface_scale
SURFACE_SIZE_REF = float(DEFAULT_METRIC_FRAME.surface_size)
VISIBLE_METERS_IN_BEV = DEFAULT_METRIC_FRAME.visible_meters
SURFACE_METERS_PER_PIXEL = DEFAULT_METRIC_FRAME.meters_per_pixel
RAW_METERS_PER_PIXEL = DEFAULT_METRIC_FRAME.raw_meters_per_pixel


def as_xy(position) -> tuple[float, float]:
    return MetricFrame._as_xy(position)


def raw_to_surface(position):
    return DEFAULT_METRIC_FRAME.raw_to_surface(position)


def surface_to_raw(position):
    return DEFAULT_METRIC_FRAME.surface_to_raw(position)


def surface_to_meters(position):
    return DEFAULT_METRIC_FRAME.surface_to_meters(position)


def meters_to_surface(position):
    return DEFAULT_METRIC_FRAME.meters_to_surface(position)


def raw_to_meters(position):
    return DEFAULT_METRIC_FRAME.raw_to_meters(position)


def meters_to_raw(position):
    return DEFAULT_METRIC_FRAME.meters_to_raw(position)


def distance_surface_to_meters(distance: float) -> float:
    return DEFAULT_METRIC_FRAME.distance_surface_to_meters(distance)


def distance_meters_to_surface(distance: float) -> float:
    return DEFAULT_METRIC_FRAME.distance_meters_to_surface(distance)


def speed_mps_to_surface(speed_mps: float) -> float:
    return DEFAULT_METRIC_FRAME.speed_mps_to_surface(speed_mps)


def speed_surface_to_mps(speed_surface: float) -> float:
    return DEFAULT_METRIC_FRAME.speed_surface_to_mps(speed_surface)


def route_length_meters(rx, ry) -> float:
    return DEFAULT_METRIC_FRAME.route_length_meters(rx, ry)
