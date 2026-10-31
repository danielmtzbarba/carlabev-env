"""Metric footprints and oriented bodies.

A :class:`Footprint` describes *what* an actor occupies (a box ``length x width``
or a circle) and where the actor's kinematic reference point sits on it. A
:class:`Body` binds a footprint to a pose ``(x_m, y_m, yaw)`` and answers
geometric questions in the metric frame: centre, oriented corners, AABB.

Reference points
----------------
The kinematic state of a vehicle is its **rear axle** (the bicycle model and the
Stanley tracker assume it: the front axle is ``wheelbase`` ahead). The footprint
centre is therefore ``center_offset_m`` ahead of the reference point along the
heading. Pedestrians and legacy squares use the centre as reference.

``oriented=False`` reproduces the legacy behaviour: an axis-aligned square
centred on the state position that ignores yaw (rendering and collision).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Footprint:
    length_m: float
    width_m: float
    shape: str = "box"  # "box" | "circle"
    ref_point: str = "center"  # "center" | "rear_axle"
    center_offset_m: float = 0.0
    """Distance from the reference point to the footprint centre, along the heading."""
    oriented: bool = True
    """False = axis-aligned square ignoring yaw (legacy rendering/collision)."""
    name: str = "box"

    def __post_init__(self):
        if self.length_m <= 0 or self.width_m <= 0:
            raise ValueError("footprint dimensions must be positive")
        if self.shape not in {"box", "circle"}:
            raise ValueError(f"unknown footprint shape {self.shape!r}")
        if self.ref_point not in {"center", "rear_axle"}:
            raise ValueError(f"unknown reference point {self.ref_point!r}")

    @property
    def radius_m(self) -> float:
        """Bounding radius around the centre."""
        if self.shape == "circle":
            return 0.5 * max(self.length_m, self.width_m)
        return 0.5 * math.hypot(self.length_m, self.width_m)

    @property
    def half_length_m(self) -> float:
        return 0.5 * self.length_m

    @property
    def half_width_m(self) -> float:
        return 0.5 * self.width_m

    @staticmethod
    def square(size_m: float, *, name: str = "legacy_square") -> "Footprint":
        """Legacy axis-aligned square centred on the state position."""
        return Footprint(
            size_m,
            size_m,
            shape="box",
            ref_point="center",
            center_offset_m=0.0,
            oriented=False,
            name=name,
        )

    @staticmethod
    def car(
        length_m: float,
        width_m: float,
        wheelbase_m: float,
        *,
        rear_overhang_m: float | None = None,
        name: str = "car",
    ) -> "Footprint":
        """Rear-axle referenced vehicle box. Default rear overhang puts the wheelbase centred in the body."""
        if rear_overhang_m is None:
            rear_overhang_m = 0.5 * (length_m - wheelbase_m)
        return Footprint(
            length_m,
            width_m,
            shape="box",
            ref_point="rear_axle",
            center_offset_m=0.5 * length_m - rear_overhang_m,
            oriented=True,
            name=name,
        )

    @staticmethod
    def circle(radius_m: float, *, name: str = "pedestrian") -> "Footprint":
        return Footprint(
            2.0 * radius_m,
            2.0 * radius_m,
            shape="circle",
            ref_point="center",
            center_offset_m=0.0,
            oriented=True,
            name=name,
        )


@dataclass(frozen=True)
class OrientedBox:
    """Rectangle with centre ``(cx, cy)``, half extents and heading ``yaw`` (radians)."""

    cx: float
    cy: float
    half_length: float
    half_width: float
    yaw: float

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([c, s]), np.array([-s, c])

    def corners(self) -> np.ndarray:
        """4x2 array: front-left, front-right, rear-right, rear-left (in heading frame)."""
        return np.array(self.corner_tuples())

    def corner_tuples(self) -> tuple[tuple[float, float], ...]:
        """Corners as plain tuples (fast path for rendering/collision)."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        hl, hw = self.half_length, self.half_width
        cx, cy = self.cx, self.cy
        return (
            (cx + hl * c - hw * s, cy + hl * s + hw * c),
            (cx + hl * c + hw * s, cy + hl * s - hw * c),
            (cx - hl * c + hw * s, cy - hl * s - hw * c),
            (cx - hl * c - hw * s, cy - hl * s + hw * c),
        )

    def aabb(self) -> tuple[float, float, float, float]:
        pts = self.corners()
        return (
            float(pts[:, 0].min()),
            float(pts[:, 1].min()),
            float(pts[:, 0].max()),
            float(pts[:, 1].max()),
        )

    @property
    def radius(self) -> float:
        return math.hypot(self.half_length, self.half_width)


class Body:
    """A footprint at a pose. All quantities in meters / radians."""

    __slots__ = ("footprint", "x_m", "y_m", "yaw", "v_mps")

    def __init__(
        self,
        footprint: Footprint,
        x_m: float,
        y_m: float,
        yaw: float,
        v_mps: float = 0.0,
    ):
        self.footprint = footprint
        self.x_m = float(x_m)
        self.y_m = float(y_m)
        self.yaw = float(yaw)
        self.v_mps = float(v_mps)

    @property
    def effective_yaw(self) -> float:
        return self.yaw if self.footprint.oriented else 0.0

    def center(self) -> tuple[float, float]:
        off = self.footprint.center_offset_m
        if off == 0.0:
            return self.x_m, self.y_m
        return self.x_m + off * math.cos(self.yaw), self.y_m + off * math.sin(self.yaw)

    def obb(self) -> OrientedBox:
        cx, cy = self.center()
        return OrientedBox(
            cx,
            cy,
            self.footprint.half_length_m,
            self.footprint.half_width_m,
            self.effective_yaw,
        )

    def corners(self) -> np.ndarray:
        return self.obb().corners()

    def corner_tuples(self):
        return self.obb().corner_tuples()

    def aabb(self) -> tuple[float, float, float, float]:
        if self.footprint.shape == "circle":
            cx, cy = self.center()
            r = self.footprint.radius_m
            return cx - r, cy - r, cx + r, cy + r
        return self.obb().aabb()

    def velocity(self) -> tuple[float, float]:
        return self.v_mps * math.cos(self.yaw), self.v_mps * math.sin(self.yaw)

    def projection(self, axis_yaw: float) -> tuple[float, float]:
        """Occupied interval along a world-frame unit axis, in metres.

        Uses the footprint centre and orientation, rather than assuming the
        kinematic reference point is the centre of the occupied shape.
        """
        cx, cy = self.center()
        centre = cx * math.cos(axis_yaw) + cy * math.sin(axis_yaw)
        if self.is_circle:
            extent = self.footprint.radius_m
        else:
            angle = self.effective_yaw - axis_yaw
            extent = (self.footprint.half_length_m * abs(math.cos(angle))
                      + self.footprint.half_width_m * abs(math.sin(angle)))
        return centre - extent, centre + extent

    def heading_vector(self) -> tuple[float, float]:
        return math.cos(self.yaw), math.sin(self.yaw)

    @property
    def is_circle(self) -> bool:
        return self.footprint.shape == "circle"

    def __repr__(self) -> str:
        return f"Body({self.footprint.name}, x={self.x_m:.2f}, y={self.y_m:.2f}, yaw={self.yaw:.2f}, v={self.v_mps:.2f})"
