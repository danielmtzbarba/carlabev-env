"""Collision system: metric-frame overlap, surface gap, shape TTC and contact classification.

All computations are in meters on :class:`~CarlaBEV.src.geometry.body.Body`
objects — never on render rectangles. Supported shapes are oriented boxes and
circles (pedestrians):

* **overlap** — separating-axis test (box–box), closest-point test (circle–box),
  centre distance (circle–circle);
* **gap** — surface-to-surface distance (0 when overlapping);
* **TTC** — swept separating-axis test: time until the two shapes first overlap
  when moving at their current velocities (headings held constant), ``inf`` when
  they never meet;
* **classification** — ``rear_end`` (ego hits the actor from behind),
  ``rear_ended`` (actor hits the ego from behind), ``head_on``, ``side``,
  ``pedestrian``.

The primitives are written with plain ``math`` on 2-tuples: for 4-vertex
polygons this is an order of magnitude faster than NumPy calls (measured in
the throughput benchmark). Broad phase: bounding-radius / AABB rejection,
sufficient for this simulator's actor counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from CarlaBEV.src.geometry.body import Body, OrientedBox

_EPS = 1e-9


@dataclass
class ContactInfo:
    other: object
    kind: str  # vehicle | pedestrian | ...
    overlapping: bool
    gap_m: float
    center_distance_m: float
    bearing_rad: float
    """Bearing of the other body's centre in the ego frame (0 = straight ahead)."""
    heading_delta_rad: float
    """Other heading minus ego heading, wrapped to [-pi, pi)."""
    closing_speed_mps: float
    """Rate at which the centre distance shrinks (positive = approaching)."""
    ttc_s: float
    classification: str
    contact_normal: tuple[float, float] | None = None
    extra: dict = field(default_factory=dict)

    def to_info(self) -> dict:
        return {
            "type": self.kind,
            "overlapping": self.overlapping,
            "gap_m": self.gap_m,
            "center_distance_m": self.center_distance_m,
            "bearing_rad": self.bearing_rad,
            "heading_delta_rad": self.heading_delta_rad,
            "closing_speed_mps": self.closing_speed_mps,
            "ttc_s": self.ttc_s,
            "classification": self.classification,
        }


# --------------------------------------------------------------------------- primitives
def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _box_data(box: OrientedBox):
    """Corners (4 tuples) and unit axes (2 tuples) of an oriented box."""
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    ux, uy = c, s
    vx, vy = -s, c
    hl, hw = box.half_length, box.half_width
    cx, cy = box.cx, box.cy
    corners = (
        (cx + hl * ux + hw * vx, cy + hl * uy + hw * vy),
        (cx + hl * ux - hw * vx, cy + hl * uy - hw * vy),
        (cx - hl * ux - hw * vx, cy - hl * uy - hw * vy),
        (cx - hl * ux + hw * vx, cy - hl * uy + hw * vy),
    )
    return corners, ((ux, uy), (vx, vy))


def _project(corners, axis):
    ax, ay = axis
    lo = hi = corners[0][0] * ax + corners[0][1] * ay
    for x, y in corners[1:]:
        d = x * ax + y * ay
        if d < lo:
            lo = d
        elif d > hi:
            hi = d
    return lo, hi


def box_box_overlap(
    a: OrientedBox, b: OrientedBox
) -> tuple[bool, float, tuple[float, float] | None]:
    """SAT. Returns (overlapping, penetration_depth, minimum-translation axis)."""
    pa, axes_a = _box_data(a)
    pb, axes_b = _box_data(b)
    best_depth, best_axis = math.inf, None
    for axis in (*axes_a, *axes_b):
        amin, amax = _project(pa, axis)
        bmin, bmax = _project(pb, axis)
        depth = min(amax, bmax) - max(amin, bmin)
        if depth < 0.0:
            return False, 0.0, None
        if depth < best_depth:
            best_depth, best_axis = depth, axis
    return True, float(best_depth), best_axis


def _segment_point_distance(px, py, ax, ay, bx, by) -> float:
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom < _EPS:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def _point_in_box(px, py, box: OrientedBox) -> bool:
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    dx, dy = px - box.cx, py - box.cy
    return (
        abs(dx * c + dy * s) <= box.half_length + _EPS
        and abs(-dx * s + dy * c) <= box.half_width + _EPS
    )


def _polygon_distance(pa, pb) -> float:
    best = math.inf
    for pts, other in ((pa, pb), (pb, pa)):
        for px, py in pts:
            for i in range(4):
                ax, ay = other[i]
                bx, by = other[(i + 1) % 4]
                d = _segment_point_distance(px, py, ax, ay, bx, by)
                if d < best:
                    best = d
    return best


def box_box_gap(a: OrientedBox, b: OrientedBox) -> float:
    """Surface-to-surface distance between two convex boxes (0 when overlapping)."""
    if box_box_overlap(a, b)[0]:
        return 0.0
    pa, _ = _box_data(a)
    pb, _ = _box_data(b)
    return float(_polygon_distance(pa, pb))


def circle_box_gap(center, radius: float, box: OrientedBox) -> float:
    px, py = float(center[0]), float(center[1])
    if _point_in_box(px, py, box):
        return 0.0
    pts, _ = _box_data(box)
    dist = min(
        _segment_point_distance(
            px, py, pts[i][0], pts[i][1], pts[(i + 1) % 4][0], pts[(i + 1) % 4][1]
        )
        for i in range(4)
    )
    return float(max(0.0, dist - radius))


def gap_between(a: Body, b: Body) -> float:
    if a.is_circle and b.is_circle:
        ca, cb = a.center(), b.center()
        return max(0.0, math.dist(ca, cb) - a.footprint.radius_m - b.footprint.radius_m)
    if a.is_circle:
        return circle_box_gap(a.center(), a.footprint.radius_m, b.obb())
    if b.is_circle:
        return circle_box_gap(b.center(), b.footprint.radius_m, a.obb())
    return box_box_gap(a.obb(), b.obb())


def bodies_overlap(a: Body, b: Body) -> bool:
    # Cheap bounding-radius rejection first.
    ca, cb = a.center(), b.center()
    if math.dist(ca, cb) > a.footprint.radius_m + b.footprint.radius_m + _EPS:
        return False
    return gap_between(a, b) <= _EPS


def swept_overlap_time(a: Body, b: Body, horizon_s: float = 60.0) -> float:
    """Time until ``a`` and ``b`` first overlap at constant velocities and headings.

    Swept separating-axis test: on every candidate axis the projected intervals
    move with the relative velocity; the shapes overlap when all axes overlap
    simultaneously. Returns 0 if already overlapping, ``inf`` if never (within
    ``horizon_s``). Circles are projected exactly on every axis (exact for
    circle–circle, conservative for circle–box).
    """
    if bodies_overlap(a, b):
        return 0.0
    va, vb = a.velocity(), b.velocity()
    rvx, rvy = vb[0] - va[0], vb[1] - va[1]  # velocity of b relative to a
    ca, cb = a.center(), b.center()
    if a.is_circle and b.is_circle:
        dx, dy = cb[0] - ca[0], cb[1] - ca[1]
        r = a.footprint.radius_m + b.footprint.radius_m
        qa = rvx * rvx + rvy * rvy
        if qa < _EPS:
            return math.inf
        qb = 2.0 * (dx * rvx + dy * rvy)
        qc = dx * dx + dy * dy - r * r
        disc = qb * qb - 4 * qa * qc
        if disc < 0:
            return math.inf
        t = (-qb - math.sqrt(disc)) / (2 * qa)
        return float(t) if 0.0 <= t <= horizon_s else math.inf
    axes = []
    pa = pb = None
    if not a.is_circle:
        pa, axes_a = _box_data(a.obb())
        axes.extend(axes_a)
    if not b.is_circle:
        pb, axes_b = _box_data(b.obb())
        axes.extend(axes_b)
    t_enter, t_exit = 0.0, math.inf
    for axis in axes:
        ax, ay = axis
        if a.is_circle:
            d = ca[0] * ax + ca[1] * ay
            amin, amax = d - a.footprint.radius_m, d + a.footprint.radius_m
        else:
            amin, amax = _project(pa, axis)
        if b.is_circle:
            d = cb[0] * ax + cb[1] * ay
            bmin, bmax = d - b.footprint.radius_m, d + b.footprint.radius_m
        else:
            bmin, bmax = _project(pb, axis)
        v = rvx * ax + rvy * ay
        if abs(v) < _EPS:
            if bmax < amin or amax < bmin:
                return math.inf
            continue
        t0 = (amin - bmax) / v
        t1 = (amax - bmin) / v
        if t0 > t1:
            t0, t1 = t1, t0
        if t0 > t_enter:
            t_enter = t0
        if t1 < t_exit:
            t_exit = t1
        if t_enter > t_exit:
            return math.inf
    return float(t_enter) if t_enter <= horizon_s else math.inf


def classify_contact(ego: Body, other: Body, other_kind: str) -> str:
    if other_kind == "pedestrian" or other.is_circle:
        return "pedestrian"
    ecx, ecy = ego.center()
    ocx, ocy = other.center()
    bearing = _wrap(math.atan2(ocy - ecy, ocx - ecx) - ego.yaw)
    heading_delta = abs(_wrap(other.yaw - ego.yaw))
    if heading_delta > 3 * math.pi / 4:
        return "head_on"
    if heading_delta <= math.pi / 4:
        if abs(bearing) <= math.pi / 4:
            return "rear_end"  # other ahead, same direction: ego hits its rear
        if abs(bearing) >= 3 * math.pi / 4:
            return "rear_ended"  # other behind, same direction
    return "side"


# --------------------------------------------------------------------------- system
class CollisionSystem:
    """Computes ego–actor contacts on bodies in the metric frame."""

    def __init__(self, *, proximity_radius_m: float, ttc_horizon_s: float = 60.0):
        self.proximity_radius_m = float(proximity_radius_m)
        self.ttc_horizon_s = float(ttc_horizon_s)

    @staticmethod
    def candidates(ego: Body, bodies, *, margin_m: float) -> list:
        """AABB broad phase: bodies whose AABB is within ``margin_m`` of the ego AABB."""
        ex0, ey0, ex1, ey1 = ego.aabb()
        out = []
        for item in bodies:
            body = item[1] if isinstance(item, tuple) else item
            bx0, by0, bx1, by1 = body.aabb()
            if (
                bx0 > ex1 + margin_m
                or bx1 < ex0 - margin_m
                or by0 > ey1 + margin_m
                or by1 < ey0 - margin_m
            ):
                continue
            out.append(item)
        return out

    def contact(
        self,
        ego: Body,
        other: Body,
        kind: str,
        owner=None,
        *,
        detailed: bool | None = None,
    ) -> ContactInfo:
        """Contact between the ego and another body.

        Gap and TTC are exact only when the other body is within the proximity
        radius (plus bounding radii) — that is where they are consumed
        (``actors_state``, reward). Farther bodies get a cheap lower-bound gap
        and ``ttc = inf`` unless ``detailed=True``.
        """
        ecx, ecy = ego.center()
        ocx, ocy = other.center()
        dx, dy = ocx - ecx, ocy - ecy
        center_distance = math.hypot(dx, dy)
        radii = ego.footprint.radius_m + other.footprint.radius_m
        if detailed is None:
            detailed = center_distance <= self.proximity_radius_m + radii
        bearing = _wrap(math.atan2(dy, dx) - ego.yaw) if center_distance > _EPS else 0.0
        va, vb = ego.velocity(), other.velocity()
        closing = (
            -((vb[0] - va[0]) * dx + (vb[1] - va[1]) * dy) / center_distance
            if center_distance > _EPS
            else 0.0
        )
        if detailed:
            gap = gap_between(ego, other)
            ttc = swept_overlap_time(ego, other, self.ttc_horizon_s)
        else:
            gap = max(0.0, center_distance - radii)  # lower bound; cannot overlap
            ttc = math.inf
        return ContactInfo(
            other=owner,
            kind=kind,
            overlapping=gap <= _EPS,
            gap_m=float(gap),
            center_distance_m=float(center_distance),
            bearing_rad=float(bearing),
            heading_delta_rad=float(_wrap(other.yaw - ego.yaw)),
            closing_speed_mps=float(closing),
            ttc_s=float(ttc),
            classification=classify_contact(ego, other, kind),
        )

    def within_proximity(self, contact: ContactInfo) -> bool:
        """Proximity is defined on centre distance (radius semantics of the unit contract)."""
        return contact.center_distance_m < self.proximity_radius_m
