"""Kinematically feasible route geometry for background actors (Stage 7, finding V-03).

Graph routes are polylines with ~4.3 m waypoint spacing, so a 90° junction is a
single vertex whose three-point radius (~2.5 m) is far below the bicycle's
minimum turning radius (``L / tan(max_steer)`` ≈ 5.0 m). No speed can make
such a corner feasible; the controller overshoots and the body sweeps through
the inner corner. :func:`fillet_route` replaces every sharp vertex by a
circular arc tangent to both legs, as large as the legs and the drivable mask
allow (the arc *cuts the inside* of the corner, so the inner edge of the
vehicle is checked against the mask), and densifies the path to a uniform
sample spacing with a proper curvature array for curvature-aware speed caps.

All quantities are metres / radians. The function is deterministic and makes
no random draw.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_SAMPLE_SPACING_M = 1.0
DEFAULT_MIN_TURN_RAD = math.radians(12.0)
CONNECTOR_MIN_TURN_RAD = math.radians(20.0)


class RouteGeometryError(ValueError):
    """A sampled path has invalid numerical coordinates."""


def sampled_path_geometry(cx_m, cy_m):
    """Coordinates, tangent heading, signed curvature and chord arc length.

    Circumcircle curvature is computed from neighbouring points, so unequal
    sample spacing does not bias a circular arc. Endpoints use the nearest
    three-point circle; one/two-point paths have zero curvature. Consecutive
    coincident points are removed before geometry is computed.
    """
    cx, cy = np.asarray(cx_m, dtype=float), np.asarray(cy_m, dtype=float)
    if cx.ndim != 1 or cy.ndim != 1 or len(cx) != len(cy) or not len(cx):
        raise RouteGeometryError("path must have matching, nonempty one-dimensional coordinates")
    if not np.isfinite(cx).all() or not np.isfinite(cy).all():
        raise RouteGeometryError("path coordinates must be finite")
    cx, cy = _dedupe(cx, cy)
    if len(cx) == 1:
        return cx, cy, np.zeros(1), np.zeros(1), np.zeros(1)
    segments = np.column_stack((np.diff(cx), np.diff(cy)))
    ds = np.hypot(segments[:, 0], segments[:, 1])
    headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    s = np.concatenate(([0.0], np.cumsum(ds)))
    if len(cx) == 2:
        return cx, cy, np.full(2, headings[0]), np.zeros(2), s
    before, after = segments[:-1], segments[1:]
    across = before + after
    chord = np.hypot(across[:, 0], across[:, 1])
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    denominator = ds[:-1] * ds[1:] * chord
    interior = np.divide(2.0 * cross, denominator, out=np.zeros_like(cross), where=chord > 1e-6)
    # A reversal has no finite osculating circle. Represent its sampled turn
    # rate rather than rejecting the navigation route or pretending it is straight.
    reversing = np.abs(np.diff(headings)) >= math.pi - 1e-6
    interior[reversing] = np.diff(headings)[reversing] / (0.5 * (ds[:-1] + ds[1:]))[reversing]
    ck = np.concatenate(([interior[0]], interior, [interior[-1]]))
    # Tangent at each point of its local circle; exact on a sampled circular arc.
    yaw = np.empty(len(cx))
    yaw[1:] = headings + np.arcsin(np.clip(ck[1:] * ds * 0.5, -1.0, 1.0))
    yaw[0] = headings[0] - math.asin(float(np.clip(ck[0] * ds[0] * 0.5, -1.0, 1.0)))
    return cx, cy, np.unwrap(yaw), ck, s


def _unit(vec):
    norm = float(np.hypot(vec[0], vec[1]))
    if norm <= 1e-9:
        return None, 0.0
    return vec / norm, norm


def _dedupe(ax, ay, eps=1e-6):
    xs, ys = [float(ax[0])], [float(ay[0])]
    for x, y in zip(ax[1:], ay[1:]):
        if math.hypot(float(x) - xs[-1], float(y) - ys[-1]) > eps:
            xs.append(float(x))
            ys.append(float(y))
    return np.asarray(xs), np.asarray(ys)


def _simplify(points, tol_m):
    """Drop vertices of straight runs (every dropped point within ``tol_m`` of the merged chord).

    Graph routes sample straight roads every ~4 m; a fillet needs legs of
    several metres on each side of a corner, so collinear runs are merged into
    single legs first (gentle curves with more than ``tol_m`` deviation are kept).
    """
    n = len(points)
    if n <= 2:
        return points
    keep = [0]
    anchor = 0
    for i in range(1, n - 1):
        chord = points[i + 1] - points[anchor]
        length = float(np.hypot(chord[0], chord[1]))
        if length <= 1e-9:
            continue
        inner = points[anchor + 1 : i + 1]
        rel = inner - points[anchor]
        deviation = np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0]) / length
        if deviation.max() <= tol_m:
            continue
        keep.append(i)
        anchor = i
    keep.append(n - 1)
    return points[keep]


def _merge_short_connectors(points, max_leg_m, max_deviation_m):
    """Collapse junction connector chains into the corner of the two approach legs.

    Graph junctions are encoded as two or three diagonal legs of 2-8 m between
    long road legs; no feasible fillet fits between such vertices. Each maximal
    run of short legs bounded by long legs on both sides is replaced by the
    intersection point of the two long legs' lines (both legs keep their lane
    alignment; the fillet then runs through the junction close to the original
    diagonal). Runs whose corner point would lie farther than
    ``max_deviation_m`` from the chain, nearly parallel legs, and runs at the
    route ends are kept as they are.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 4:
        return pts
    legs = pts[1:] - pts[:-1]
    lengths = np.hypot(legs[:, 0], legs[:, 1])
    short = lengths < max_leg_m
    out = [pts[0]]
    i = 0  # index of the last emitted vertex
    while i < n - 1:
        if not short[i]:
            out.append(pts[i + 1])
            i += 1
            continue
        # run of short legs i .. j-1 (vertices i .. j)
        j = i
        while j < n - 1 and short[j]:
            j += 1
        has_long_before = i >= 1 and not short[i - 1]
        has_long_after = j <= n - 2 and not short[j]
        replaced = False
        if has_long_before and has_long_after:
            p0, d1 = pts[i], legs[i - 1] / lengths[i - 1]
            q0, d2 = pts[j], legs[j] / lengths[j]
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(cross) > 0.2:
                # Intersection of p0 + a*d1 and q0 + b*d2.
                rel = q0 - p0
                a = (rel[0] * d2[1] - rel[1] * d2[0]) / cross
                corner = p0 + a * d1
                chain = pts[i : j + 1]
                deviation = float(np.min(np.hypot(chain[:, 0] - corner[0], chain[:, 1] - corner[1])))
                if a > -1e-6 and deviation <= max_deviation_m:
                    out.pop()  # the run's first vertex (end of the long leg) is replaced by the corner
                    out.append(corner)
                    replaced = True
        if not replaced:
            out.extend(pts[i + 1 : j + 1])
        i = j
    return np.asarray(out)


def _arc_points(center, radius, start_angle, sweep, spacing):
    """Points along an arc (excluding the start point, including the end point)."""
    n = max(1, int(math.ceil(abs(sweep) * radius / spacing)))
    angles = start_angle + sweep * np.arange(1, n + 1) / n
    return np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))


def _line_points(p0, p1, spacing):
    """Points from p0 (exclusive) to p1 (inclusive) at most ``spacing`` apart."""
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if length <= 1e-9:
        return np.empty((0, 2))
    n = max(1, int(math.ceil(length / spacing)))
    t = np.arange(1, n + 1) / n
    return np.column_stack((p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])))


def fillet_route(
    ax_m,
    ay_m,
    *,
    target_radius_m: float,
    min_radius_m: float,
    sample_spacing_m: float = DEFAULT_SAMPLE_SPACING_M,
    inner_clearance_m: float = 0.0,
    is_drivable=None,
    min_turn_rad: float = DEFAULT_MIN_TURN_RAD,
    simplify_tol_m: float = 0.35,
    connector_max_leg_m: float = 12.0,
    connector_max_deviation_m: float = 6.0,
):
    """Fillet a polyline and resample it.

    Parameters
    ----------
    target_radius_m, min_radius_m
        Preferred / smallest acceptable fillet radius. The radius is reduced
        (never below ``min_radius_m``) until the arc fits the adjacent legs and
        — when ``is_drivable(x_m, y_m)`` is given — its inner offset by
        ``inner_clearance_m`` (half the vehicle width) stays drivable. If no
        radius >= ``min_radius_m`` passes, the largest radius whose *centreline*
        is drivable is used; failing that the vertex is kept sharp.
    Returns
    -------
    (cx, cy, cyaw, ck, s): arrays in metres / radians / 1/m / metres.
    """
    ax, ay = _dedupe(np.asarray(ax_m, dtype=float), np.asarray(ay_m, dtype=float))
    if len(ax) == 1:
        return ax.copy(), ay.copy(), np.zeros(1), np.zeros(1), np.zeros(1)
    points = _simplify(np.column_stack((ax, ay)), simplify_tol_m)
    points = _merge_short_connectors(points, connector_max_leg_m, connector_max_deviation_m)
    ax, ay = points[:, 0], points[:, 1]
    n = len(ax)
    spacing = float(sample_spacing_m)

    out = [points[0]]
    curv = [0.0]
    cursor = points[0]  # last emitted point (start of the next straight piece)
    fillets = []  # (entry point, exit point, center, radius, start angle, sweep, sign)
    sharp_vertices = {}  # vertex index -> turn sign, for corners where no feasible fillet fits
    # Lengths of the legs and how much of each leg is still free for a fillet.
    seg_len = np.hypot(np.diff(ax), np.diff(ay))
    free_before = seg_len.copy()  # free length at the end of segment i (towards vertex i+1)

    for i in range(1, n - 1):
        d_in, len_in = _unit(points[i] - points[i - 1])
        d_out, len_out = _unit(points[i + 1] - points[i])
        if d_in is None or d_out is None:
            continue
        cos_turn = float(np.clip(np.dot(d_in, d_out), -1.0, 1.0))
        turn = math.acos(cos_turn)
        if turn < min_turn_rad or turn > math.pi - 1e-6:
            continue  # straight enough, or a U-turn (no fillet possible)
        sign = 1.0 if (d_in[0] * d_out[1] - d_in[1] * d_out[0]) > 0.0 else -1.0  # +1 = left turn
        tan_half = math.tan(turn / 2.0)
        # Tangent length available on each leg: half of an interior leg (shared
        # with the neighbouring vertex), the whole leg at the route ends.
        avail_in = free_before[i - 1] * (1.0 if i == 1 else 0.5)
        avail_out = seg_len[i] * (1.0 if i == n - 2 else 0.5)
        t_max = min(avail_in, avail_out)
        radius_cap = t_max / tan_half if tan_half > 1e-9 else 0.0
        candidates = []
        radius = min(float(target_radius_m), radius_cap)
        while radius >= float(min_radius_m) - 1e-9:
            candidates.append(radius)
            radius *= 0.85
        # No feasible radius fits: keep the vertex sharp (never emit an infeasible tiny arc).

        chosen = None
        fallback = None
        for radius in candidates:
            t = radius * tan_half
            entry = points[i] - d_in * t
            exit_ = points[i] + d_out * t
            normal_in = np.array([-d_in[1], d_in[0]]) * sign  # towards the inside of the turn
            center = entry + normal_in * radius
            start_angle = math.atan2(entry[1] - center[1], entry[0] - center[0])
            sweep = sign * turn
            if is_drivable is None:
                chosen = (entry, exit_, center, radius, start_angle, sweep, sign)
                break
            arc = _arc_points(center, radius, start_angle, sweep, spacing)
            centre_ok = all(is_drivable(float(x), float(y)) for x, y in arc)
            if not centre_ok:
                continue
            if inner_clearance_m > 0.0 and radius > inner_clearance_m:
                inner = _arc_points(center, radius - inner_clearance_m, start_angle, sweep, spacing)
                if not all(is_drivable(float(x), float(y)) for x, y in inner):
                    if fallback is None:
                        fallback = (entry, exit_, center, radius, start_angle, sweep, sign)
                    continue
            chosen = (entry, exit_, center, radius, start_angle, sweep, sign)
            break
        if chosen is None:
            chosen = fallback
        if chosen is None:
            sharp_vertices[i] = sign  # kept sharp; flagged with the turn direction for the speed cap
            continue
        entry, exit_, center, radius, start_angle, sweep, sign = chosen
        fillets.append((i, chosen))
        free_before[i] = seg_len[i] - radius * tan_half  # consumed on the outgoing leg

    # --- emit the path: straights between fillets, arcs at fillets -------------
    fillet_by_vertex = {i: f for i, f in fillets}
    for i in range(1, n):
        if i in fillet_by_vertex:
            entry, exit_, center, radius, start_angle, sweep, sign = fillet_by_vertex[i]
            straight = _line_points(cursor, entry, spacing)
            out.extend(straight)
            curv.extend([0.0] * len(straight))
            arc = _arc_points(center, radius, start_angle, sweep, spacing)
            out.extend(arc)
            curv.extend([sign / radius] * len(arc))
            cursor = exit_
        else:
            # Sharp vertex (no fillet fits) or the route end: straight to the vertex itself.
            straight = _line_points(cursor, points[i], spacing)
            out.extend(straight)
            curv.extend([0.0] * len(straight))
            if i in sharp_vertices and len(straight):
                # An unfilleted corner is at least as tight as the minimum turning
                # radius; flag it so the curvature speed cap slows the approach.
                curv[-1] = sharp_vertices[i] / float(min_radius_m)
            cursor = points[i]

    path = np.asarray(out, dtype=float)
    ck = np.asarray(curv, dtype=float)
    cx, cy = path[:, 0], path[:, 1]
    dx, dy = np.gradient(cx), np.gradient(cy)
    # Snap rounding residues so straight legs get a deterministic heading (P-01).
    dx = np.where(np.abs(dx) < 1e-9, 0.0, dx)
    dy = np.where(np.abs(dy) < 1e-9, 0.0, dy)
    cyaw = np.arctan2(dy, dx)
    seg = np.hypot(np.diff(cx), np.diff(cy))
    s = np.concatenate(([0.0], np.cumsum(seg)))
    return cx, cy, cyaw, ck, s


def nudge_from_blocking(cx_m, cy_m, clearance_px, meters_per_pixel, *, target_clearance_m, max_shift_m, is_drivable=None, window=2):
    """Shift path points that run closer than ``target_clearance_m`` to blocking pixels away from them.

    ``clearance_px`` is the Euclidean distance transform of the non-blocking
    pixels (distance of every pixel to the nearest blocking pixel, in px). The
    shift follows the transform's gradient, is capped at ``max_shift_m``, must
    land on a drivable pixel when ``is_drivable`` is given, and is smoothed
    over ``±window`` points so the path stays kink-free. At 128 px the lane
    centrelines of a few road segments lie 0.94 m from a building — less than
    half a car width — so the body clipped the building even with perfect
    tracking.
    """
    cx = np.asarray(cx_m, dtype=float).copy()
    cy = np.asarray(cy_m, dtype=float).copy()
    height, width = clearance_px.shape
    inv = 1.0 / meters_per_pixel

    def sample(xs, ys):
        # Bilinear sample of the transform at continuous pixel coordinates
        # (pixel-centre granularity would under-read the clearance by up to half a pixel).
        x0 = np.clip(np.floor(xs).astype(int), 1, width - 3)
        y0 = np.clip(np.floor(ys).astype(int), 1, height - 3)
        fx = np.clip(xs - x0, 0.0, 1.0)
        fy = np.clip(ys - y0, 0.0, 1.0)
        return (
            clearance_px[y0, x0] * (1 - fx) * (1 - fy)
            + clearance_px[y0, x0 + 1] * fx * (1 - fy)
            + clearance_px[y0 + 1, x0] * (1 - fx) * fy
            + clearance_px[y0 + 1, x0 + 1] * fx * fy
        )

    xs, ys = cx * inv, cy * inv
    clearance_m = sample(xs, ys) * meters_per_pixel
    gx = (sample(xs + 1.0, ys) - sample(xs - 1.0, ys)) * 0.5
    gy = (sample(xs, ys + 1.0) - sample(xs, ys - 1.0)) * 0.5
    norm = np.hypot(gx, gy)
    need = np.clip(target_clearance_m - clearance_m, 0.0, max_shift_m)
    need[norm <= 1e-9] = 0.0
    norm[norm <= 1e-9] = 1.0
    shift_x = need * gx / norm
    shift_y = need * gy / norm
    if window > 0 and len(cx) > 2 * window + 1:
        kernel = np.ones(2 * window + 1) / (2 * window + 1)
        shift_x = np.convolve(shift_x, kernel, mode="same")
        shift_y = np.convolve(shift_y, kernel, mode="same")
    moving = np.flatnonzero((np.abs(shift_x) >= 1e-9) | (np.abs(shift_y) >= 1e-9))
    if len(moving) == 0:
        return cx, cy
    if is_drivable is None:
        cx[moving] += shift_x[moving]
        cy[moving] += shift_y[moving]
        return cx, cy
    # Shifted points must stay drivable; halve the shift up to three times otherwise.
    pending = moving
    sx, sy = shift_x[moving].copy(), shift_y[moving].copy()
    for _ in range(3):
        ok = np.array([is_drivable(float(cx[i] + dx), float(cy[i] + dy)) for i, dx, dy in zip(pending, sx, sy)], dtype=bool)
        cx[pending[ok]] += sx[ok]
        cy[pending[ok]] += sy[ok]
        pending, sx, sy = pending[~ok], sx[~ok] * 0.5, sy[~ok] * 0.5
        if len(pending) == 0:
            break
    return cx, cy


def route_has_uturn(
    ax_m, ay_m, *, max_turn_rad: float = math.radians(150.0), window_m: float = 12.0, simplify_tol_m: float = 0.35
) -> bool:
    """True if the route reverses direction within a short distance (dead-end turnarounds in the graph).

    The heading change is accumulated over consecutive vertices joined by legs
    shorter than ``window_m`` (a turnaround is typically two ~90 deg turns a few
    metres apart). Such connectors demand a turning radius far below a car's
    minimum; background actors are not given these routes (generation policy
    v3, lifecycle respawns).
    """
    ax, ay = _dedupe(np.asarray(ax_m, dtype=float), np.asarray(ay_m, dtype=float))
    if len(ax) < 3:
        return False
    pts = _simplify(np.column_stack((ax, ay)), simplify_tol_m)
    turns = []
    for i in range(1, len(pts) - 1):
        d_in, len_in = _unit(pts[i] - pts[i - 1])
        d_out, len_out = _unit(pts[i + 1] - pts[i])
        if d_in is None or d_out is None:
            turns.append((0.0, len_out))
            continue
        turns.append((math.acos(float(np.clip(np.dot(d_in, d_out), -1.0, 1.0))), len_out))
    for i in range(len(turns)):
        total = turns[i][0]
        if total >= max_turn_rad:
            return True
        j = i
        while j + 1 < len(turns) and turns[j][1] < window_m:
            total += turns[j + 1][0]
            if total >= max_turn_rad:
                return True
            j += 1
    return False


def curvature_speed_profile(ck, s, *, lat_accel_mps2: float, decel_mps2: float, v_max_mps: float = 1e9):
    """Speed cap per path point: lateral-acceleration limit on curves, propagated backwards by ``decel``.

    Returns an array (same length as ``ck``); ``inf``-free (capped by ``v_max_mps``).
    """
    ck = np.asarray(ck, dtype=float)
    s = np.asarray(s, dtype=float)
    with np.errstate(divide="ignore"):
        cap = np.where(np.abs(ck) > 1e-9, np.sqrt(lat_accel_mps2 / np.maximum(np.abs(ck), 1e-9)), v_max_mps)
    cap = np.minimum(cap, v_max_mps)
    for i in range(len(cap) - 2, -1, -1):
        ds = max(float(s[i + 1] - s[i]), 0.0)
        cap[i] = min(cap[i], math.sqrt(cap[i + 1] ** 2 + 2.0 * decel_mps2 * ds))
    return cap
